import argparse
import glob
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")

from common_4x4 import NET_FILE, ROUTE_FILE, OUTPUT_4X4_DIR, build_env_kwargs
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


def clean_old_outputs(run_id: int):
    patterns = [
        os.path.join(OUTPUT_4X4_DIR, f"ql-4x4grid_run{run_id}_conn*_ep*.csv"),
        os.path.join(OUTPUT_4X4_DIR, f"custom_metrics_run{run_id}_ep*.csv"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning on the shared 4x4 SUMO setup.")
    parser.add_argument("--run-id", type=int, default=1, help="Logical run id used in output filenames.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="If set, delete existing CSV outputs for the selected run id before training.",
    )
    args = parser.parse_args()

    alpha = 0.1
    gamma = 0.99
    decay = 0.98

    if args.clean:
        clean_old_outputs(args.run_id)

    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        **build_env_kwargs(),
    )

    # Reset the environment and get the initial states
    initial_states = env.reset()

    # Create one Q-learning agent for each traffic signal
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=0.05,
                min_epsilon=0.005,
                decay=decay
            ),
        )
        for ts in env.ts_ids
    }

    for episode in range(1, args.episodes + 1):
        # List used to store custom step-level metrics for this episode
        custom_metrics = []

        # Reset the environment at the beginning of each episode,
        # except for episode 1 which already started from the initial reset
        if episode != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)

        done = {"__all__": False}
        step_counter = 0  # Track the current simulation step

        while not done["__all__"]:
            # Let each agent choose an action
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            # Apply the actions to the environment
            s, r, done, info = env.step(action=actions)

            # Collect custom traffic metrics at the current step
            current_step_data = {"step": step_counter}

            # Iterate through each traffic signal (agent)
            for ts_id in env.ts_ids:
                # Get the traffic signal object
                traffic_signal = env.traffic_signals[ts_id]

                # Compute the total number of vehicles and halted vehicles
                # across all lanes controlled by this traffic signal
                total_vehicles = 0
                stopped_vehicles = 0
                for lane in traffic_signal.lanes:
                    total_vehicles += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
                    stopped_vehicles += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

                # Compute stop ratio and avoid division by zero
                stop_ratio = stopped_vehicles / total_vehicles if total_vehicles > 0 else 0.0

                # Save the metrics into the dictionary
                # Example column names:
                # '0_total_vehicles', '0_stopped', '0_stop_ratio', '0_reward'
                current_step_data[f"{ts_id}_total_vehicles"] = total_vehicles
                current_step_data[f"{ts_id}_stopped"] = stopped_vehicles
                current_step_data[f"{ts_id}_stop_ratio"] = stop_ratio
                current_step_data[f"{ts_id}_reward"] = r[ts_id]

            # Append the current step data to the episode list
            custom_metrics.append(current_step_data)
            step_counter += 1

            # Update each agent using the observed reward and next state
            for agent_id in s.keys():
                ql_agents[agent_id].learn(
                    next_state=env.encode(s[agent_id], agent_id),
                    reward=r[agent_id]
                )

        # Save the default SUMO-RL output CSV for this episode
        env.save_csv(os.path.join(OUTPUT_4X4_DIR, f"ql-4x4grid_run{args.run_id}"), episode)

        # Save the custom collected metrics to a separate CSV file
        df_custom = pd.DataFrame(custom_metrics)
        custom_path = os.path.join(OUTPUT_4X4_DIR, f"custom_metrics_run{args.run_id}_ep{episode}.csv")
        df_custom.to_csv(custom_path, index=False)
        print(f"Custom data saved: {custom_path}")

    env.close()
