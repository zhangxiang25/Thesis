# Evaluate a trained PPO policy on the shared 4x4 grid setup.
# A saved warmup episode is produced first, followed by official evaluation episodes.

import argparse
import glob
import os
import sys

import pandas as pd

# SUMO tools
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl


def custom_combined_reward(traffic_signal):
    lanes = traffic_signal.lanes
    total_stopped = 0
    total_vehicles = 0
    total_wait = 0

    for lane in lanes:
        total_stopped += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
        total_vehicles += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
        total_wait += traffic_signal.sumo.lane.getWaitingTime(lane)

    if total_vehicles == 0:
        return 0.0

    stop_ratio = total_stopped / total_vehicles
    avg_wait = total_wait / total_vehicles

    MAX_WAIT_THRESHOLD = 50.0
    normalized_wait = min(avg_wait, MAX_WAIT_THRESHOLD) / MAX_WAIT_THRESHOLD

    w_ratio = 0.9
    w_wait = 0.1
    return -1.0 * ((w_ratio * stop_ratio) + (w_wait * normalized_wait))


def find_latest_checkpoint(ray_results_dir: str) -> str:
    ckpts = glob.glob(os.path.join(ray_results_dir, "**", "checkpoint_*"), recursive=True)
    ckpts = [c for c in ckpts if os.path.isdir(c)]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_* found under: {ray_results_dir}")
    ckpts.sort(key=lambda p: os.path.getmtime(p))
    return ckpts[-1]


def run_one_episode(env: ParallelPettingZooEnv, algo: Algorithm):
    obs, _ = env.reset()
    done_all = False
    while not done_all:
        actions = {}
        for agent_id, ob in obs.items():
            actions[agent_id] = algo.compute_single_action(
                observation=ob,
                policy_id="default_policy",
                explore=False,  # deterministic eval
            )
        obs, rew, terminated, truncated, info = env.step(actions)
        done_all = bool(terminated.get("__all__", False) or truncated.get("__all__", False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ray_results", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)  # number of OFFICIAL eval episodes
    parser.add_argument("--num_seconds", type=int, default=20000)
    parser.add_argument("--delta_time", type=int, default=5)
    parser.add_argument("--min_green", type=int, default=5)
    parser.add_argument("--use_gui", action="store_true", default=False)
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="If set, delete old warmup+official outputs before running.",
    )
    args = parser.parse_args()

    # Absolute paths
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    NET = os.path.join(ROOT, "sumo_rl", "nets", "4x4-Lucas", "4x4.net.xml")
    ROUTE = os.path.join(ROOT, "sumo_rl", "nets", "4x4-Lucas", "4x4c1c2c1c2.rou.xml")

    OUT_DIR = os.path.join(ROOT, "outputs", "4x4grid")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ✅ Warmup prefix (SAVED)
    warm_prefix = os.path.join(OUT_DIR, "ppo_warmup")
    # ✅ Official prefix (ep1..epN)
    official_prefix = os.path.join(OUT_DIR, "ppo_test_final")

    if not os.path.exists(NET):
        raise FileNotFoundError(f"NET not found: {NET}")
    if not os.path.exists(ROUTE):
        raise FileNotFoundError(f"ROUTE not found: {ROUTE}")

    # Checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.ray_results:
        ckpt_path = find_latest_checkpoint(args.ray_results)
    else:
        default_ray = os.path.join(os.path.expanduser("~"), "ray_results", "PPO")
        ckpt_path = find_latest_checkpoint(default_ray)

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    # Optional cleanup (now it deletes BOTH warmup and official if you pass --clean)
    if args.clean:
        for pfx in [warm_prefix, official_prefix]:
            for f in glob.glob(pfx + "_conn*_ep*.csv"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    print("ROOT =", ROOT)
    print("NET  =", NET)
    print("ROUTE=", ROUTE)
    print("WARM =", warm_prefix)
    print("OUT  =", official_prefix)
    print("CKPT =", ckpt_path)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Register env name used during training BEFORE restoring algo
    env_name = "4x4grid"

    def env_creator(_cfg):
        return ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=NET,
                route_file=ROUTE,
                out_csv_name=official_prefix,  # training env name registration uses official prefix
                use_gui=args.use_gui,
                num_seconds=args.num_seconds,
                reward_fn=custom_combined_reward,
                enforce_max_green=True,
                min_green=args.min_green,
                delta_time=args.delta_time,
            )
        )

    register_env(env_name, env_creator)

    algo = Algorithm.from_checkpoint(ckpt_path)

    # ---- 1) Warm-up env: SAVED to ppo_warmup_... ----
    warm_env = ParallelPettingZooEnv(
        sumo_rl.parallel_env(
            net_file=NET,
            route_file=ROUTE,
            out_csv_name=warm_prefix,
            use_gui=args.use_gui,
            num_seconds=args.num_seconds,
            reward_fn=custom_combined_reward,
            enforce_max_green=True,
            min_green=args.min_green,
            delta_time=args.delta_time,
        )
    )
    print("[warmup] running 1 warmup episode (will be saved)...")
    run_one_episode(warm_env, algo)
    try:
        warm_env.close()
    except Exception:
        pass
    print("[warmup] done. Warmup CSV saved with prefix:", warm_prefix)

    # ---- 2) Official evaluation env: SAVED to ppo_test_final_... ----
    eval_env = ParallelPettingZooEnv(
        sumo_rl.parallel_env(
            net_file=NET,
            route_file=ROUTE,
            out_csv_name=official_prefix,
            use_gui=args.use_gui,
            num_seconds=args.num_seconds,
            reward_fn=custom_combined_reward,
            enforce_max_green=True,
            min_green=args.min_green,
            delta_time=args.delta_time,
        )
    )

    for ep in range(1, args.episodes + 1):
        run_one_episode(eval_env, algo)
        print(f"[eval] finished official episode {ep}/{args.episodes}")

    try:
        eval_env.close()
    except Exception:
        pass
    ray.shutdown()

    print("Done.")
    print("Warmup pattern :", warm_prefix + "_conn*_ep*.csv")
    print("Official pattern:", official_prefix + "_conn*_ep*.csv")
    print("Folder:", OUT_DIR)


if __name__ == "__main__":
    main()
