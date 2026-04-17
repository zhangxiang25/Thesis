import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from common_4x4 import NET_FILE, ROUTE_FILE, build_env_kwargs

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

import sumo_rl

if __name__ == "__main__":
    # Use:
    # ray[rllib]==2.7.0
    # numpy == 1.23.4
    # Pillow>=9.4.0
    # ray[rllib]==2.7.0
    # SuperSuit>=3.9.0
    # torch>=1.13.1
    # tensorflow-probability>=0.19.0
    ray.init()

    env_name = "4x4grid"
    results_dir = os.path.abspath(os.path.join("ray_results", env_name))
    os.makedirs(results_dir, exist_ok=True)

    def make_env(out_csv_name="outputs/4x4grid/ppo"):
        return ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                **build_env_kwargs(out_csv_name=out_csv_name),
            )
        )

    # RLlib 2.10 needs the per-agent spaces for a shared multi-agent policy.
    space_env = make_env()
    first_agent = sorted(space_env.par_env.possible_agents)[0]
    obs_space = space_env.observation_space.spaces[first_agent]
    action_space = space_env.action_space.spaces[first_agent]
    space_env.close()

    register_env(
        env_name,
        lambda _: make_env(),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=0, rollout_fragment_length=400,batch_mode="complete_episodes",)
        .training(
            train_batch_size=4000,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .multi_agent(
            policies={
                "default_policy": PolicySpec(
                    observation_space=obs_space,
                    action_space=action_space,
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
            policies_to_train=["default_policy"],
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"episodes_total": 20},
        checkpoint_freq=5,
        storage_path=results_dir,
        config=config.to_dict(),
    )
