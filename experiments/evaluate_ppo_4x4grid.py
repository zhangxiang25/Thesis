# Evaluate a trained PPO policy on the shared 4x4 grid setup.
# A saved warmup episode is produced first, followed by official evaluation episodes.

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

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
from common_4x4 import NET_FILE, ROUTE_FILE, OUTPUT_4X4GRID_DIR, build_env_kwargs


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
                explore=False,
            )
        obs, _, terminated, truncated, _ = env.step(actions)
        done_all = bool(terminated.get("__all__", False) or truncated.get("__all__", False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=1, help="Logical run id used in output filenames and default checkpoint lookup.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ray_results", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
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

    os.makedirs(OUTPUT_4X4GRID_DIR, exist_ok=True)
    warm_prefix = os.path.join(OUTPUT_4X4GRID_DIR, f"ppo_warmup_run{args.run_id}")
    official_prefix = os.path.join(OUTPUT_4X4GRID_DIR, f"ppo_test_final_run{args.run_id}")

    if not os.path.exists(NET_FILE):
        raise FileNotFoundError(f"NET not found: {NET_FILE}")
    if not os.path.exists(ROUTE_FILE):
        raise FileNotFoundError(f"ROUTE not found: {ROUTE_FILE}")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.ray_results:
        ckpt_path = find_latest_checkpoint(args.ray_results)
    else:
        default_ray = os.path.abspath(os.path.join("ray_results", f"4x4grid_run{args.run_id}"))
        ckpt_path = find_latest_checkpoint(default_ray)

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    if args.clean:
        for pfx in [warm_prefix, official_prefix]:
            for path in glob.glob(pfx + "_conn*_ep*.csv"):
                try:
                    os.remove(path)
                except OSError:
                    pass

    print("NET  =", NET_FILE)
    print("ROUTE=", ROUTE_FILE)
    print("WARM =", warm_prefix)
    print("OUT  =", official_prefix)
    print("CKPT =", ckpt_path)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # The restored checkpoint keeps the training env id in its saved RLlib config.
    # Re-register that exact name here so Algorithm.from_checkpoint can rebuild
    # the worker set before we create our dedicated evaluation env below.
    env_name = f"4x4grid_run{args.run_id}"

    def create_parallel_env(out_csv_name):
        return ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                **build_env_kwargs(
                    out_csv_name=out_csv_name,
                    use_gui=args.use_gui,
                    num_seconds=args.num_seconds,
                    min_green=args.min_green,
                    delta_time=args.delta_time,
                ),
            )
        )

    register_env(env_name, lambda _cfg: create_parallel_env(official_prefix))

    algo = Algorithm.from_checkpoint(ckpt_path)

    warm_env = create_parallel_env(warm_prefix)
    print("[warmup] running 1 warmup episode (will be saved)...")
    run_one_episode(warm_env, algo)
    try:
        warm_env.close()
    except Exception:
        pass
    print("[warmup] done. Warmup CSV saved with prefix:", warm_prefix)

    eval_env = create_parallel_env(official_prefix)
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
    print("Folder:", OUTPUT_4X4GRID_DIR)


if __name__ == "__main__":
    main()
