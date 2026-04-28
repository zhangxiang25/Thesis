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

from common_4x4 import NET_FILE, ROUTE_FILE, OUTPUT_4X4GRID_DIR, build_env_kwargs
from sumo_rl import SumoEnvironment


def clean_old_outputs(prefix):
    for path in glob.glob(prefix + "_conn*_ep*.csv"):
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the shared 4x4 SUMO setup with fixed-time control.")
    parser.add_argument("--run-id", type=int, default=1, help="Logical run id used in output filenames.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num_seconds", type=int, default=None)
    parser.add_argument("--delta_time", type=int, default=None)
    parser.add_argument("--min_green", type=int, default=None)
    parser.add_argument("--use_gui", action="store_true", default=False)
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="If set, delete old fixed-time CSV outputs before running.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_4X4GRID_DIR, exist_ok=True)
    out_prefix = os.path.join(OUTPUT_4X4GRID_DIR, f"fixedtime_run{args.run_id}")

    if args.clean:
        clean_old_outputs(out_prefix)

    env_overrides = {
        "out_csv_name": out_prefix,
        "fixed_ts": True,
        "use_gui": args.use_gui,
    }
    if args.num_seconds is not None:
        env_overrides["num_seconds"] = args.num_seconds
    if args.delta_time is not None:
        env_overrides["delta_time"] = args.delta_time
    if args.min_green is not None:
        env_overrides["min_green"] = args.min_green

    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        **build_env_kwargs(**env_overrides),
    )

    for episode in range(1, args.episodes + 1):
        env.reset()
        done = {"__all__": False}

        while not done["__all__"]:
            _, _, done, _ = env.step(action=None)

        print(f"[fixed-time] finished episode {episode}/{args.episodes}")

    env.close()
    env.save_csv(out_prefix, env.episode)

    print("Done.")
    print("NET  =", NET_FILE)
    print("ROUTE=", ROUTE_FILE)
    print("OUT  =", out_prefix + "_conn*_ep*.csv")
