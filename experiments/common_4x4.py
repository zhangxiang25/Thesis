import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NET_FILE = os.path.join(ROOT, "sumo_rl", "nets", "4x4-Lucas", "4x4.net.xml")
ROUTE_FILE = os.path.join(ROOT, "sumo_rl", "nets", "4x4-Lucas", "4x4c1c2c1c2.rou.xml")
OUTPUT_4X4_DIR = os.path.join(ROOT, "outputs", "4x4")
OUTPUT_4X4GRID_DIR = os.path.join(ROOT, "outputs", "4x4grid")

# 4x4 grid environment shared settings for PPO and Q-Learning comparison
COMMON_ENV_KWARGS = {
    "use_gui": False,
    "num_seconds": 20000,
    "reward_fn": None,  # set after function definition
    "enforce_max_green": True,
    "min_green": 5,
    "delta_time": 5,
}


def custom_combined_reward(traffic_signal):
    """Shared 4x4 grid reward function for fair PPO vs Q-Learning comparison."""
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


COMMON_ENV_KWARGS["reward_fn"] = custom_combined_reward


def build_env_kwargs(**overrides):
    """Build a shared 4x4 environment config for fair method comparisons."""
    kwargs = dict(COMMON_ENV_KWARGS)
    kwargs.update(overrides)
    return kwargs
