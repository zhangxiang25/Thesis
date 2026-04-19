import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import ALGO_COLORS, apply_publication_style, finish_and_save, style_axis


OUTPUT_PATH = os.path.join("outputs", "final_total_wait_comparison.png")
MAX_EPISODES = 20


def extract_episode(path):
    match = re.search(r"_ep(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else None


def latest_matching_files(pattern):
    files = glob.glob(pattern)
    if not files:
        return []

    grouped = {}
    for path in files:
        match = re.search(r"_conn(\d+)_ep\d+\.csv$", os.path.basename(path))
        conn_id = int(match.group(1)) if match else -1
        grouped.setdefault(conn_id, []).append(path)

    _, selected = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    return sorted(selected, key=lambda path: extract_episode(path) or 10**9)


def collect_total_wait_series(files, label):
    rows = []
    for path in files:
        episode = extract_episode(path)
        if episode is None or episode > MAX_EPISODES:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Failed to read {path}: {exc}")
            continue

        if "system_total_waiting_time" not in df.columns:
            print(f"Missing system_total_waiting_time in {path}")
            continue

        rows.append(
            {
                "Episode": episode,
                "Value": float(df["system_total_waiting_time"].mean()),
                "Algorithm": label,
            }
        )
    return pd.DataFrame(rows)


def main():
    ql_files = sorted(glob.glob(os.path.join("outputs", "4x4", "ql-4x4grid_run1_conn0_ep*.csv")))
    ppo_files = latest_matching_files(os.path.join("outputs", "4x4grid", "ppo_test_final_conn*_ep*.csv"))
    fixed_time_files = latest_matching_files(os.path.join("outputs", "4x4grid", "fixedtime_conn*_ep*.csv"))

    df_ql = collect_total_wait_series(ql_files, "QL")
    df_ppo = collect_total_wait_series(ppo_files, "PPO")
    df_fixed = collect_total_wait_series(fixed_time_files, "Fixed-Time")
    df_all = pd.concat([df_ql, df_ppo, df_fixed], ignore_index=True)

    if df_all.empty:
        print("No episode CSV files were found for QL, PPO, or Fixed-Time.")
        return

    apply_publication_style()
    fig, ax = plt.subplots()
    for algo in sorted(df_all["Algorithm"].unique()):
        data = df_all[df_all["Algorithm"] == algo].sort_values("Episode")
        ax.plot(
            data["Episode"],
            data["Value"],
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=ALGO_COLORS.get(algo),
            label=algo,
        )

    style_axis(
        ax,
        title="Time-Averaged Total Waiting Time: Algorithm Comparison",
        ylabel="Time-Averaged Total Waiting Time (s)",
    )
    ax.legend(title="Algorithm")
    finish_and_save(fig, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
