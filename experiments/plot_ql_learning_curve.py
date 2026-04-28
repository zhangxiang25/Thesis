import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import ALGO_COLORS, apply_publication_style, finish_and_save, style_axis


RUN_IDS = [1, 2, 3]
MAX_EPISODES = 20
OUTPUT_FOLDER = "outputs/4x4"
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")


def plot_reward_curve():
    apply_publication_style()
    run_colors = {
        1: "#8fbce6",
        2: "#5fa8e8",
        3: "#2f7fd1",
    }
    all_rows = []

    for run_id in RUN_IDS:
        print(f"Reading data for Run {run_id}...")
        for ep in range(1, MAX_EPISODES + 1):
            file_path = os.path.join(OUTPUT_FOLDER, f"custom_metrics_run{run_id}_ep{ep}.csv")

            if not os.path.exists(file_path):
                print(f"Warning: file not found, skipping: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)
                reward_cols = [c for c in df.columns if "_reward" in c]
                total_reward = df[reward_cols].sum().sum()
                all_rows.append(
                    {
                        "run_id": run_id,
                        "episode": ep,
                        "total_reward": total_reward,
                    }
                )
            except Exception as exc:
                print(f"Failed to read {file_path}: {exc}")

    if not all_rows:
        print("No episode data found. Please check the output folder.")
        return

    rewards_df = pd.DataFrame(all_rows)

    fig, ax = plt.subplots()
    for run_id in RUN_IDS:
        run_df = rewards_df[rewards_df["run_id"] == run_id].sort_values("episode")
        if run_df.empty:
            continue
        ax.plot(
            run_df["episode"],
            run_df["total_reward"],
            alpha=0.75,
            linewidth=1.6,
            color=run_colors.get(run_id, "#5fa8e8"),
            label=f"Run {run_id}",
        )

    mean_rewards = (
        rewards_df.groupby("episode", as_index=False)["total_reward"]
        .mean()
        .sort_values("episode")
    )
    if len(mean_rewards) > 5:
        window_size = 5
        smoothed_mean = mean_rewards["total_reward"].rolling(window=window_size).mean()
        ax.plot(
            mean_rewards["episode"],
            smoothed_mean,
            color=ALGO_COLORS["QL"],
            linewidth=2.6,
            label=f"Run Mean Moving Avg (Window={window_size})",
        )

    style_axis(
        ax,
        title="QL Learning Curves Across Runs",
        ylabel="Total Cumulative Reward",
    )
    ax.legend(title=None)

    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    save_path = os.path.join(FIGURE_FOLDER, "learning_curve_runs1_2_3.png")
    finish_and_save(fig, save_path)
    print(f"Learning curve saved to: {save_path}")


if __name__ == "__main__":
    plot_reward_curve()
