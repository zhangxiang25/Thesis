import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import NEUTRAL_COLORS, apply_publication_style, finish_and_save, style_axis


RUN_ID = 1
MAX_EPISODES = 20
OUTPUT_FOLDER = "outputs/4x4"
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")


def plot_reward_curve():
    apply_publication_style()
    episode_rewards = []
    episode_numbers = []

    print(f"Reading data for Run {RUN_ID}...")

    for ep in range(1, MAX_EPISODES + 1):
        file_path = os.path.join(OUTPUT_FOLDER, f"custom_metrics_run{RUN_ID}_ep{ep}.csv")

        if not os.path.exists(file_path):
            print(f"Warning: file not found, skipping: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            reward_cols = [c for c in df.columns if "_reward" in c]
            total_reward = df[reward_cols].sum().sum()

            episode_rewards.append(total_reward)
            episode_numbers.append(ep)
        except Exception as exc:
            print(f"Failed to read {file_path}: {exc}")

    if not episode_rewards:
        print("No episode data found. Please check the output folder.")
        return

    fig, ax = plt.subplots()
    ax.plot(
        episode_numbers,
        episode_rewards,
        alpha=0.3,
        color=NEUTRAL_COLORS["raw"],
        label="Raw Episode Reward",
    )

    if len(episode_rewards) > 5:
        window_size = 5
        smoothed_rewards = pd.Series(episode_rewards).rolling(window=window_size).mean()
        ax.plot(
            episode_numbers,
            smoothed_rewards,
            color=NEUTRAL_COLORS["smooth"],
            linewidth=2.4,
            label=f"Moving Avg (Window={window_size})",
        )

    style_axis(
        ax,
        title=f"QL Learning Curve - Run {RUN_ID}",
        ylabel="Total Cumulative Reward",
    )
    ax.legend(title=None)

    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    save_path = os.path.join(FIGURE_FOLDER, f"learning_curve_run{RUN_ID}.png")
    finish_and_save(fig, save_path)
    print(f"Learning curve saved to: {save_path}")


if __name__ == "__main__":
    plot_reward_curve()
