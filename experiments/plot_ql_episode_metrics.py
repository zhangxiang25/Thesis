import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import ALGO_COLORS, METRIC_LABELS, apply_publication_style, finish_and_save, style_axis


RUN_IDS = [1, 2, 3]
MAX_EPISODES = 20
INPUT_FOLDER = "outputs/4x4"
PLOT_FOLDER = os.path.join(INPUT_FOLDER, "plots")


def collect_episode_metrics():
    rows = []

    for run_id in RUN_IDS:
        for ep in range(1, MAX_EPISODES + 1):
            file_path = os.path.join(INPUT_FOLDER, f"ql-4x4grid_run{run_id}_conn0_ep{ep}.csv")
            if not os.path.exists(file_path):
                print(f"Warning: file not found, skipping: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)
                rows.append(
                    {
                        "run_id": run_id,
                        "episode": ep,
                        "mean_waiting_time": df["system_mean_waiting_time"].mean(),
                        "total_waiting_time": df["system_total_waiting_time"].mean(),
                        "mean_speed": df["system_mean_speed"].mean(),
                        "stopped_ratio": (
                            (df["system_total_stopped"] / df["system_total_running"].replace(0, pd.NA))
                            .fillna(0.0)
                            .mean()
                        ),
                    }
                )
            except Exception as exc:
                print(f"Failed to read {file_path}: {exc}")

    return pd.DataFrame(rows)


def save_line_plot(df, metric, ylabel, filename):
    apply_publication_style()
    fig, ax = plt.subplots()

    run_colors = {
        1: "#8fbce6",
        2: "#5fa8e8",
        3: "#2f7fd1",
    }
    for run_id in RUN_IDS:
        run_df = df[df["run_id"] == run_id].sort_values("episode")
        if run_df.empty:
            continue
        ax.plot(
            run_df["episode"],
            run_df[metric],
            marker="o",
            markersize=3.8,
            linewidth=1.5,
            alpha=0.8,
            color=run_colors.get(run_id, "#5fa8e8"),
            label=f"Run {run_id}",
        )

    summary = (
        df.groupby("episode", as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
    )
    summary["std"] = summary["std"].fillna(0.0)

    x = summary["episode"].to_numpy(dtype=float)
    mean_y = summary["mean"].to_numpy(dtype=float)
    std_y = summary["std"].to_numpy(dtype=float)

    ax.plot(
        x,
        mean_y,
        marker="o",
        markersize=4.6,
        linewidth=2.4,
        color=ALGO_COLORS["QL"],
        label="Across-Run Mean",
    )
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=ALGO_COLORS["QL"], alpha=0.16)

    style_axis(ax, title=f"QL {ylabel} Across Runs", ylabel=ylabel)
    ax.legend(title=None)

    save_path = os.path.join(PLOT_FOLDER, filename)
    finish_and_save(fig, save_path)
    print(f"Saved: {save_path}")


def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    df = collect_episode_metrics()

    if df.empty:
        print("No QL episode CSV files were found.")
        return

    summary_path = os.path.join(PLOT_FOLDER, "ql_metrics_runs1_2_3.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    save_line_plot(
        df,
        metric="mean_waiting_time",
        ylabel=METRIC_LABELS["mean_waiting_time"],
        filename="ql_mean_waiting_time_runs1_2_3.png",
    )
    save_line_plot(
        df,
        metric="stopped_ratio",
        ylabel=METRIC_LABELS["stopped_ratio"],
        filename="ql_stopped_ratio_runs1_2_3.png",
    )
    save_line_plot(
        df,
        metric="mean_speed",
        ylabel=METRIC_LABELS["mean_speed"],
        filename="ql_mean_speed_runs1_2_3.png",
    )


if __name__ == "__main__":
    main()
