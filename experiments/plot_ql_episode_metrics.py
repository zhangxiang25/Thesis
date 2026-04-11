import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import ALGO_COLORS, METRIC_LABELS, apply_publication_style, finish_and_save, style_axis


RUN_ID = 1
MAX_EPISODES = 20
INPUT_FOLDER = "outputs/4x4"
PLOT_FOLDER = os.path.join(INPUT_FOLDER, "plots")


def collect_episode_metrics():
    rows = []

    for ep in range(1, MAX_EPISODES + 1):
        file_path = os.path.join(INPUT_FOLDER, f"ql-4x4grid_run{RUN_ID}_conn0_ep{ep}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: file not found, skipping: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            rows.append(
                {
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
    ax.plot(
        df["episode"],
        df[metric],
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        color=ALGO_COLORS["QL"],
    )
    style_axis(ax, title=f"QL {ylabel} by Episode - Run {RUN_ID}", ylabel=ylabel)

    save_path = os.path.join(PLOT_FOLDER, filename)
    finish_and_save(fig, save_path)
    print(f"Saved: {save_path}")


def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    df = collect_episode_metrics()

    if df.empty:
        print("No QL episode CSV files were found.")
        return

    summary_path = os.path.join(PLOT_FOLDER, f"ql_metrics_run{RUN_ID}.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    save_line_plot(
        df,
        metric="mean_waiting_time",
        ylabel=METRIC_LABELS["mean_waiting_time"],
        filename=f"ql_mean_waiting_time_run{RUN_ID}.png",
    )
    save_line_plot(
        df,
        metric="stopped_ratio",
        ylabel=METRIC_LABELS["stopped_ratio"],
        filename=f"ql_stopped_ratio_run{RUN_ID}.png",
    )
    save_line_plot(
        df,
        metric="mean_speed",
        ylabel=METRIC_LABELS["mean_speed"],
        filename=f"ql_mean_speed_run{RUN_ID}.png",
    )


if __name__ == "__main__":
    main()
