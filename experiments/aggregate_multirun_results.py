import argparse
import glob
import os
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from plot_style import ALGO_COLORS, METRIC_LABELS, apply_publication_style, finish_and_save, style_axis


OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["mean_wait", "total_wait", "mean_speed", "stopped_ratio"]


def extract_ep(path: str):
    match = re.search(r"_ep(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else None


def latest_matching_files(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return []

    by_conn: Dict[int, List[str]] = {}
    for path in files:
        match = re.search(r"_conn(\d+)_ep\d+\.csv$", os.path.basename(path))
        conn_id = int(match.group(1)) if match else -1
        by_conn.setdefault(conn_id, []).append(path)

    _, selected = max(by_conn.items(), key=lambda item: (len(item[1]), item[0]))
    return sorted(selected, key=lambda path: extract_ep(path) or 10**9)


def is_full_episode_csv(csv_path: str, min_rows=3000, min_step_max=19995):
    try:
        df = pd.read_csv(csv_path, usecols=["step"])
        if df.empty:
            return False, -1.0, 0
        step_max = float(df["step"].max())
        rows = int(len(df))
        ok = (rows >= min_rows) and (step_max >= min_step_max)
        return ok, step_max, rows
    except Exception:
        return False, -1.0, 0


def summarize_episode(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    def mean_col(col):
        return float(df[col].mean()) if col in df.columns else np.nan

    if "system_total_running" in df.columns and "system_total_stopped" in df.columns:
        denom = df["system_total_running"].replace(0, np.nan)
        stopped_ratio = (df["system_total_stopped"] / denom).fillna(0.0)
        sr_mean = float(stopped_ratio.mean())
    else:
        sr_mean = np.nan

    return {
        "file": os.path.basename(csv_path),
        "ep": extract_ep(csv_path),
        "rows": int(len(df)),
        "step_min": float(df["step"].min()) if "step" in df.columns else np.nan,
        "step_max": float(df["step"].max()) if "step" in df.columns else np.nan,
        "mean_wait": mean_col("system_mean_waiting_time"),
        "total_wait": mean_col("system_total_waiting_time"),
        "mean_speed": mean_col("system_mean_speed"),
        "total_stopped": mean_col("system_total_stopped"),
        "stopped_ratio": sr_mean,
    }


def files_for_run(run_id: int):
    if run_id == 1:
        return {
            "QL": sorted(glob.glob(os.path.join("outputs", "4x4", "ql-4x4grid_run1_conn*_ep*.csv"))),
            "PPO": latest_matching_files(os.path.join("outputs", "4x4grid", "ppo_test_final_conn*_ep*.csv")),
            "Fixed-Time": latest_matching_files(os.path.join("outputs", "4x4grid", "fixedtime_conn*_ep*.csv")),
        }

    return {
        "QL": sorted(glob.glob(os.path.join("outputs", "4x4", f"ql-4x4grid_run{run_id}_conn*_ep*.csv"))),
        "PPO": latest_matching_files(os.path.join("outputs", "4x4grid", f"ppo_test_final_run{run_id}_conn*_ep*.csv")),
        "Fixed-Time": latest_matching_files(os.path.join("outputs", "4x4grid", f"fixedtime_run{run_id}_conn*_ep*.csv")),
    }


def build_combined_summary(run_ids: List[int]):
    rows = []
    for run_id in run_ids:
        by_algo = files_for_run(run_id)
        for algo, files in by_algo.items():
            for path in files:
                ok, _, _ = is_full_episode_csv(path)
                if not ok:
                    continue
                record = summarize_episode(path)
                record["algo"] = algo
                record["run_id"] = run_id
                rows.append(record)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["algo", "run_id", "ep"]).reset_index(drop=True)


def plot_metric_vs_episode(df: pd.DataFrame, metric: str, save_path: str):
    sub = df[["algo", "run_id", "ep", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()

    episode_summary = (
        sub.groupby(["algo", "ep"], as_index=False)
        .agg(mean_value=(metric, "mean"), std_value=(metric, "std"), n_runs=(metric, "count"))
    )
    episode_summary["std_value"] = episode_summary["std_value"].fillna(0.0)

    apply_publication_style()
    fig, ax = plt.subplots()

    for algo in sorted(episode_summary["algo"].unique()):
        data = episode_summary[episode_summary["algo"] == algo].sort_values("ep")
        x = data["ep"].to_numpy(dtype=float)
        y = data["mean_value"].to_numpy(dtype=float)
        std = data["std_value"].to_numpy(dtype=float)
        color = ALGO_COLORS.get(algo)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=color,
            label=algo,
        )
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16)

    ylabel = METRIC_LABELS.get(metric, metric)
    style_axis(ax, title=f"{ylabel}: Across Runs", ylabel=ylabel)
    ax.legend(title="Method")
    finish_and_save(fig, save_path)


def plot_overall_run_means(df: pd.DataFrame, save_path: str):
    run_level = (
        df.groupby(["algo", "run_id"], as_index=False)[METRICS]
        .mean(numeric_only=True)
    )

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        summary = (
            run_level.groupby("algo", as_index=False)
            .agg(mean=(metric, "mean"), std=(metric, "std"))
        )
        summary["std"] = summary["std"].fillna(0.0)
        summary = summary.sort_values("algo")

        colors = [ALGO_COLORS.get(algo) for algo in summary["algo"]]
        ax.bar(summary["algo"], summary["mean"], yerr=summary["std"], color=colors, alpha=0.9, capsize=4)
        style_axis(ax, title=METRIC_LABELS.get(metric, metric), xlabel="")
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Method Comparison Using Run-Level Means", y=1.02, fontsize=13, fontweight="semibold")
    finish_and_save(fig, save_path)


def main():
    parser = argparse.ArgumentParser(description="Aggregate results across multiple runs and generate summary plots.")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3], help="Run ids to aggregate.")
    args = parser.parse_args()

    df = build_combined_summary(args.runs)
    if df.empty:
        raise FileNotFoundError("No valid episode CSV files were found for the requested runs.")

    run_tag = "runs" + "_".join(str(run_id) for run_id in args.runs)
    summary_path = os.path.join(OUT_DIR, f"compare_eval_summary_{run_tag}.csv")
    run_means_path = os.path.join(OUT_DIR, f"compare_eval_run_means_{run_tag}.csv")

    df.to_csv(summary_path, index=False)
    df.groupby(["algo", "run_id"], as_index=False)[METRICS].mean(numeric_only=True).to_csv(run_means_path, index=False)

    plot_metric_vs_episode(df, "mean_wait", os.path.join(OUT_DIR, f"fig_mean_wait_vs_ep_{run_tag}.png"))
    plot_metric_vs_episode(df, "stopped_ratio", os.path.join(OUT_DIR, f"fig_stopped_ratio_vs_ep_{run_tag}.png"))
    plot_metric_vs_episode(df, "total_wait", os.path.join(OUT_DIR, f"fig_total_wait_vs_ep_{run_tag}.png"))
    plot_overall_run_means(df, os.path.join(OUT_DIR, f"fig_overall_metrics_{run_tag}.png"))

    print("Saved:")
    print(" -", summary_path)
    print(" -", run_means_path)
    print(" -", os.path.join(OUT_DIR, f"fig_mean_wait_vs_ep_{run_tag}.png"))
    print(" -", os.path.join(OUT_DIR, f"fig_stopped_ratio_vs_ep_{run_tag}.png"))
    print(" -", os.path.join(OUT_DIR, f"fig_total_wait_vs_ep_{run_tag}.png"))
    print(" -", os.path.join(OUT_DIR, f"fig_overall_metrics_{run_tag}.png"))


if __name__ == "__main__":
    main()
