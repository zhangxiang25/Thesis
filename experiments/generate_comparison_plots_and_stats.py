# Generate thesis-ready comparison plots and summary statistics.
import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import ALGO_COLORS, METRIC_LABELS, apply_publication_style, finish_and_save, style_axis

# Optional: statistical test (Welch's t-test). If scipy not installed, we skip p-values.
try:
    from scipy.stats import ttest_ind
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["mean_wait", "total_wait", "mean_speed", "total_stopped", "stopped_ratio", "arrived_last"]


def ci95(mean, std, n):
    # normal approx; good enough for n>=15 (you have ~19/20 episodes)
    if n <= 1 or np.isnan(std):
        return (np.nan, np.nan)
    half = 1.96 * (std / np.sqrt(n))
    return (mean - half, mean + half)


def load_data(path):
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] != "train_csv"]
    return df


def summarize(df):
    rows = []
    for algo in sorted(df["algo"].unique()):
        d = df[df["algo"] == algo].copy()
        for m in METRICS:
            if m not in d.columns:
                continue
            x = pd.to_numeric(d[m], errors="coerce").dropna()
            n = int(x.shape[0])
            if n == 0:
                continue
            mu = float(x.mean())
            sd = float(x.std(ddof=1)) if n > 1 else 0.0
            lo, hi = ci95(mu, sd, n)
            rows.append({
                "algo": algo,
                "metric": m,
                "n": n,
                "mean": mu,
                "std": sd,
                "ci95_low": lo,
                "ci95_high": hi
            })
    return pd.DataFrame(rows)


def welch_tests(df):
    if not HAS_SCIPY:
        return pd.DataFrame([{
            "note": "scipy not installed -> p-values skipped. Install with: pip install scipy"
        }])

    out = []
    algos = sorted(df["algo"].unique())
    if len(algos) < 2:
        return pd.DataFrame([{"note": "Need at least two algorithms in data to compute tests."}])

    for algo_a, algo_b in combinations(algos, 2):
        for m in METRICS:
            if m not in df.columns:
                continue
            x1 = pd.to_numeric(df[df["algo"] == algo_a][m], errors="coerce").dropna().values
            x2 = pd.to_numeric(df[df["algo"] == algo_b][m], errors="coerce").dropna().values
            if len(x1) < 2 or len(x2) < 2:
                continue
            stat, p = ttest_ind(x1, x2, equal_var=False)
            out.append({
                "algo_a": algo_a,
                "algo_b": algo_b,
                "metric": m,
                "welch_t_stat": float(stat),
                "p_value": float(p),
                "algo_a_n": int(len(x1)),
                "algo_b_n": int(len(x2)),
            })
    return pd.DataFrame(out)


def plot_metric_vs_episode(df, metric, out_path):
    # Build per-episode series
    sub = df[["algo", "ep", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()

    # Sort for nicer lines
    sub = sub.sort_values(["algo", "ep"])

    apply_publication_style()
    fig, ax = plt.subplots()
    for algo in sorted(sub["algo"].unique()):
        d = sub[sub["algo"] == algo]
        ax.plot(
            d["ep"],
            d[metric],
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            color=ALGO_COLORS.get(algo, None),
            label=algo,
        )

    ylabel = METRIC_LABELS.get(metric, metric)
    style_axis(ax, title=f"{ylabel}: Method Comparison", ylabel=ylabel)
    ax.legend(title="Method")
    finish_and_save(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate plots and statistics from comparison summaries.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="If set, read/write run-specific comparison outputs such as compare_eval_summary_run2.csv.",
    )
    args = parser.parse_args()

    suffix = f"_run{args.run_id}" if args.run_id is not None else ""
    in_path = os.path.join("outputs", f"compare_eval_summary{suffix}.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Cannot find {in_path}. Run your comparison script first.")

    df = load_data(in_path)

    # Save stats
    stats = summarize(df)
    stats_path = os.path.join(OUT_DIR, f"eval_stats_summary{suffix}.csv")
    stats.to_csv(stats_path, index=False)

    tests = welch_tests(df)
    tests_path = os.path.join(OUT_DIR, f"eval_welch_tests{suffix}.csv")
    tests.to_csv(tests_path, index=False)

    # Plots
    mean_wait_path = os.path.join(OUT_DIR, f"fig_mean_wait_vs_ep{suffix}.png")
    stopped_ratio_path = os.path.join(OUT_DIR, f"fig_stopped_ratio_vs_ep{suffix}.png")
    plot_metric_vs_episode(df, "mean_wait", mean_wait_path)
    plot_metric_vs_episode(df, "stopped_ratio", stopped_ratio_path)

    print("Saved:")
    print(" -", stats_path)
    print(" -", tests_path)
    print(" -", mean_wait_path)
    print(" -", stopped_ratio_path)
    if not HAS_SCIPY:
        print("Note: scipy not installed, so p-values were skipped.")


if __name__ == "__main__":
    main()
