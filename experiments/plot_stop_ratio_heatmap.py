import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_OUTPUT_DIR = os.path.join("outputs", "4x4", "plots")


def extract_episode(path):
    match = re.search(r"_ep(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_default_csv():
    files = glob.glob(os.path.join("outputs", "4x4", "custom_metrics_run1_ep*.csv"))
    if not files:
        return None
    return max(files, key=extract_episode)


def plot_congestion_heatmap(csv_path, save_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find CSV file: {csv_path}")

    df = pd.read_csv(csv_path)
    ratio_cols = [col for col in df.columns if col.endswith("_stop_ratio")]

    if not ratio_cols:
        raise ValueError("No '*_stop_ratio' columns were found in the CSV file.")

    sorted_cols = sorted(ratio_cols, key=lambda name: int(name.split("_")[0]))
    heatmap_data = df[sorted_cols].T
    heatmap_data.index = [name.split("_")[0] for name in heatmap_data.index]

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        data=heatmap_data,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Stop Ratio"},
    )
    plt.title(f"Stop Ratio Heatmap - {os.path.basename(csv_path)}")
    plt.xlabel("Simulation Step")
    plt.ylabel("Intersection Agent ID")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to a custom_metrics CSV file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    args = parser.parse_args()

    csv_path = args.csv or find_default_csv()
    if not csv_path:
        raise FileNotFoundError("No custom_metrics_run1_ep*.csv files were found under outputs/4x4.")

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    default_name = f"stop_ratio_heatmap_ep{extract_episode(csv_path)}.png"
    save_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, default_name)

    plot_congestion_heatmap(csv_path, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
