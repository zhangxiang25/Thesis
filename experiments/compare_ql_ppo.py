import glob
import os
import re
import argparse
import numpy as np
import pandas as pd


def extract_ep(path: str):
    m = re.search(r"_ep(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


def latest_matching_files(pattern: str):
    files = glob.glob(pattern)
    if not files:
        return []

    by_conn = {}
    for path in files:
        m = re.search(r"_conn(\d+)_ep\d+\.csv$", os.path.basename(path))
        conn = int(m.group(1)) if m else -1
        by_conn.setdefault(conn, []).append(path)

    # Prefer the connection id with the most files; tie-break with larger conn id.
    best_conn, best_files = max(by_conn.items(), key=lambda item: (len(item[1]), item[0]))
    print(f"Using files from conn{best_conn} for pattern: {pattern}")
    return sorted(best_files)


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

    def last_col(col):
        return int(df[col].iloc[-1]) if col in df.columns else np.nan

    # stopped ratio
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

        "arrived_last": last_col("system_total_arrived"),
        "departed_last": last_col("system_total_departed"),
        "teleported_last": last_col("system_total_teleported"),
    }


def load_files(file_list, algo_tag, label):
    kept, skipped = [], []
    rows = []

    for f in sorted(file_list, key=lambda x: extract_ep(x) if extract_ep(x) is not None else 10**9):
        ep = extract_ep(f)
        if ep is None:
            continue
        ok, step_max, nrows = is_full_episode_csv(f)
        if not ok:
            skipped.append((os.path.basename(f), ep, nrows, step_max))
            continue
        kept.append((os.path.basename(f), ep, nrows, step_max))
        r = summarize_episode(f)
        r["algo"] = algo_tag
        r["split"] = label
        rows.append(r)

    df_sum = pd.DataFrame(rows)
    if not df_sum.empty:
        df_sum = df_sum.sort_values(["algo", "ep"])

    print(f"\n[{algo_tag} | {label}] kept full episodes: {len(kept)}")
    if kept:
        print("  examples kept:", ", ".join([f"ep{ep}" for _, ep, _, _ in kept[:8]]), "...")
    if skipped:
        print(f"[{algo_tag} | {label}] skipped (not full): {len(skipped)}  (showing up to 5)")
        for name, ep, nrows, stepmax in skipped[:5]:
            print(f"  SKIP {name}  ep{ep} rows={nrows} step_max={stepmax}")

    return df_sum


def last_k_mean(df: pd.DataFrame, k=5):
    if df.empty:
        return pd.Series(dtype=float)
    df2 = df.sort_values("ep")
    use_k = min(k, len(df2))
    tail = df2.tail(use_k)
    cols = ["mean_wait", "total_wait", "mean_speed", "total_stopped", "stopped_ratio", "arrived_last"]
    cols = [c for c in cols if c in tail.columns]
    return tail[cols].mean(numeric_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate QL, PPO, and fixed-time CSV outputs.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="If set, only compare files for the selected run id and write run-specific summaries.",
    )
    args = parser.parse_args()

    suffix = f"_run{args.run_id}" if args.run_id is not None else ""

    if args.run_id is None:
        # Legacy patterns keep thesis-compatible default filenames.
        ql_pattern = "outputs/4x4/ql-*.csv"
        ppo_train_pattern = "outputs/4x4grid/ppo_conn*_ep*.csv"
        ppo_eval_pattern = "outputs/4x4grid/ppo_test_final_conn*_ep*.csv"
        fixed_time_pattern = "outputs/4x4grid/fixedtime_conn*_ep*.csv"
    else:
        ql_pattern = f"outputs/4x4/ql-4x4grid_run{args.run_id}_conn*_ep*.csv"
        ppo_train_pattern = f"outputs/4x4grid/ppo_run{args.run_id}_conn*_ep*.csv"
        ppo_eval_pattern = f"outputs/4x4grid/ppo_test_final_run{args.run_id}_conn*_ep*.csv"
        fixed_time_pattern = f"outputs/4x4grid/fixedtime_run{args.run_id}_conn*_ep*.csv"

    # QL files (env.save_csv)
    ql_files = glob.glob(ql_pattern)

    # PPO training env CSV (often noisy/partial under parallel workers)
    ppo_train_files = latest_matching_files(ppo_train_pattern)

    # PPO evaluation/test files (recommended for thesis main comparison)
    ppo_eval_files = latest_matching_files(ppo_eval_pattern)

    # Fixed-time baseline files on the same shared 4x4 setup
    fixed_time_files = latest_matching_files(fixed_time_pattern)

    ql_sum = load_files(ql_files, "QL", "train_or_run")

    ppo_train_sum = load_files(ppo_train_files, "PPO", "train_csv")
    ppo_eval_sum = load_files(ppo_eval_files, "PPO", "test_final")
    fixed_time_sum = load_files(fixed_time_files, "Fixed-Time", "run")

    os.makedirs("outputs", exist_ok=True)

    # 1) Training-ish comparison (optional)
    train_compare = pd.concat([ql_sum, ppo_train_sum], ignore_index=True)
    train_out = f"outputs/compare_train_summary{suffix}.csv"
    train_compare.to_csv(train_out, index=False)
    print(f"\nSaved -> {train_out}")
    print("\n=== Last episodes mean (train_csv) ===")
    print("QL:\n", last_k_mean(ql_sum, 5))
    print("PPO train_csv:\n", last_k_mean(ppo_train_sum, 5))

    # 2) Thesis main: Evaluation comparison using test_final + optional fixed-time baseline
    eval_frames = [df for df in [ql_sum, ppo_eval_sum, fixed_time_sum] if not df.empty]
    eval_compare = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    eval_out = f"outputs/compare_eval_summary{suffix}.csv"
    eval_compare.to_csv(eval_out, index=False)
    print(f"\nSaved -> {eval_out}")
    print("\n=== Last episodes mean (test_final) ===")
    print("QL:\n", last_k_mean(ql_sum, 5))
    print("PPO test_final:\n", last_k_mean(ppo_eval_sum, 5))
    if not fixed_time_sum.empty:
        print("Fixed-Time:\n", last_k_mean(fixed_time_sum, 5))
