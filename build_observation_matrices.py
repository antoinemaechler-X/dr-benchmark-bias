#!/usr/bin/env python3.10
"""
Build observation matrices and item statistics for all 16 benchmarks.

Data loading follows the measurement-db schema from:
  https://github.com/aims-foundation/measurement-db
  (see data/DATA_FORMAT.md for the parquet schema specification)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "processed"

BENCHMARKS = [
    "afrimedqa", "agentdojo", "ai2d_test", "androidworld", "bfcl",
    "cybench", "hle", "livecodebench", "matharena", "mathvista_mini",
    "mmbench_v11", "mmlupro", "mtbench", "rewardbench", "swebench",
    "ultrafeedback",
]

# Primary test condition rules
# None = use all rows; string = filter to that value; "largest" = pick largest condition
TC_RULES = {
    "afrimedqa": "source=afrimedqa-v2|prompt=base",
    "agentdojo": "metric=utility|attack=important_instructions",
    "livecodebench": "source=submissions",
    "matharena": None,  # special handling: use null-TC rows (the main data)
    "cybench": "mode=unguided",  # pick one mode; unguided is the primary evaluation
    "ultrafeedback": "aspect=helpfulness",  # pick largest (all same size, helpfulness is standard)
}
# For mmbench_v11, rewardbench: conditions are subsets (skills/categories),
# not experimental variants. Use all rows together.

# Normalization: benchmark_id -> (min_val, max_val) for rescaling to [0,1]
NORMALIZE = {
    "mtbench": (1, 10),       # likert 1-10 -> (x-1)/9
    "ultrafeedback": (1, 5),  # likert 1-5 -> (x-1)/4
}


def filter_test_condition(df, benchmark_id):
    """Filter to primary test condition for a benchmark."""
    rule = TC_RULES.get(benchmark_id, None)

    if benchmark_id == "matharena":
        # Use rows with null test_condition (the main per-attempt binary data)
        # Non-null TC rows are judge/criterion rubric evaluations for a small subset
        filtered = df[df["test_condition"].isna()].copy()
        return filtered, "null (per-attempt binary data)"

    if rule is not None:
        filtered = df[df["test_condition"] == rule].copy()
        return filtered, rule

    # Default: check if test_condition has values
    tc_values = df["test_condition"].dropna().unique()
    if len(tc_values) == 0:
        return df.copy(), "all (no conditions)"

    if len(tc_values) == 1:
        return df.copy(), f"all (single condition: {tc_values[0]})"

    # Multiple conditions that are subsets (like mmbench skills, rewardbench subsets)
    # Use all rows — each item belongs to one category
    return df.copy(), f"all ({len(tc_values)} subset conditions)"


def normalize_response(response_series, benchmark_id):
    """Normalize responses to [0, 1]."""
    if benchmark_id in NORMALIZE:
        lo, hi = NORMALIZE[benchmark_id]
        return (response_series - lo) / (hi - lo)
    return response_series


def compute_item_discrimination(response_matrix):
    """
    Point-biserial correlation between each item and model total score.
    response_matrix: DataFrame (models x items), NaN for missing.
    Returns Series indexed by item_id.
    """
    # Model total score = mean across all observed items for each model
    model_means = response_matrix.mean(axis=1)
    discriminations = {}
    for item_id in response_matrix.columns:
        col = response_matrix[item_id]
        observed = col.dropna()
        if len(observed) < 3:
            discriminations[item_id] = np.nan
            continue
        model_scores = model_means.loc[observed.index]
        # Use Pearson correlation (point-biserial is Pearson for binary vs continuous)
        if observed.std() == 0 or model_scores.std() == 0:
            discriminations[item_id] = np.nan
        else:
            discriminations[item_id] = observed.corr(model_scores)
    return pd.Series(discriminations)


def analyze_block_structure(obs_matrix):
    """Check if missingness has block structure by sorting and looking for steps."""
    O = obs_matrix.values.copy()
    # Sort by row sum descending, then column sum descending
    row_sums = O.sum(axis=1)
    col_sums = O.sum(axis=0)
    row_order = np.argsort(-row_sums)
    col_order = np.argsort(-col_sums)
    O_sorted = O[row_order][:, col_order]

    # Check for "step" pattern: for each row, find the last observed column
    n_models, n_items = O_sorted.shape
    if n_models == 0 or n_items == 0:
        return "empty"

    last_obs = []
    for i in range(n_models):
        obs_cols = np.where(O_sorted[i] == 1)[0]
        if len(obs_cols) > 0:
            last_obs.append(obs_cols[-1])
        else:
            last_obs.append(-1)

    last_obs = np.array(last_obs)
    # If last_obs is monotonically non-increasing, it's a clean staircase
    diffs = np.diff(last_obs)
    n_increases = (diffs > 0).sum()
    pct_monotone = 1.0 - n_increases / max(len(diffs), 1)

    if pct_monotone > 0.9:
        return f"strong block/staircase (monotonicity={pct_monotone:.2f})"
    elif pct_monotone > 0.7:
        return f"moderate block structure (monotonicity={pct_monotone:.2f})"
    else:
        return f"weak/no block structure (monotonicity={pct_monotone:.2f})"


def process_benchmark(benchmark_id, items_df, subjects_df, benchmarks_df):
    """Process a single benchmark: build matrices, compute stats."""
    print(f"\n{'='*60}")
    print(f"Processing: {benchmark_id}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_parquet(DATA_DIR / f"{benchmark_id}.parquet")
    print(f"  Raw rows: {len(df):,}, models: {df.subject_id.nunique()}, items: {df.item_id.nunique()}")

    # Step 1a: Filter test condition
    df, tc_desc = filter_test_condition(df, benchmark_id)
    print(f"  Test condition: {tc_desc}")
    print(f"  After filter: {len(df):,} rows, {df.subject_id.nunique()} models, {df.item_id.nunique()} items")

    # Step 1b: Aggregate trials — mean response per (subject_id, item_id)
    agg = df.groupby(["subject_id", "item_id"])["response"].mean().reset_index()
    print(f"  After trial aggregation: {len(agg):,} (model, item) pairs")

    # Step 1c: Build response matrix (pivot)
    response_matrix = agg.pivot(index="subject_id", columns="item_id", values="response")
    n_models, n_items = response_matrix.shape
    print(f"  Response matrix: {n_models} models × {n_items} items")

    # Step 1d: Normalize to [0,1]
    response_min_raw = response_matrix.min().min()
    response_max_raw = response_matrix.max().max()
    response_matrix = normalize_response(response_matrix, benchmark_id)
    response_min = response_matrix.min().min()
    response_max = response_matrix.max().max()
    response_mean = response_matrix.mean().mean()
    print(f"  Response range: [{response_min_raw:.2f}, {response_max_raw:.2f}] -> [{response_min:.4f}, {response_max:.4f}]")

    # Step 1e: Build observation matrix
    obs_matrix = (~response_matrix.isna()).astype(int)
    n_observed = obs_matrix.values.sum()
    n_total = n_models * n_items
    density = n_observed / n_total
    print(f"  Density: {n_observed:,}/{n_total:,} = {density:.4f} ({density*100:.1f}%)")

    # Step 2: Item-level statistics
    item_difficulty = response_matrix.mean(axis=0)
    item_discrimination = compute_item_discrimination(response_matrix)
    n_models_observed = obs_matrix.sum(axis=0)

    # Content length from items.parquet
    bench_items = items_df[items_df["benchmark_id"] == benchmark_id].set_index("item_id")
    content_lengths = bench_items["content"].str.len()

    item_stats = pd.DataFrame({
        "item_id": response_matrix.columns,
        "item_difficulty": item_difficulty.values,
        "item_discrimination": item_discrimination.values,
        "n_models_observed": n_models_observed.values,
    })
    item_stats = item_stats.merge(
        content_lengths.rename("content_length").reset_index(),
        on="item_id", how="left"
    )

    # Model-level statistics
    model_mean = response_matrix.mean(axis=1)
    n_items_observed = obs_matrix.sum(axis=1)
    model_obs_rate = n_items_observed / n_items

    model_stats = pd.DataFrame({
        "subject_id": response_matrix.index,
        "model_mean": model_mean.values,
        "n_items_observed": n_items_observed.values,
        "model_obs_rate": model_obs_rate.values,
    })
    model_stats = model_stats.merge(
        subjects_df[["subject_id", "display_name"]], on="subject_id", how="left"
    )
    # Reorder columns
    model_stats = model_stats[["subject_id", "display_name", "model_mean", "n_items_observed", "model_obs_rate"]]

    # Step 3: Missingness analysis
    item_cov = n_models_observed.values
    model_cov = n_items_observed.values

    item_cov_pcts = np.percentile(item_cov, [10, 25, 50, 75, 90])
    model_cov_pcts = np.percentile(model_cov, [10, 25, 50, 75, 90])

    # Correlation: difficulty vs coverage
    valid_diff = item_stats.dropna(subset=["item_difficulty", "n_models_observed"])
    if len(valid_diff) > 2 and valid_diff["n_models_observed"].std() > 0:
        corr_diff_cov, _ = spearmanr(valid_diff["item_difficulty"], valid_diff["n_models_observed"])
    else:
        corr_diff_cov = np.nan

    # Correlation: model strength vs coverage
    valid_str = model_stats.dropna(subset=["model_mean", "n_items_observed"])
    if len(valid_str) > 2 and valid_str["n_items_observed"].std() > 0:
        corr_str_cov, _ = spearmanr(valid_str["model_mean"], valid_str["n_items_observed"])
    else:
        corr_str_cov = np.nan

    # Block structure
    block_desc = analyze_block_structure(obs_matrix)

    print(f"  Item coverage percentiles (p10/p50/p90): {item_cov_pcts[0]:.0f}/{item_cov_pcts[2]:.0f}/{item_cov_pcts[4]:.0f}")
    print(f"  Model coverage percentiles (p10/p50/p90): {model_cov_pcts[0]:.0f}/{model_cov_pcts[2]:.0f}/{model_cov_pcts[4]:.0f}")
    print(f"  Corr(difficulty, coverage): {corr_diff_cov:.4f}" if not np.isnan(corr_diff_cov) else "  Corr(difficulty, coverage): N/A")
    print(f"  Corr(model_strength, coverage): {corr_str_cov:.4f}" if not np.isnan(corr_str_cov) else "  Corr(model_strength, coverage): N/A")
    print(f"  Block structure: {block_desc}")

    # Get response_type from benchmarks metadata
    bm_row = benchmarks_df[benchmarks_df["benchmark_id"] == benchmark_id]
    response_type = bm_row["response_type"].values[0] if len(bm_row) > 0 else "unknown"

    # Step 4: Save outputs
    out_dir = OUT_DIR / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Response matrix as dense numpy array (float32)
    resp_arr = response_matrix.values.astype(np.float32)
    np.savez_compressed(out_dir / "response_matrix.npz", data=resp_arr)

    # Observation matrix
    obs_arr = obs_matrix.values.astype(np.float32)
    np.savez_compressed(out_dir / "observation_matrix.npz", data=obs_arr)

    # IDs
    with open(out_dir / "model_ids.json", "w") as f:
        json.dump(list(response_matrix.index), f)
    with open(out_dir / "item_ids.json", "w") as f:
        json.dump(list(response_matrix.columns), f)

    # Stats
    item_stats.to_csv(out_dir / "item_stats.csv", index=False)
    model_stats.to_csv(out_dir / "model_stats.csv", index=False)

    # Summary row
    summary = {
        "benchmark_id": benchmark_id,
        "n_models": n_models,
        "n_items": n_items,
        "n_observed": int(n_observed),
        "n_total": int(n_total),
        "density": density,
        "item_coverage_p10": item_cov_pcts[0],
        "item_coverage_p50": item_cov_pcts[2],
        "item_coverage_p90": item_cov_pcts[4],
        "model_coverage_p10": model_cov_pcts[0],
        "model_coverage_p50": model_cov_pcts[2],
        "model_coverage_p90": model_cov_pcts[4],
        "corr_difficulty_coverage": corr_diff_cov,
        "corr_strength_coverage": corr_str_cov,
        "response_min": response_min,
        "response_max": response_max,
        "response_mean": response_mean,
        "response_type": response_type,
    }

    return summary


def main():
    # Load metadata
    subjects_df = pd.read_parquet(DATA_DIR / "subjects.parquet")
    items_df = pd.read_parquet(DATA_DIR / "items.parquet")
    benchmarks_df = pd.read_parquet(DATA_DIR / "benchmarks.parquet")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for benchmark_id in BENCHMARKS:
        summary = process_benchmark(benchmark_id, items_df, subjects_df, benchmarks_df)
        summaries.append(summary)

    # Save observation_summary.csv
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "observation_summary.csv", index=False)

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OBSERVATION SUMMARY")
    print("=" * 80)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))

    # Top 3 benchmarks with highest missingness
    print("\n" + "=" * 80)
    print("TOP 3 BENCHMARKS WITH HIGHEST MISSINGNESS")
    print("=" * 80)
    top_missing = summary_df.nsmallest(3, "density")
    for _, row in top_missing.iterrows():
        bm = row["benchmark_id"]
        print(f"\n--- {bm} (density={row['density']:.4f}) ---")
        print(f"  Corr(difficulty, coverage): {row['corr_difficulty_coverage']:.4f}")
        print(f"  Corr(model_strength, coverage): {row['corr_strength_coverage']:.4f}")

        # Load item stats for more detail
        istats = pd.read_csv(OUT_DIR / bm / "item_stats.csv")
        mstats = pd.read_csv(OUT_DIR / bm / "model_stats.csv")

        print(f"  Item difficulty: mean={istats.item_difficulty.mean():.4f}, std={istats.item_difficulty.std():.4f}")
        print(f"  Item discrimination: mean={istats.item_discrimination.mean():.4f}, std={istats.item_discrimination.std():.4f}")

        # Top 5 items by least coverage
        least_covered = istats.nsmallest(5, "n_models_observed")
        print(f"  5 least-covered items:")
        for _, irow in least_covered.iterrows():
            print(f"    {irow['item_id']}: n_models={int(irow['n_models_observed'])}, difficulty={irow['item_difficulty']:.4f}")

        # Top 5 models by least coverage
        least_models = mstats.nsmallest(5, "n_items_observed")
        print(f"  5 least-covered models:")
        for _, mrow in least_models.iterrows():
            name = mrow['display_name'] if pd.notna(mrow['display_name']) else mrow['subject_id']
            print(f"    {name}: n_items={int(mrow['n_items_observed'])}, mean_score={mrow['model_mean']:.4f}")

    # Verify all output files exist
    print("\n" + "=" * 80)
    print("FILE VERIFICATION")
    print("=" * 80)
    all_ok = True
    expected_files = [
        "response_matrix.npz", "observation_matrix.npz",
        "model_ids.json", "item_ids.json",
        "item_stats.csv", "model_stats.csv"
    ]
    for bm in BENCHMARKS:
        for fname in expected_files:
            fpath = OUT_DIR / bm / fname
            if not fpath.exists():
                print(f"  MISSING: {fpath}")
                all_ok = False

    summary_path = OUT_DIR / "observation_summary.csv"
    if not summary_path.exists():
        print(f"  MISSING: {summary_path}")
        all_ok = False

    if all_ok:
        print("  All output files verified OK!")
    print(f"\nTotal output files: {len(BENCHMARKS) * len(expected_files) + 1}")


if __name__ == "__main__":
    main()
