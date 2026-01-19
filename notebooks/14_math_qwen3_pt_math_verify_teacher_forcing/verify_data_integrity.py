"""Verify data integrity for NLL uptick analysis.

This script performs comprehensive checks to ensure no data has been lost
or corrupted during processing. It verifies:
- Run configuration completeness
- Parquet file structure
- Run ID coverage
- Row counts per run
- NaN patterns in log_prob columns
- Expected sequence counts
"""

import ast
import hashlib
import os
import re
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.analyze
import src.globals

# Add missing model size mappings
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["62M"] = 62e6
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["153M"] = 153e6

# Configuration
DATA_DIR = "notebooks/14_math_qwen3_pt_math_verify_teacher_forcing/data"
EVAL_SWEEP_IDS = [
    "9vtnq3bd",  # Qwen 3   34M
    "ovps81c2",  # Qwen 3   62M
    "oi9x67mh",  # Qwen 3   93M
    "em23bzb7",  # Qwen 3  153M
    "sy8h8i80",  # Qwen 3  344M
]
EXPECTED_SEQUENCES_PER_RUN = 5000  # MATH test set size
PROBLEMATIC_CONDITIONS = [
    ("34M", 0),
    ("62M", 0),
    ("93M", 1000),
    ("153M", 1000),
    ("344M", 3162),
]


def load_configs() -> pd.DataFrame:
    """Load and process evaluation run configurations."""
    df = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
        data_dir=DATA_DIR,
        sweep_ids=EVAL_SWEEP_IDS,
        refresh=False,
        wandb_username="rylan",
        finished_only=True,
    )

    df["Model"] = df["model_config"].apply(lambda x: ast.literal_eval(x)["model"])
    df["Parameters"] = df["Model"].apply(
        lambda x: re.search(r"Qwen3-([\d.]+[MB])", x).group(1)
    )
    df["Num. Replicas Per Epoch"] = df["Model"].apply(
        lambda x: int(re.search(r"rep_(\d+)_sbst", x).group(1))
    )
    df["Num. Epochs"] = df["Model"].apply(
        lambda x: int(re.search(r"epch_(\d+)_ot", x).group(1))
    )
    df["Num. MATH Test Set Replicas"] = (
        df["Num. Replicas Per Epoch"] * df["Num. Epochs"]
    )

    return df


def get_parquet_path() -> str:
    """Get the path to the histories parquet file."""
    filename = "sweeps=" + ",".join(EVAL_SWEEP_IDS)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    return os.path.join(DATA_DIR, f"{hashed_filename}_runs_histories.parquet")


def verify_run_configs(configs_df: pd.DataFrame) -> List[str]:
    """Verify run configuration completeness."""
    issues = []

    print("\n[1] VERIFYING RUN CONFIGS")
    print("-" * 50)
    print(f"Total runs in config: {len(configs_df)}")
    print(f"Unique run_ids: {configs_df['run_id'].nunique()}")

    print(f"\nRuns by model size:")
    print(configs_df.groupby("Parameters").size())

    print(f"\nRuns by replica level:")
    print(configs_df.groupby("Num. MATH Test Set Replicas").size())

    print(f"\nRuns by (model size, replicas):")
    cross_tab = configs_df.groupby(
        ["Parameters", "Num. MATH Test Set Replicas"]
    ).size().unstack(fill_value=0)
    print(cross_tab)

    return issues


def verify_parquet_structure(parquet_path: str) -> Tuple[pq.ParquetFile, List[str], List[str]]:
    """Verify parquet file structure."""
    issues = []

    print("\n[2] VERIFYING PARQUET FILE STRUCTURE")
    print("-" * 50)
    print(f"Parquet file: {parquet_path}")
    print(f"File exists: {os.path.exists(parquet_path)}")

    if not os.path.exists(parquet_path):
        issues.append("Parquet file does not exist")
        return None, [], issues

    parquet_file = pq.ParquetFile(parquet_path)
    print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
    print(f"Total rows: {parquet_file.metadata.num_rows}")

    all_columns = [field.name for field in parquet_file.schema]
    log_prob_cols = [c for c in all_columns if c.startswith("log_prob_token_")]
    print(f"Total columns: {len(all_columns)}")
    print(f"Log prob columns: {len(log_prob_cols)}")

    token_indices = sorted([int(c.replace("log_prob_token_", "")) for c in log_prob_cols])
    print(f"Token index range: {min(token_indices)} to {max(token_indices)}")

    return parquet_file, log_prob_cols, issues


def verify_run_id_coverage(
    parquet_file: pq.ParquetFile,
    configs_df: pd.DataFrame
) -> Tuple[Set[str], List[str]]:
    """Verify run_id coverage between parquet and config."""
    issues = []

    print("\n[3] VERIFYING RUN_ID COVERAGE")
    print("-" * 50)

    table = parquet_file.read_row_group(0, columns=["run_id"])
    parquet_run_ids = set(table.to_pandas()["run_id"].unique())
    config_run_ids = set(configs_df["run_id"].unique())

    print(f"Run IDs in parquet: {len(parquet_run_ids)}")
    print(f"Run IDs in config: {len(config_run_ids)}")
    print(f"Run IDs in both: {len(parquet_run_ids & config_run_ids)}")
    print(f"Run IDs only in parquet: {len(parquet_run_ids - config_run_ids)}")
    print(f"Run IDs only in config: {len(config_run_ids - parquet_run_ids)}")

    if parquet_run_ids - config_run_ids:
        print(f"  WARNING: {parquet_run_ids - config_run_ids}")
    if config_run_ids - parquet_run_ids:
        print(f"  WARNING: {config_run_ids - parquet_run_ids}")
        issues.append("Run ID mismatch between parquet and config")

    return parquet_run_ids & config_run_ids, issues


def verify_row_counts(df: pd.DataFrame, config_run_ids: Set[str]) -> List[str]:
    """Verify row counts per run."""
    issues = []

    print("\n[4] VERIFYING ROW COUNTS PER RUN")
    print("-" * 50)

    rows_per_run = df.groupby("run_id").size()
    print(f"Rows per run - min: {rows_per_run.min()}, max: {rows_per_run.max()}, "
          f"mean: {rows_per_run.mean():.1f}")
    print(f"Total rows: {len(df)}")

    if rows_per_run.nunique() == 1:
        print(f"✓ All runs have exactly {rows_per_run.iloc[0]} rows (sequences)")
    else:
        print("⚠ Runs have varying row counts:")
        print(rows_per_run.value_counts().head(10))
        issues.append("Variable row counts per run")

    expected_total = EXPECTED_SEQUENCES_PER_RUN * len(config_run_ids)
    if len(df) != expected_total:
        issues.append(f"Sequence count mismatch: {len(df)} vs {expected_total}")

    return issues


def verify_nan_patterns(df: pd.DataFrame, log_prob_cols: List[str]) -> List[str]:
    """Verify NaN patterns in log_prob columns."""
    issues = []

    print("\n[5] VERIFYING NaN PATTERNS IN LOG_PROB COLUMNS")
    print("-" * 50)

    sample_tokens = [0, 100, 200, 400, 600, 800]
    sample_cols = [f"log_prob_token_{t}" for t in sample_tokens
                   if f"log_prob_token_{t}" in log_prob_cols]

    print("Non-NaN counts by token position (should decrease for later tokens):")
    for col in sample_cols:
        token_idx = int(col.replace("log_prob_token_", ""))
        non_nan = df[col].notna().sum()
        pct = non_nan / len(df) * 100
        print(f"  Token {token_idx:4d}: {non_nan:6d} non-NaN values ({pct:5.1f}%)")

    return issues


def verify_sequence_counts(df: pd.DataFrame, config_run_ids: Set[str]) -> List[str]:
    """Verify total sequence counts match expectations."""
    issues = []

    print("\n[6] VERIFYING COUNTS MATCH EXPECTATIONS")
    print("-" * 50)

    total_sequences = len(df)
    expected_total = EXPECTED_SEQUENCES_PER_RUN * len(config_run_ids)

    print(f"Expected sequences per run: {EXPECTED_SEQUENCES_PER_RUN}")
    print(f"Expected total sequences: {expected_total}")
    print(f"Actual total sequences: {total_sequences}")

    if total_sequences == expected_total:
        print("✓ Total sequence count matches expectation")
    else:
        print(f"⚠ Mismatch: expected {expected_total}, got {total_sequences}")
        print(f"  Difference: {total_sequences - expected_total}")
        issues.append(f"Sequence count mismatch: {total_sequences} vs {expected_total}")

    return issues


def verify_token_zero_coverage(df: pd.DataFrame, configs_df: pd.DataFrame) -> List[str]:
    """Verify all sequences have valid log_prob at token 0."""
    issues = []

    print("\n[7] VERIFYING AGGREGATION PRESERVES DATA")
    print("-" * 50)

    run_id_to_config = configs_df.set_index("run_id")[
        ["Parameters", "Num. MATH Test Set Replicas"]
    ].to_dict("index")

    col = "log_prob_token_0"
    token_0_data = df[["run_id", col]].copy()
    token_0_data = token_0_data[token_0_data["run_id"].isin(run_id_to_config)]
    token_0_non_nan = token_0_data[col].notna().sum()

    print(f"Token 0 - sequences with valid log_prob: {token_0_non_nan}")
    print(f"Token 0 - expected (all sequences): {len(token_0_data)}")

    if token_0_non_nan == len(token_0_data):
        print("✓ All sequences have valid log_prob at token 0")
    else:
        missing = len(token_0_data) - token_0_non_nan
        print(f"⚠ {missing} sequences missing log_prob at token 0")
        issues.append("Missing log_prob values at token 0")

    return issues


def verify_problematic_conditions(df: pd.DataFrame, configs_df: pd.DataFrame) -> None:
    """Verify data exists for problematic conditions."""
    print("\n[8] VERIFYING PROBLEMATIC CONDITIONS")
    print("-" * 50)

    for param, replica in PROBLEMATIC_CONDITIONS:
        matching_runs = configs_df[
            (configs_df["Parameters"] == param)
            & (configs_df["Num. MATH Test Set Replicas"] == replica)
        ]

        print(f"\n{param}, R={replica}:")
        print(f"  Matching runs: {len(matching_runs)}")

        if len(matching_runs) == 0:
            print("  ⚠ NO MATCHING RUNS FOUND!")
            continue

        for _, run in matching_runs.iterrows():
            run_id = run["run_id"]
            run_data = df[df["run_id"] == run_id]
            print(f"  Run {run_id}: {len(run_data)} sequences")

            for token in [0, 100, 400, 800]:
                col = f"log_prob_token_{token}"
                if col in run_data.columns:
                    non_nan = run_data[col].notna().sum()
                    print(f"    Token {token}: {non_nan} valid values")


def verify_cached_data(parquet_path: str) -> None:
    """Check if cached cumulative NLL data exists."""
    print("\n[9] COMPARING WITH MAIN NOTEBOOK CACHE")
    print("-" * 50)

    filename = "sweeps=" + ",".join(EVAL_SWEEP_IDS)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    cumulative_nll_cache = os.path.join(
        DATA_DIR, f"{hashed_filename}_cumulative_nll_per_run.parquet"
    )

    if os.path.exists(cumulative_nll_cache):
        cached_df = pd.read_parquet(cumulative_nll_cache)
        print("Cached cumulative NLL file exists")
        print(f"  Shape: {cached_df.shape}")
        print(f"  Columns: {list(cached_df.columns)}")
        print(f"  Run IDs: {cached_df['run_id'].nunique()}")
        print(f"  Token indices: {cached_df['Token Index'].nunique()}")
    else:
        print("Cached cumulative NLL file not found")


def main():
    """Run all data integrity verification checks."""
    print("=" * 70)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 70)

    all_issues = []

    # Load configs
    configs_df = load_configs()
    all_issues.extend(verify_run_configs(configs_df))

    # Verify parquet structure
    parquet_path = get_parquet_path()
    parquet_file, log_prob_cols, issues = verify_parquet_structure(parquet_path)
    all_issues.extend(issues)

    if parquet_file is None:
        print("\n⚠ Cannot continue - parquet file missing")
        return

    # Verify run_id coverage
    valid_run_ids, issues = verify_run_id_coverage(parquet_file, configs_df)
    all_issues.extend(issues)

    # Load full data for remaining checks
    df = parquet_file.read().to_pandas()

    # Verify row counts
    all_issues.extend(verify_row_counts(df, valid_run_ids))

    # Verify NaN patterns
    all_issues.extend(verify_nan_patterns(df, log_prob_cols))

    # Verify sequence counts
    all_issues.extend(verify_sequence_counts(df, valid_run_ids))

    # Verify token 0 coverage
    all_issues.extend(verify_token_zero_coverage(df, configs_df))

    # Verify problematic conditions
    verify_problematic_conditions(df, configs_df)

    # Check cached data
    verify_cached_data(parquet_path)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if all_issues:
        print("⚠ ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✓ ALL CHECKS PASSED - Data integrity verified")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
