"""Investigate NLL uptick phenomenon in teacher-forced evaluation.

For certain combinations of model size and number of replicas, the NLL *increases*
with token index, which is unexpected. This script investigates the following
conditions where this occurs:
- 34M + 0 Replicas
- 62M + 0 Replicas
- 93M + 1000 Replicas
- 153M + 1000 Replicas
- 344M + 3162 Replicas

The analysis computes NLL trends, examines sample counts, and tests hypotheses
about selection bias due to varying sequence lengths.
"""

import ast
import gc
import hashlib
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import seaborn as sns

import src.analyze
import src.globals
import src.plot

# Add missing model size mappings (these models use non-standard naming)
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["62M"] = 62e6
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["153M"] = 153e6

# Configuration
MAX_TOKEN_INDEX = 800
PROBLEMATIC_CONDITIONS = [
    ("34M", 0),
    ("62M", 0),
    ("93M", 1000),
    ("153M", 1000),
    ("344M", 3162),
]
EVAL_SWEEP_IDS = [
    "9vtnq3bd",  # Qwen 3   34M     1xOT    Subset Fraction=1.0
    "ovps81c2",  # Qwen 3   62M     1xOT    Subset Fraction=1.0
    "oi9x67mh",  # Qwen 3   93M     1xOT    Subset Fraction=1.0
    "em23bzb7",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "sy8h8i80",  # Qwen 3  344M     1xOT    Subset Fraction=1.0
]


def load_run_configs(data_dir: str) -> pd.DataFrame:
    """Load and process evaluation run configurations."""
    df = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
        data_dir=data_dir,
        sweep_ids=EVAL_SWEEP_IDS,
        refresh=False,
        wandb_username="rylan",
        finished_only=True,
    )

    # Extract model metadata
    df["Model"] = df["model_config"].apply(
        lambda x: ast.literal_eval(x)["model"]
    )
    df["Parameters"] = df["Model"].apply(
        lambda x: re.search(r"Qwen3-([\d.]+[MB])", x).group(1)
    )
    df["Num. Parameters"] = df["Parameters"].apply(
        lambda x: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[x]
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


def load_nll_data(data_dir: str, configs_df: pd.DataFrame) -> pd.DataFrame:
    """Load and aggregate NLL data from parquet file."""
    filename = "sweeps=" + ",".join(EVAL_SWEEP_IDS)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    histories_path = os.path.join(data_dir, f"{hashed_filename}_runs_histories.parquet")

    print(f"Loading histories from {histories_path}")

    run_id_to_config = configs_df.set_index("run_id")[
        ["Parameters", "Num. Parameters", "Num. MATH Test Set Replicas"]
    ].to_dict("index")

    parquet_file = pq.ParquetFile(histories_path)
    all_columns = [field.name for field in parquet_file.schema]

    log_prob_cols = sorted(
        [
            c for c in all_columns
            if c.startswith("log_prob_token_")
            and int(c.replace("log_prob_token_", "")) <= MAX_TOKEN_INDEX
        ],
        key=lambda x: int(x.replace("log_prob_token_", "")),
    )
    print(f"Using {len(log_prob_cols)} token positions (0 to {MAX_TOKEN_INDEX})")

    # Process parquet file
    aggregated_results = []
    num_row_groups = parquet_file.metadata.num_row_groups

    for rg_idx in range(num_row_groups):
        print(f"  Processing row group {rg_idx + 1}/{num_row_groups}...")

        table = parquet_file.read_row_group(rg_idx, columns=["run_id"] + log_prob_cols)
        chunk_df = table.to_pandas()
        del table
        gc.collect()

        for run_id in chunk_df["run_id"].unique():
            if run_id not in run_id_to_config:
                continue

            config = run_id_to_config[run_id]
            run_data = chunk_df[chunk_df["run_id"] == run_id]

            for col in log_prob_cols:
                token_idx = int(col.replace("log_prob_token_", ""))
                log_probs = run_data[col].dropna()
                if len(log_probs) == 0:
                    continue

                nlls = -log_probs
                aggregated_results.append({
                    "Token Index": token_idx,
                    "Parameters": config["Parameters"],
                    "Num. Parameters": config["Num. Parameters"],
                    "Num. MATH Test Set Replicas": config["Num. MATH Test Set Replicas"],
                    "mean_NLL": nlls.mean(),
                    "std_NLL": nlls.std(),
                    "count": len(nlls),
                    "run_id": run_id,
                })

        del chunk_df
        gc.collect()

    # Aggregate across runs
    per_run_df = pd.DataFrame(aggregated_results)
    nll_df = (
        per_run_df.groupby(
            ["Token Index", "Parameters", "Num. Parameters", "Num. MATH Test Set Replicas"]
        )
        .agg(mean_NLL=("mean_NLL", "mean"), std_NLL=("std_NLL", "mean"), count=("count", "sum"))
        .reset_index()
    )
    nll_df["sem_NLL"] = nll_df["std_NLL"] / np.sqrt(nll_df["count"])

    return nll_df


def compute_nll_trends(nll_df: pd.DataFrame) -> pd.DataFrame:
    """Compute NLL trend (slope in log-log space) for each condition."""
    unique_params = sorted(
        nll_df["Parameters"].unique(),
        key=lambda x: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[x]
    )
    unique_replicas = sorted(nll_df["Num. MATH Test Set Replicas"].unique())

    trend_results = []

    for param in unique_params:
        for replica in unique_replicas:
            subset = nll_df[
                (nll_df["Parameters"] == param)
                & (nll_df["Num. MATH Test Set Replicas"] == replica)
            ].sort_values("Token Index")

            if len(subset) < 10:
                continue

            t = subset["Token Index"].values
            nll = subset["mean_NLL"].values

            # Linear regression on log(NLL) vs log(t+1)
            log_t = np.log(t + 1)
            log_nll = np.log(nll)
            valid = np.isfinite(log_t) & np.isfinite(log_nll)

            if valid.sum() < 10:
                continue

            slope, _, r_value, p_value, _ = stats.linregress(log_t[valid], log_nll[valid])

            # Late slope (token indices 400-800)
            late_mask = (t >= 400) & valid
            if late_mask.sum() >= 10:
                late_slope, _, late_r, late_p, _ = stats.linregress(
                    log_t[late_mask], log_nll[late_mask]
                )
            else:
                late_slope, late_r, late_p = np.nan, np.nan, np.nan

            trend_results.append({
                "Parameters": param,
                "Num. Replicas": replica,
                "Overall Slope": slope,
                "Overall R²": r_value**2,
                "Overall p-value": p_value,
                "Late Slope (400-800)": late_slope,
                "Late R²": late_r**2 if not np.isnan(late_r) else np.nan,
                "Late p-value": late_p,
                "NLL Start": nll[0],
                "NLL End": nll[-1],
                "NLL Change": nll[-1] - nll[0],
                "NLL % Change": (nll[-1] - nll[0]) / nll[0] * 100,
            })

    return pd.DataFrame(trend_results)


def analyze_problematic_conditions(nll_df: pd.DataFrame) -> None:
    """Print detailed analysis for problematic conditions."""
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF PROBLEMATIC CONDITIONS")
    print("=" * 70)

    for param, replica in PROBLEMATIC_CONDITIONS:
        print(f"\n{'='*50}")
        print(f"Model: {param}, Replicas: {replica}")
        print("=" * 50)

        subset = nll_df[
            (nll_df["Parameters"] == param)
            & (nll_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) == 0:
            print("No data for this condition")
            continue

        t = subset["Token Index"].values
        nll = subset["mean_NLL"].values
        count = subset["count"].values

        print(f"Token range: {t.min()} to {t.max()}")
        print(f"NLL range: {nll.min():.4f} to {nll.max():.4f}")
        print(f"Count range: {count.min()} to {count.max()}")

        min_idx, max_idx = np.argmin(nll), np.argmax(nll)
        print(f"NLL minimum at token {t[min_idx]}: {nll[min_idx]:.4f} (count={count[min_idx]})")
        print(f"NLL maximum at token {t[max_idx]}: {nll[max_idx]:.4f} (count={count[max_idx]})")

        early_count = count[t < 100].mean()
        late_count = count[t >= 400].mean()
        print(f"Average count (t<100): {early_count:.0f}")
        print(f"Average count (t>=400): {late_count:.0f}")
        print(f"Count ratio (late/early): {late_count/early_count:.2%}")

        late_mask = t >= 400
        if late_mask.sum() > 10:
            corr, p = stats.pearsonr(count[late_mask], nll[late_mask])
            print(f"Correlation between count and NLL (t>=400): r={corr:.3f}, p={p:.4f}")


def plot_problematic_conditions(nll_df: pd.DataFrame, results_dir: str) -> None:
    """Generate diagnostic plots for problematic conditions."""
    fig, axes = plt.subplots(
        len(PROBLEMATIC_CONDITIONS), 2,
        figsize=(14, 4 * len(PROBLEMATIC_CONDITIONS))
    )

    for row, (param, replica) in enumerate(PROBLEMATIC_CONDITIONS):
        subset = nll_df[
            (nll_df["Parameters"] == param)
            & (nll_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) == 0:
            continue

        t = subset["Token Index"].values
        nll = subset["mean_NLL"].values
        sem = subset["sem_NLL"].values
        count = subset["count"].values

        # Left: NLL vs token index
        ax1 = axes[row, 0]
        ax1.errorbar(t, nll, yerr=1.96 * sem, fmt='-', alpha=0.7, linewidth=1, capsize=0)
        ax1.set_xlabel("Token Index")
        ax1.set_ylabel("Mean NLL")
        ax1.set_title(f"{param}, R={replica}: NLL vs Token Index")
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3)

        # Right: Count vs token index
        ax2 = axes[row, 1]
        ax2.plot(t, count, 'g-', alpha=0.7)
        ax2.set_xlabel("Token Index")
        ax2.set_ylabel("Count (num. sequences)")
        ax2.set_title(f"{param}, R={replica}: Sample Count vs Token Index")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename="y=nll_and_count_x=token_index_problematic_conditions",
    )
    plt.close()


def compute_early_late_comparison(nll_df: pd.DataFrame) -> pd.DataFrame:
    """Compare early (token 1-50) vs late (token 600-800) NLL across conditions."""
    unique_params = sorted(
        nll_df["Parameters"].unique(),
        key=lambda x: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[x]
    )
    unique_replicas = sorted(nll_df["Num. MATH Test Set Replicas"].unique())

    results = []
    for param in unique_params:
        for replica in unique_replicas:
            subset = nll_df[
                (nll_df["Parameters"] == param)
                & (nll_df["Num. MATH Test Set Replicas"] == replica)
            ].sort_values("Token Index")

            if len(subset) < 10:
                continue

            t = subset["Token Index"].values
            nll = subset["mean_NLL"].values
            count = subset["count"].values

            early_mask = (t >= 1) & (t <= 50)
            late_mask = (t >= 600) & (t <= 800)

            if early_mask.sum() < 5 or late_mask.sum() < 5:
                continue

            early_nll = nll[early_mask].mean()
            late_nll = nll[late_mask].mean()
            early_count = count[early_mask].mean()
            late_count = count[late_mask].mean()

            results.append({
                "Parameters": param,
                "Num. Replicas": replica,
                "Early NLL (1-50)": early_nll,
                "Late NLL (600-800)": late_nll,
                "Early Count": early_count,
                "Late Count": late_count,
                "Count Drop %": (1 - late_count / early_count) * 100,
                "NLL Change %": (late_nll - early_nll) / early_nll * 100,
            })

    return pd.DataFrame(results)


def plot_nll_vs_count_change(comparison_df: pd.DataFrame, results_dir: str) -> None:
    """Plot NLL change vs count drop for all conditions."""
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_replicas = sorted(comparison_df["Num. Replicas"].unique())
    for replica in unique_replicas:
        sub = comparison_df[comparison_df["Num. Replicas"] == replica]
        ax.scatter(
            sub["Count Drop %"], sub["NLL Change %"],
            label=f"R={replica}", s=100, alpha=0.7
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["Parameters"],
                (row["Count Drop %"], row["NLL Change %"]),
                fontsize=8, alpha=0.7
            )

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Count Drop % (early to late)")
    ax.set_ylabel("NLL Change % (early to late)")
    ax.legend(title="Replicas")
    ax.grid(True, alpha=0.3)

    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename="y=nll_change_pct_x=count_drop_pct",
    )
    plt.close()


def main():
    """Run the NLL uptick analysis."""
    # Setup directories
    data_dir, results_dir = src.analyze.setup_notebook_dir(
        notebook_dir=os.path.dirname(os.path.abspath(__file__)),
        refresh=False,
    )

    # Load data
    configs_df = load_run_configs(data_dir)
    nll_df = load_nll_data(data_dir, configs_df)
    print(f"Aggregated shape: {nll_df.shape}")

    unique_params = sorted(
        nll_df["Parameters"].unique(),
        key=lambda x: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[x]
    )
    unique_replicas = sorted(nll_df["Num. MATH Test Set Replicas"].unique())
    print(f"Model sizes: {unique_params}")
    print(f"Replica levels: {unique_replicas}")

    # Compute trends
    print("\n" + "=" * 70)
    print("NLL TREND ANALYSIS")
    print("=" * 70)

    trend_df = compute_nll_trends(nll_df)

    # Display increasing NLL conditions
    print("\n" + "=" * 70)
    print("CONDITIONS WITH POSITIVE SLOPE (NLL INCREASES WITH TOKEN INDEX)")
    print("=" * 70)

    increasing = trend_df[trend_df["Overall Slope"] > 0].sort_values(
        "Overall Slope", ascending=False
    )
    print(f"\nFound {len(increasing)} conditions with increasing NLL:")
    print(increasing[[
        "Parameters", "Num. Replicas", "Overall Slope", "Overall R²",
        "NLL Start", "NLL End", "NLL % Change"
    ]].to_string(index=False))

    # Display late uptick conditions
    print("\n" + "=" * 70)
    print("CONDITIONS WITH POSITIVE LATE SLOPE (NLL UPTICK IN TOKEN 400-800)")
    print("=" * 70)

    late_uptick = trend_df[trend_df["Late Slope (400-800)"] > 0].sort_values(
        "Late Slope (400-800)", ascending=False
    )
    print(f"\nFound {len(late_uptick)} conditions with late uptick:")
    print(late_uptick[[
        "Parameters", "Num. Replicas", "Late Slope (400-800)", "Late R²",
        "NLL Start", "NLL End", "NLL % Change"
    ]].to_string(index=False))

    # All conditions sorted by slope
    print("\n" + "=" * 70)
    print("ALL CONDITIONS SORTED BY OVERALL SLOPE")
    print("=" * 70)
    print(trend_df.sort_values("Overall Slope", ascending=False)[[
        "Parameters", "Num. Replicas", "Overall Slope", "NLL Start", "NLL End", "NLL % Change"
    ]].to_string(index=False))

    # Detailed analysis
    analyze_problematic_conditions(nll_df)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 70)

    plot_problematic_conditions(nll_df, results_dir)

    # Selection bias hypothesis
    print("\n" + "=" * 70)
    print("HYPOTHESIS: SELECTION BIAS DUE TO SEQUENCE LENGTH")
    print("=" * 70)
    print("""
The key insight is that not all sequences have the same length.
- Short sequences end early (e.g., at token 100)
- Long sequences continue to token 800

If short sequences are 'easier' (lower NLL), then:
- At early tokens: we average over ALL sequences (easy + hard)
- At late tokens: we only average over LONG sequences (potentially harder)

This creates an artificial uptick in NLL at late positions.

The 'count' column shows how many sequences contribute to each token position.
If count drops and NLL rises, selection bias is likely the explanation.
""")

    # Early vs late comparison
    print("\n" + "=" * 70)
    print("EARLY VS LATE COMPARISON ACROSS ALL CONDITIONS")
    print("=" * 70)

    comparison_df = compute_early_late_comparison(nll_df)
    print(comparison_df.sort_values("NLL Change %", ascending=False).to_string(index=False))

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION: COUNT DROP VS NLL CHANGE")
    print("=" * 70)

    # Check if count drop has variance (if all identical, correlation is undefined)
    count_drop_std = comparison_df["Count Drop %"].std()
    if count_drop_std < 0.01:
        print(f"Count Drop % is nearly constant ({comparison_df['Count Drop %'].mean():.2f}%)")
        print("Correlation undefined - count drop does not vary across conditions.")
        print("This means sequence length distribution is identical for all conditions.")
    else:
        corr, p = stats.pearsonr(comparison_df["Count Drop %"], comparison_df["NLL Change %"])
        print(f"Correlation: r = {corr:.3f}, p = {p:.4f}")

        if corr > 0.3 and p < 0.05:
            print("\n*** CONFIRMED: Count drop is correlated with NLL increase ***")
        else:
            print("\nNo strong correlation found - other factors may be at play")

    # Generate comparison plot
    plot_nll_vs_count_change(comparison_df, results_dir)

    print("\nPlots saved to results directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
