"""Analyze teacher-forced evaluation results: NLL by token position.

This notebook analyzes how negative log likelihood varies across token positions
in the solution, broken down by model size and contamination level (num replicas).
"""

import ast
import gc
import hashlib
import os
import re

import matplotlib
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

import src.analyze
import src.globals
import src.plot

# Add missing model size mappings (62M and 153M are used in this notebook)
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["62M"] = 62e6
src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["153M"] = 153e6


def enable_minor_gridlines(g):
    """Enable minor gridlines on all axes of a FacetGrid."""
    for ax in g.axes.flat:
        ax.grid(which="minor", alpha=0.3, linewidth=0.5)
        ax.grid(which="major", alpha=0.5, linewidth=0.8)
        # Ensure minor ticks are shown on log scales
        ax.xaxis.set_minor_locator(LogLocator(subs="all", numticks=100))
        ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=100))


# %%
# Setup
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# %%
# Load evaluation run configs
eval_teacher_forcing_sweep_ids = [
    "9vtnq3bd",  # Qwen 3   34M     1xOT    Subset Fraction=1.0
    "ovps81c2",  # Qwen 3   62M     1xOT    Subset Fraction=1.0
    "oi9x67mh",  # Qwen 3   93M     1xOT    Subset Fraction=1.0
    "em23bzb7",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "sy8h8i80",  # Qwen 3  344M     1xOT    Subset Fraction=1.0
]

eval_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
    data_dir=data_dir,
    sweep_ids=eval_teacher_forcing_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

# Extract model metadata from config
eval_runs_configs_df["Model"] = eval_runs_configs_df["model_config"].apply(
    lambda model_config: ast.literal_eval(model_config)["model"]
)
eval_runs_configs_df["Parameters"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: re.search(r"Qwen3-([\d.]+[MB])", model_name).group(1)
)
eval_runs_configs_df["Num. Parameters"] = eval_runs_configs_df["Parameters"].apply(
    lambda parameters: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[parameters]
)
eval_runs_configs_df["Num. Replicas Per Epoch"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"rep_(\d+)_sbst", model_name).group(1))
)
eval_runs_configs_df["Num. Epochs"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
)
eval_runs_configs_df["Num. MATH Test Set Replicas"] = (
    eval_runs_configs_df["Num. Replicas Per Epoch"]
    * eval_runs_configs_df["Num. Epochs"]
)

# %%
# Load and process histories data (memory-efficient chunked processing)
filename = "sweeps=" + ",".join(eval_teacher_forcing_sweep_ids)
hashed_filename = hashlib.md5(filename.encode()).hexdigest()
histories_path = os.path.join(data_dir, hashed_filename + "_runs_histories.parquet")

print(f"Loading histories from {histories_path}")

# Create mapping from run_id to config values
run_id_to_config = eval_runs_configs_df.set_index("run_id")[
    ["Parameters", "Num. Parameters", "Num. MATH Test Set Replicas"]
].to_dict("index")

# Process parquet in row groups to manage memory
print("Processing data in chunked, memory-efficient manner...")

parquet_file = pq.ParquetFile(histories_path)
all_columns = [field.name for field in parquet_file.schema]

# Get log_prob columns, filtered to token index <= 800
# (only ~2% of sequences reach 800+ tokens, so data beyond is noisy)
MAX_TOKEN_INDEX = 800
all_log_prob_cols = sorted(
    [
        c
        for c in all_columns
        if c.startswith("log_prob_token_")
        and int(c.replace("log_prob_token_", "")) <= MAX_TOKEN_INDEX
    ],
    key=lambda x: int(x.replace("log_prob_token_", "")),
)
print(f"Using token indices 0 to {MAX_TOKEN_INDEX}")

# Sample ~200 token positions uniformly in log space
num_samples = 200
log_spaced_indices = np.unique(
    np.geomspace(1, MAX_TOKEN_INDEX + 1, num_samples).astype(int) - 1
).tolist()
if 0 not in log_spaced_indices:
    log_spaced_indices = [0] + log_spaced_indices
log_spaced_indices = sorted(log_spaced_indices)

log_prob_cols = [
    f"log_prob_token_{i}"
    for i in log_spaced_indices
    if f"log_prob_token_{i}" in all_log_prob_cols
]
print(
    f"Using {len(log_prob_cols)} token positions (log-spaced from 0 to {MAX_TOKEN_INDEX})"
)

# Process each row group for per-token NLL stats
aggregated_results = []
num_row_groups = parquet_file.metadata.num_row_groups

for rg_idx in range(num_row_groups):
    print(f"  Processing row group {rg_idx + 1}/{num_row_groups}...")

    cols_to_load = ["run_id"] + log_prob_cols
    table = parquet_file.read_row_group(rg_idx, columns=cols_to_load)
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
            aggregated_results.append(
                {
                    "Token Index": token_idx,
                    "Parameters": config["Parameters"],
                    "Num. Parameters": config["Num. Parameters"],
                    "Num. MATH Test Set Replicas": config[
                        "Num. MATH Test Set Replicas"
                    ],
                    "mean_NLL": nlls.mean(),
                    "std_NLL": nlls.std(),
                    "count": len(nlls),
                    "run_id": run_id,
                }
            )

    del chunk_df
    gc.collect()

# %%
# Second pass: compute per-sequence cumulative NLLs
# This is expensive, so we cache the results
cumulative_nll_cache_path = os.path.join(
    data_dir, hashed_filename + "_cumulative_nll_per_run.parquet"
)

if os.path.exists(cumulative_nll_cache_path) and not refresh:
    print(f"Loading cached cumulative NLL data from {cumulative_nll_cache_path}")
    per_run_cumulative_df = pd.read_parquet(cumulative_nll_cache_path)
else:
    print("Computing per-sequence cumulative NLLs (this may take a while)...")
    cumulative_nll_results = []

    # Process columns in batches to avoid memory issues
    batch_size = 200  # Number of columns to load at a time

    for rg_idx in range(num_row_groups):
        print(
            f"  Processing row group {rg_idx + 1}/{num_row_groups} for cumulative NLLs..."
        )

        # First, load run_id to identify sequences
        table = parquet_file.read_row_group(rg_idx, columns=["run_id"])
        run_ids_df = table.to_pandas()
        del table
        gc.collect()

        # For each run, compute cumulative NLLs by processing columns in batches
        for run_id in run_ids_df["run_id"].unique():
            if run_id not in run_id_to_config:
                continue

            config = run_id_to_config[run_id]
            run_mask = run_ids_df["run_id"] == run_id
            num_sequences = run_mask.sum()

            # Initialize running cumulative NLL sum for each sequence
            cumulative_nll_running = np.zeros(num_sequences, dtype=np.float64)

            # Track cumulative NLL at log-spaced positions
            cumulative_at_positions = {}

            # Process columns in batches
            for batch_start in range(0, len(all_log_prob_cols), batch_size):
                batch_end = min(batch_start + batch_size, len(all_log_prob_cols))
                batch_cols = all_log_prob_cols[batch_start:batch_end]

                # Load batch of columns
                table = parquet_file.read_row_group(
                    rg_idx, columns=["run_id"] + batch_cols
                )
                batch_df = table.to_pandas()
                del table
                gc.collect()

                # Filter to current run
                run_batch_data = batch_df[batch_df["run_id"] == run_id][
                    batch_cols
                ].values
                del batch_df
                gc.collect()

                # Convert to NLL and add to running cumulative sum
                nll_batch = -run_batch_data
                # Handle NaN by replacing with 0 (no contribution to cumulative sum)
                nll_batch = np.nan_to_num(nll_batch, nan=0.0)

                for col_offset, col in enumerate(batch_cols):
                    token_idx = int(col.replace("log_prob_token_", ""))
                    cumulative_nll_running += nll_batch[:, col_offset]

                    # Record cumulative NLL at log-spaced positions
                    if token_idx in log_spaced_indices:
                        cumulative_at_positions[
                            token_idx
                        ] = cumulative_nll_running.copy()

                del run_batch_data, nll_batch
                gc.collect()

            # Store results for each log-spaced position
            for token_idx in sorted(cumulative_at_positions.keys()):
                cumulative_nlls = cumulative_at_positions[token_idx]
                # Filter out sequences that are too short (would have 0 cumulative NLL)
                if cumulative_nlls.max() == 0:
                    continue

                cumulative_nll_results.append(
                    {
                        "Token Index": token_idx,
                        "Parameters": config["Parameters"],
                        "Num. Parameters": config["Num. Parameters"],
                        "Num. MATH Test Set Replicas": config[
                            "Num. MATH Test Set Replicas"
                        ],
                        "mean_cumulative_NLL": cumulative_nlls.mean(),
                        "std_cumulative_NLL": cumulative_nlls.std(),
                        "count": len(cumulative_nlls),
                        "run_id": run_id,
                    }
                )

            del cumulative_at_positions
            gc.collect()

        del run_ids_df
        gc.collect()

    print("Creating cumulative NLL dataframe...")
    per_run_cumulative_df = pd.DataFrame(cumulative_nll_results)
    del cumulative_nll_results
    gc.collect()

    # Cache the results
    print(f"Caching cumulative NLL data to {cumulative_nll_cache_path}")
    per_run_cumulative_df.to_parquet(cumulative_nll_cache_path, index=False)

# Aggregate results
print("Creating aggregated dataframe...")
per_run_df = pd.DataFrame(aggregated_results)
del aggregated_results
gc.collect()

# Combine across row groups if needed
per_run_df = (
    per_run_df.groupby(
        [
            "Token Index",
            "Parameters",
            "Num. Parameters",
            "Num. MATH Test Set Replicas",
            "run_id",
        ]
    )
    .agg({"mean_NLL": "mean", "std_NLL": "mean", "count": "sum"})
    .reset_index()
)

# Aggregate across runs for same experimental conditions
nll_by_token_df = (
    per_run_df.groupby(
        ["Token Index", "Parameters", "Num. Parameters", "Num. MATH Test Set Replicas"]
    )
    .agg(
        mean_NLL=("mean_NLL", "mean"),
        std_NLL=("std_NLL", "mean"),
        count=("count", "sum"),
    )
    .reset_index()
)
nll_by_token_df["sem_NLL"] = nll_by_token_df["std_NLL"] / np.sqrt(
    nll_by_token_df["count"]
)

print(f"Aggregated shape: {nll_by_token_df.shape}")
del per_run_df
gc.collect()

# Aggregate cumulative NLL data across runs for same experimental conditions
cumulative_nll_by_token_df = (
    per_run_cumulative_df.groupby(
        ["Token Index", "Parameters", "Num. Parameters", "Num. MATH Test Set Replicas"]
    )
    .agg(
        mean_cumulative_NLL=("mean_cumulative_NLL", "mean"),
        std_cumulative_NLL=("std_cumulative_NLL", "mean"),
        count=("count", "sum"),
    )
    .reset_index()
)
cumulative_nll_by_token_df["sem_cumulative_NLL"] = cumulative_nll_by_token_df[
    "std_cumulative_NLL"
] / np.sqrt(cumulative_nll_by_token_df["count"])
cumulative_nll_by_token_df["Token Index + 1"] = (
    cumulative_nll_by_token_df["Token Index"] + 1
)

print(f"Cumulative NLL aggregated shape: {cumulative_nll_by_token_df.shape}")
del per_run_cumulative_df
gc.collect()

# %%
# Plot 1: NLL vs Token Index (hue=num_replicas, col=model_size)

# Add 1 to token index for log scaling (log(0) is undefined)
nll_by_token_df["Token Index + 1"] = nll_by_token_df["Token Index"] + 1

# Check unique values
unique_replicas = sorted(nll_by_token_df["Num. MATH Test Set Replicas"].unique())
unique_params = sorted(
    nll_by_token_df["Parameters"].unique(),
    key=lambda x: int(x.replace("M", "").replace("B", "000")),
)
print(f"Unique replica values: {unique_replicas}")
print(f"Unique parameter values: {unique_params}")

# Convert replicas to string for categorical hue (ensures legend shows actual values)
nll_by_token_df["Replicas"] = nll_by_token_df["Num. MATH Test Set Replicas"].astype(str)

# Create color palette mapping for replica values (viridis for replicas)
# Use SymLogNorm for consistent colors with notebooks 10/11 (continuous mapping, not discrete)
from matplotlib.colors import SymLogNorm

all_replicas = [0, 1, 3, 10, 32, 100, 316, 1000, 3162]
replica_sym_norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=max(all_replicas))
viridis_cmap = matplotlib.colormaps["viridis"]
replica_palette = {str(r): viridis_cmap(replica_sym_norm(r)) for r in all_replicas}

# Filter to Token Index <= 800 to avoid noisy tail (but keep ALL replicas)
plot1_df = nll_by_token_df[nll_by_token_df["Token Index + 1"] <= 800].copy()

# Compute CI bounds (mean ± 1.96 * SEM for 95% CI)
plot1_df["ci_lower"] = plot1_df["mean_NLL"] - 1.96 * plot1_df["sem_NLL"]
plot1_df["ci_upper"] = plot1_df["mean_NLL"] + 1.96 * plot1_df["sem_NLL"]
# Clip lower bound to small positive value for log scale
plot1_df["ci_lower"] = plot1_df["ci_lower"].clip(lower=1e-6)

# For cleaner legend, only show subset of replica values
replicas_for_legend = [0, 1, 10, 100, 1000]

plt.close()
g = sns.relplot(
    data=plot1_df,
    x="Token Index + 1",
    y="mean_NLL",
    hue="Replicas",
    hue_order=[str(r) for r in unique_replicas],  # Plot ALL replicas
    palette=replica_palette,
    col="Parameters",
    col_order=unique_params,
    col_wrap=3,
    kind="line",
    facet_kws={"sharey": False, "sharex": True},
    height=4,
    aspect=1.2,
)

# Add uncertainty bands for ALL replicas
for ax, param in zip(g.axes.flat, unique_params):
    for replica in unique_replicas:
        subset = plot1_df[
            (plot1_df["Parameters"] == param)
            & (plot1_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")
        if len(subset) == 0:
            continue
        color = replica_palette[str(replica)]
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower"],
            subset["ci_upper"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

g.set(
    xlabel=r"Token Index",
    ylabel=r"Negative Log Likelihood",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
)
g.set_titles(r"{col_name}")

# Filter legend to show only subset of replicas
handles, labels = g.axes.flat[0].get_legend_handles_labels()
filtered_handles = []
filtered_labels = []
for h, l in zip(handles, labels):
    if l in [str(r) for r in replicas_for_legend]:
        filtered_handles.append(h)
        filtered_labels.append(l)

# Remove old legend and add filtered one
g._legend.remove()
g.fig.legend(
    filtered_handles,
    filtered_labels,
    loc="center",
    bbox_to_anchor=(0.85, 0.25),
    title=r"Num. Replicas",
    frameon=True,
)
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=nll_x=token_index_hue=num_replicas_col=model_size",
)
# plt.show()

# %%
# Plot 2: NLL vs Token Index (hue=model_size, col=num_replicas) - TRANSPOSED

# Create color palette for model sizes (flare for Num. Parameters, with LogNorm)
# Use LogNorm to sample colors at the same positions as notebook 11
param_values = [src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p] for p in unique_params]
num_parameters_log_norm = LogNorm(vmin=min(param_values), vmax=max(param_values))
flare_cmap = matplotlib.colormaps["flare"]
params_palette = {
    p: flare_cmap(num_parameters_log_norm(src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p]))
    for p in unique_params
}

# Filter to Token Index <= 800 (only ~2% of sequences reach 800+ tokens, so high noise beyond)
plot2_df = nll_by_token_df[nll_by_token_df["Token Index + 1"] <= 800].copy()
plot2_df["ci_lower"] = plot2_df["mean_NLL"] - 1.96 * plot2_df["sem_NLL"]
plot2_df["ci_upper"] = plot2_df["mean_NLL"] + 1.96 * plot2_df["sem_NLL"]
plot2_df["ci_lower"] = plot2_df["ci_lower"].clip(lower=1e-6)

# Convert replicas to sorted order for columns
replica_col_order = [str(r) for r in unique_replicas]

plt.close()
g = sns.relplot(
    data=plot2_df,
    x="Token Index + 1",
    y="mean_NLL",
    hue="Parameters",
    hue_order=unique_params,
    palette=params_palette,
    col="Replicas",
    col_order=replica_col_order,
    col_wrap=3,
    kind="line",
    facet_kws={"sharey": False, "sharex": True},
    height=4,
    aspect=1.2,
)

# Add uncertainty bands
for ax, replica_str in zip(g.axes.flat, replica_col_order):
    replica = int(replica_str)
    for param in unique_params:
        subset = plot2_df[
            (plot2_df["Parameters"] == param)
            & (plot2_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")
        if len(subset) == 0:
            continue
        color = params_palette[param]
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower"],
            subset["ci_upper"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

g.set(
    xlabel=r"Token Index",
    ylabel=r"Negative Log Likelihood",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
)
g.set_titles(r"Replicas: {col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title=r"Model Size")
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=nll_x=token_index_hue=model_size_col=num_replicas",
)
# plt.show()

# %%
# Plot 3: Probability vs Token Index (hue=num_replicas, col=model_size)
# prob = exp(-NLL)

# Compute probability and CI bounds
# Note: CI bounds swap because exp(-x) is decreasing
plot3_df = plot1_df.copy()
plot3_df["mean_prob"] = np.exp(-plot3_df["mean_NLL"])
plot3_df["ci_lower_prob"] = np.exp(-plot3_df["ci_upper"])  # swap!
plot3_df["ci_upper_prob"] = np.exp(-plot3_df["ci_lower"])  # swap!

plt.close()
g = sns.relplot(
    data=plot3_df,
    x="Token Index + 1",
    y="mean_prob",
    hue="Replicas",
    hue_order=[str(r) for r in unique_replicas],  # Plot ALL replicas
    palette=replica_palette,
    col="Parameters",
    col_order=unique_params,
    col_wrap=3,
    kind="line",
    facet_kws={"sharey": False, "sharex": True},
    height=4,
    aspect=1.2,
)

# Add uncertainty bands for ALL replicas
for ax, param in zip(g.axes.flat, unique_params):
    for replica in unique_replicas:
        subset = plot3_df[
            (plot3_df["Parameters"] == param)
            & (plot3_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")
        if len(subset) == 0:
            continue
        color = replica_palette[str(replica)]
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower_prob"],
            subset["ci_upper_prob"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

g.set(
    xlabel=r"Token Index",
    ylabel=r"Per-Token Probability",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
)
g.set_titles(r"{col_name}")

# Filter legend to show only subset of replicas
handles, labels = g.axes.flat[0].get_legend_handles_labels()
filtered_handles = []
filtered_labels = []
for h, l in zip(handles, labels):
    if l in [str(r) for r in replicas_for_legend]:
        filtered_handles.append(h)
        filtered_labels.append(l)

# Remove old legend and add filtered one
g._legend.remove()
g.fig.legend(
    filtered_handles,
    filtered_labels,
    loc="center",
    bbox_to_anchor=(0.85, 0.25),
    title=r"Num. Replicas",
    frameon=True,
)
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=prob_x=token_index_hue=num_replicas_col=model_size",
)
# plt.show()

# %%
# Plot 4: Probability vs Token Index (hue=model_size, col=num_replicas) - TRANSPOSED

# Compute probability and CI bounds
plot4_df = plot2_df.copy()
plot4_df["mean_prob"] = np.exp(-plot4_df["mean_NLL"])
plot4_df["ci_lower_prob"] = np.exp(-plot4_df["ci_upper"])  # swap!
plot4_df["ci_upper_prob"] = np.exp(-plot4_df["ci_lower"])  # swap!

plt.close()
g = sns.relplot(
    data=plot4_df,
    x="Token Index + 1",
    y="mean_prob",
    hue="Parameters",
    hue_order=unique_params,
    palette=params_palette,
    col="Replicas",
    col_order=replica_col_order,
    col_wrap=3,
    kind="line",
    facet_kws={"sharey": False, "sharex": True},
    height=4,
    aspect=1.2,
)

# Add uncertainty bands
for ax, replica_str in zip(g.axes.flat, replica_col_order):
    replica = int(replica_str)
    for param in unique_params:
        subset = plot4_df[
            (plot4_df["Parameters"] == param)
            & (plot4_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")
        if len(subset) == 0:
            continue
        color = params_palette[param]
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower_prob"],
            subset["ci_upper_prob"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

g.set(
    xlabel=r"Token Index",
    ylabel=r"Per-Token Probability",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
)
g.set_titles(r"Replicas: {col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title=r"Model Size")
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=prob_x=token_index_hue=model_size_col=num_replicas",
)
# plt.show()

# %%
# Plot 5: Cumulative Probability vs Token Index (hue=num_replicas, col=model_size)
# Cumulative probability = exp(-cumulative_NLL) where cumulative_NLL is computed
# per-sequence and then averaged (geometric mean of cumulative probabilities)

# Add Replicas column for categorical hue
cumulative_nll_by_token_df["Replicas"] = cumulative_nll_by_token_df[
    "Num. MATH Test Set Replicas"
].astype(str)

# Compute cumulative probability (geometric mean across sequences)
# mean_cumulative_NLL is average of per-sequence cumulative NLLs
# So exp(-mean_cumulative_NLL) is the geometric mean of cumulative probabilities
cumulative_nll_by_token_df["cumulative_prob"] = np.exp(
    -cumulative_nll_by_token_df["mean_cumulative_NLL"]
)

# CI bounds for cumulative NLL
cumulative_nll_by_token_df["ci_lower_nll"] = (
    cumulative_nll_by_token_df["mean_cumulative_NLL"]
    - 1.96 * cumulative_nll_by_token_df["sem_cumulative_NLL"]
)
cumulative_nll_by_token_df["ci_upper_nll"] = (
    cumulative_nll_by_token_df["mean_cumulative_NLL"]
    + 1.96 * cumulative_nll_by_token_df["sem_cumulative_NLL"]
)
# CI bounds for probability (swap due to exp(-x) being decreasing)
cumulative_nll_by_token_df["ci_lower_prob"] = np.exp(
    -cumulative_nll_by_token_df["ci_upper_nll"]
)
cumulative_nll_by_token_df["ci_upper_prob"] = np.exp(
    -cumulative_nll_by_token_df["ci_lower_nll"]
)

# Filter to Token Index <= 800 for consistency with plot 1
plot5_df = cumulative_nll_by_token_df[
    cumulative_nll_by_token_df["Token Index + 1"] <= 800
].copy()

plt.close()
g = sns.relplot(
    data=plot5_df,
    x="Token Index + 1",
    y="cumulative_prob",
    hue="Replicas",
    hue_order=[str(r) for r in unique_replicas],  # Plot ALL replicas
    palette=replica_palette,
    col="Parameters",
    col_order=unique_params,
    col_wrap=3,
    kind="line",
    facet_kws={"sharey": False, "sharex": True},
    height=4,
    aspect=1.2,
)

# Add uncertainty bands for ALL replicas
for ax, param in zip(g.axes.flat, unique_params):
    for replica in unique_replicas:
        subset = plot5_df[
            (plot5_df["Parameters"] == param)
            & (plot5_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")
        if len(subset) == 0:
            continue
        color = replica_palette[str(replica)]
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower_prob"],
            subset["ci_upper_prob"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

g.set(
    xlabel=r"Token Index",
    ylabel=r"Cumulative Probability",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
    ylim=(1e-6, 1),  # Truncate y-axis to avoid very small probabilities
)
g.set_titles(r"{col_name}")

# Filter legend to show only subset of replicas
handles, labels = g.axes.flat[0].get_legend_handles_labels()
filtered_handles = []
filtered_labels = []
for h, l in zip(handles, labels):
    if l in [str(r) for r in replicas_for_legend]:
        filtered_handles.append(h)
        filtered_labels.append(l)

# Remove old legend and add filtered one
g._legend.remove()
g.fig.legend(
    filtered_handles,
    filtered_labels,
    loc="center",
    bbox_to_anchor=(0.85, 0.25),
    title=r"Num. Replicas",
    frameon=True,
)
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cumulative_prob_x=token_index_hue=num_replicas_col=model_size",
)
# plt.show()

# %%
# Combined 2-row, 3-column figure for paper (34M, 93M, 344M)
# Row 1: Cumulative probability
# Row 2: NLL with fitted floor model curves
# Note: This combined figure is created later after floor_fit_df is available
# See section after floor model fitting

# %%
# =============================================================================
# CURVE FITTING: NLL vs Token Index
# =============================================================================
# Goal: Find the best functional form for NLL(t, N, R) where:
#   t = token index
#   N = number of parameters (model size)
#   R = number of test set replicas

from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_r_squared(y_true, y_pred):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Define candidate functional forms
# Each returns NLL as a function of token index t


def model_constant(t, a):
    """Constant: NLL(t) = a"""
    return np.full_like(t, a, dtype=float)


def model_linear(t, a, b):
    """Linear: NLL(t) = a + b*t"""
    return a + b * t


def model_log(t, a, b):
    """Logarithmic: NLL(t) = a + b*log(t+1)"""
    return a + b * np.log(t + 1)


def model_power_law(t, a, b, c):
    """Power law: NLL(t) = a + b*(t+1)^(-c)"""
    return a + b * (t + 1) ** (-c)


def model_exp_decay(t, nll_inf, nll_0, tau):
    """Exponential decay: NLL(t) = nll_inf + (nll_0 - nll_inf)*exp(-t/tau)"""
    return nll_inf + (nll_0 - nll_inf) * np.exp(-t / tau)


def model_exp_decay_constrained(t, nll_inf, delta_nll, tau):
    """Exponential decay (constrained): NLL(t) = nll_inf + delta_nll*exp(-t/tau)
    where delta_nll >= 0 ensures NLL decreases toward nll_inf"""
    return nll_inf + np.abs(delta_nll) * np.exp(-t / np.abs(tau))


def model_stretched_exp(t, nll_inf, delta_nll, tau, beta):
    """Stretched exponential: NLL(t) = nll_inf + delta_nll*exp(-(t/tau)^beta)"""
    return nll_inf + np.abs(delta_nll) * np.exp(-((t / np.abs(tau)) ** np.abs(beta)))


def model_rational(t, a, b, c):
    """Rational: NLL(t) = a + b / (t + c)"""
    return a + b / (t + np.abs(c))


# Dictionary of models with their initial parameter guesses and bounds
MODELS = {
    "constant": {
        "func": model_constant,
        "p0": [1.0],
        "bounds": ([0], [10]),
        "n_params": 1,
    },
    "linear": {
        "func": model_linear,
        "p0": [1.0, 0.0],
        "bounds": ([-10, -1], [10, 1]),
        "n_params": 2,
    },
    "log": {
        "func": model_log,
        "p0": [1.0, -0.1],
        "bounds": ([-10, -10], [10, 10]),
        "n_params": 2,
    },
    "power_law": {
        "func": model_power_law,
        "p0": [0.5, 1.0, 0.5],
        "bounds": ([0, -10, 0.01], [10, 10, 5]),
        "n_params": 3,
    },
    "exp_decay": {
        "func": model_exp_decay_constrained,
        "p0": [0.5, 0.5, 100],
        "bounds": ([0, -5, 1], [10, 5, 1000]),
        "n_params": 3,
    },
    "stretched_exp": {
        "func": model_stretched_exp,
        "p0": [0.5, 0.5, 100, 1.0],
        "bounds": ([0, -5, 1, 0.1], [10, 5, 1000, 3]),
        "n_params": 4,
    },
    "rational": {
        "func": model_rational,
        "p0": [0.5, 10, 10],
        "bounds": ([0, -100, 0.1], [10, 100, 1000]),
        "n_params": 3,
    },
}


def fit_model(t, nll, model_name):
    """Fit a model to data and return results."""
    model_info = MODELS[model_name]
    func = model_info["func"]
    p0 = model_info["p0"]
    bounds = model_info["bounds"]

    try:
        popt, pcov = curve_fit(
            func, t, nll, p0=p0, bounds=bounds, maxfev=10000, method="trf"
        )
        y_pred = func(t, *popt)
        r2 = compute_r_squared(nll, y_pred)
        rmse = compute_rmse(nll, y_pred)
        # Compute AIC: 2k - 2ln(L), where L is likelihood
        # For Gaussian errors: AIC = n*log(RSS/n) + 2k
        n = len(nll)
        k = len(popt)
        rss = np.sum((nll - y_pred) ** 2)
        aic = n * np.log(rss / n) + 2 * k if rss > 0 else np.inf
        bic = n * np.log(rss / n) + k * np.log(n) if rss > 0 else np.inf

        return {
            "params": popt,
            "r2": r2,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "success": True,
        }
    except Exception as e:
        return {
            "params": None,
            "r2": np.nan,
            "rmse": np.nan,
            "aic": np.inf,
            "bic": np.inf,
            "success": False,
            "error": str(e),
        }


# %%
# Fit all models to each experimental condition
print("Fitting models to NLL vs Token Index data...")

fit_results = []

for param in unique_params:
    for replica in unique_replicas:
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        t = subset["Token Index"].values.astype(float)
        nll = subset["mean_NLL"].values

        num_params = subset["Num. Parameters"].iloc[0]

        for model_name in MODELS.keys():
            result = fit_model(t, nll, model_name)
            fit_results.append(
                {
                    "Parameters": param,
                    "Num. Parameters": num_params,
                    "Num. Replicas": replica,
                    "Model": model_name,
                    "R2": result["r2"],
                    "RMSE": result["rmse"],
                    "AIC": result["aic"],
                    "BIC": result["bic"],
                    "Success": result["success"],
                    "Fitted Params": result["params"],
                }
            )

fit_results_df = pd.DataFrame(fit_results)

# %%
# Summarize results: which model fits best across conditions?
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

# Average R² by model
avg_r2_by_model = (
    fit_results_df[fit_results_df["Success"]]
    .groupby("Model")["R2"]
    .agg(["mean", "std", "min", "max"])
    .sort_values("mean", ascending=False)
)
print("\nAverage R² by Model (higher is better):")
print(avg_r2_by_model.round(4))

# Average AIC by model (lower is better)
avg_aic_by_model = (
    fit_results_df[fit_results_df["Success"]]
    .groupby("Model")["AIC"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=True)
)
print("\nAverage AIC by Model (lower is better):")
print(avg_aic_by_model.round(2))

# Best model for each condition
best_models = fit_results_df[fit_results_df["Success"]].loc[
    fit_results_df.groupby(["Parameters", "Num. Replicas"])["R2"].idxmax()
]
print("\nBest model (by R²) for each condition:")
best_model_counts = best_models["Model"].value_counts()
print(best_model_counts)

# %%
# Best model identification (visualization removed - illegible 5x9 grid)
best_model_name = avg_r2_by_model.index[0]
print(f"\nBest overall model: {best_model_name}")

# %%
# Extract and analyze fitted parameters for the best model
print(f"\n{'='*60}")
print(f"PARAMETER ANALYSIS FOR {best_model_name.upper()} MODEL")
print("=" * 60)

best_model_fits = fit_results_df[
    (fit_results_df["Model"] == best_model_name) & (fit_results_df["Success"])
].copy()

# Extract individual parameters
param_names = {
    "exp_decay": ["nll_inf", "delta_nll", "tau"],
    "stretched_exp": ["nll_inf", "delta_nll", "tau", "beta"],
    "power_law": ["a", "b", "c"],
    "log": ["a", "b"],
    "linear": ["a", "b"],
    "constant": ["a"],
    "rational": ["a", "b", "c"],
}

if best_model_name in param_names:
    names = param_names[best_model_name]
    for idx, name in enumerate(names):
        best_model_fits[name] = best_model_fits["Fitted Params"].apply(
            lambda x: x[idx] if x is not None and len(x) > idx else np.nan
        )

    # Show how parameters vary with Num. Replicas
    print(f"\nFitted parameters by Num. Replicas (averaged across model sizes):")
    param_by_replica = best_model_fits.groupby("Num. Replicas")[names].agg(
        ["mean", "std"]
    )
    print(param_by_replica.round(4))

print("\nCurve fitting complete.")

# %%
# =============================================================================
# IMPROVED FITTING: Focus on the decay structure
# =============================================================================
# The data shows:
# 1. For R=0: NLL ≈ constant (baseline)
# 2. For R>0: NLL decays from ~NLL_0 toward NLL_floor(R)
#
# Let's fit a model: NLL(t) = NLL_inf + (NLL_0 - NLL_inf) * exp(-t/tau)
# where NLL_inf and tau depend on R

print("\n" + "=" * 60)
print("IMPROVED FITTING: Exponential decay with R-dependent parameters")
print("=" * 60)


def model_exp_decay_v2(t, nll_inf, nll_0, tau):
    """Exponential decay from nll_0 to nll_inf with timescale tau."""
    return nll_inf + (nll_0 - nll_inf) * np.exp(-t / tau)


# First, extract baseline NLL from R=0 for each model size
baseline_nll = {}
for param in unique_params:
    subset = nll_by_token_df[
        (nll_by_token_df["Parameters"] == param)
        & (nll_by_token_df["Num. MATH Test Set Replicas"] == 0)
    ]
    if len(subset) > 0:
        baseline_nll[param] = subset["mean_NLL"].mean()
        print(f"Baseline NLL for {param}: {baseline_nll[param]:.4f}")

# Fit exponential decay to R>0 data with NLL_0 fixed to baseline
improved_fit_results = []

for param in unique_params:
    nll_0_fixed = baseline_nll.get(param, 1.0)

    for replica in unique_replicas:
        # Skip R=0 (baseline) and R>=1000 (NLL uptick due to hard-to-memorize sequences)
        if replica == 0 or replica >= 1000:
            continue

        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        t = subset["Token Index"].values.astype(float)
        nll = subset["mean_NLL"].values
        num_params = subset["Num. Parameters"].iloc[0]

        # Fit with NLL_0 as a free parameter
        try:
            # Model: NLL(t) = nll_inf + (nll_0 - nll_inf) * exp(-t/tau)
            # Reparameterize: NLL(t) = nll_inf + delta * exp(-t/tau)

            def model_to_fit(t, nll_inf, delta, tau):
                return nll_inf + delta * np.exp(-t / tau)

            # Initial guesses based on data
            nll_start = nll[0]
            nll_end = nll[-1]
            p0 = [nll_end, nll_start - nll_end, 100]
            bounds = ([0, 0, 1], [10, 10, 2000])

            popt, pcov = curve_fit(
                model_to_fit, t, nll, p0=p0, bounds=bounds, maxfev=10000
            )
            nll_inf_fit, delta_fit, tau_fit = popt
            nll_0_fit = nll_inf_fit + delta_fit

            y_pred = model_to_fit(t, *popt)
            r2 = compute_r_squared(nll, y_pred)
            rmse = compute_rmse(nll, y_pred)

            improved_fit_results.append(
                {
                    "Parameters": param,
                    "Num. Parameters": num_params,
                    "Num. Replicas": replica,
                    "nll_inf": nll_inf_fit,
                    "nll_0": nll_0_fit,
                    "delta": delta_fit,
                    "tau": tau_fit,
                    "R2": r2,
                    "RMSE": rmse,
                    "Success": True,
                }
            )
        except Exception as e:
            improved_fit_results.append(
                {
                    "Parameters": param,
                    "Num. Parameters": num_params,
                    "Num. Replicas": replica,
                    "nll_inf": np.nan,
                    "nll_0": np.nan,
                    "delta": np.nan,
                    "tau": np.nan,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "Success": False,
                }
            )

improved_fit_df = pd.DataFrame(improved_fit_results)

print(f"\nSuccessful fits: {improved_fit_df['Success'].sum()} / {len(improved_fit_df)}")
print(f"\nMean R² for exponential decay: {improved_fit_df['R2'].mean():.4f}")
print(f"Std R²: {improved_fit_df['R2'].std():.4f}")

# Show parameter trends
print("\nFitted parameters by Num. Replicas (averaged across model sizes):")
param_summary = improved_fit_df.groupby("Num. Replicas")[
    ["nll_inf", "nll_0", "delta", "tau", "R2"]
].mean()
print(param_summary.round(4))

# %%
# Plot the fitted parameters as functions of R
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: nll_inf vs R
ax = axes[0, 0]
for param in unique_params:
    subset = improved_fit_df[improved_fit_df["Parameters"] == param]
    ax.plot(
        subset["Num. Replicas"],
        subset["nll_inf"],
        "o-",
        label=param,
        markersize=6,
        color=params_palette[param],
    )
ax.set_xlabel(r"Num. Replicas $R$")
ax.set_ylabel(r"$\mathrm{NLL}_\infty$ (floor)")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
# Add legend in lower left of this subplot (where there's whitespace)
ax.legend(title="Model Size", loc="lower left")

# Plot 2: tau vs R
ax = axes[0, 1]
for param in unique_params:
    subset = improved_fit_df[improved_fit_df["Parameters"] == param]
    ax.plot(
        subset["Num. Replicas"],
        subset["tau"],
        "o-",
        label=param,
        markersize=6,
        color=params_palette[param],
    )
ax.set_xlabel(r"Num. Replicas $R$")
ax.set_ylabel(r"$\tau$ (decay timescale)")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

# Plot 3: delta (initial drop) vs R
ax = axes[1, 0]
for param in unique_params:
    subset = improved_fit_df[improved_fit_df["Parameters"] == param]
    ax.plot(
        subset["Num. Replicas"],
        subset["delta"],
        "o-",
        label=param,
        markersize=6,
        color=params_palette[param],
    )
ax.set_xlabel(r"Num. Replicas $R$")
ax.set_ylabel(r"$\Delta$ (initial NLL drop)")
ax.set_xscale("symlog", linthresh=1)
ax.grid(True, alpha=0.3)

# Plot 4: R² vs R
ax = axes[1, 1]
for param in unique_params:
    subset = improved_fit_df[improved_fit_df["Parameters"] == param]
    ax.plot(
        subset["Num. Replicas"],
        subset["R2"],
        "o-",
        label=param,
        markersize=6,
        color=params_palette[param],
    )
ax.set_xlabel(r"Num. Replicas $R$")
ax.set_ylabel(r"$R^2$ (fit quality)")
ax.set_xscale("symlog", linthresh=1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=exp_decay_params_x=num_replicas",
)
# plt.show()

# %%
# =============================================================================
# Per-condition floor model fitting and parameter visualization
# =============================================================================


def floor_model(t, NLL_inf, A, alpha):
    """NLL(t) = NLL_∞ + A * t^(-α)"""
    return NLL_inf + A * (t ** (-alpha))


# Create mapping from parameter string to numeric N value
param_to_N = {p: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p] for p in unique_params}

# Fit floor model for each (N, R) condition
floor_fit_results = []

# Filter conditions: exclude R=0 (no recall) and R>=1000 (saturation)
valid_replicas = [r for r in unique_replicas if 0 < r < 1000]

for param in unique_params:
    N_val = param_to_N[param]
    for replica in valid_replicas:
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 10:
            continue

        # Use token indices 1-800
        subset = subset[(subset["Token Index"] >= 1) & (subset["Token Index"] <= 800)]
        t = subset["Token Index"].values
        nll = subset["mean_NLL"].values

        # Initial guesses
        nll_inf_init = np.mean(nll[-10:])  # asymptotic value
        A_init = nll[0] - nll_inf_init if nll[0] > nll_inf_init else 1.0
        alpha_init = 0.1

        try:
            popt, pcov = curve_fit(
                floor_model,
                t,
                nll,
                p0=[nll_inf_init, A_init, alpha_init],
                bounds=([0, 0, 0.001], [nll.max(), nll.max() * 10, 2.0]),
                maxfev=5000,
            )
            NLL_inf, A, alpha = popt

            # Compute R²
            y_pred = floor_model(t, *popt)
            ss_res = np.sum((nll - y_pred) ** 2)
            ss_tot = np.sum((nll - np.mean(nll)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            floor_fit_results.append(
                {
                    "Parameters": param,
                    "N": N_val,
                    "R": replica,
                    "NLL_inf": NLL_inf,
                    "A": A,
                    "alpha": alpha,
                    "R2": r2,
                }
            )
        except Exception as e:
            print(f"Failed to fit {param}, R={replica}: {e}")

floor_fit_df = pd.DataFrame(floor_fit_results)
print(f"Successfully fit {len(floor_fit_df)} conditions")
print(f"Mean R² = {floor_fit_df['R2'].mean():.3f}")
print(f"R² > 0.8: {(floor_fit_df['R2'] > 0.8).sum()}/{len(floor_fit_df)}")

# %%
# Visualization: NLL vs Token Index with fitted floor model curves
# Faceted by model size (columns), colored by num replicas (hue)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Color palette for replicas - use SymLogNorm for consistent colors with notebooks 10/11
all_replicas = [0, 1, 3, 10, 32, 100, 316, 1000, 3162]
replica_sym_norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=max(all_replicas))
viridis_cmap = matplotlib.colormaps["viridis"]
R_to_color = {r: viridis_cmap(replica_sym_norm(r)) for r in all_replicas}

for idx, param in enumerate(unique_params):
    ax = axes[idx]
    N_val = param_to_N[param]

    for replica in valid_replicas:
        # Get data
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        t = subset["Token Index"].values
        nll = subset["mean_NLL"].values

        # Plot data points
        ax.scatter(
            t, nll, s=10, alpha=0.5, color=R_to_color[replica], label=f"{replica}"
        )

        # Get fitted parameters and plot curve
        fit_row = floor_fit_df[
            (floor_fit_df["Parameters"] == param) & (floor_fit_df["R"] == replica)
        ]
        if len(fit_row) == 1:
            NLL_inf = fit_row["NLL_inf"].values[0]
            A = fit_row["A"].values[0]
            alpha = fit_row["alpha"].values[0]

            t_smooth = np.linspace(max(1, t.min()), t.max(), 200)
            nll_fit = floor_model(t_smooth, NLL_inf, A, alpha)
            ax.plot(t_smooth, nll_fit, "-", color=R_to_color[replica], linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Negative Log Likelihood")
    ax.set_title(param)
    ax.grid(True, alpha=0.3)

# Use empty subplot for legend
axes[5].axis("off")
handles = [
    plt.Line2D([0], [0], color=R_to_color[r], marker="o", linestyle="-", markersize=5)
    for r in valid_replicas
]
axes[5].legend(
    handles,
    [str(r) for r in valid_replicas],
    title="Num. Replicas",
    loc="center",
)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=nll_x=token_index_hue=num_replicas_col=model_size_fit",
)
plt.close()

# %%
# =============================================================================
# Combined 2-row, 3-column figure for paper (34M, 93M, 344M)
# Row 1: NLL with fitted floor model curves
# Row 2: Cumulative probability
# Complete legend with all replicas from 0 to 1000
# =============================================================================

selected_params_combined = ["34M", "93M", "344M"]

# Filter cumulative probability data for selected model sizes
plot5_combined_df = cumulative_nll_by_token_df[
    (cumulative_nll_by_token_df["Parameters"].isin(selected_params_combined))
    & (cumulative_nll_by_token_df["Token Index + 1"] <= 800)
].copy()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: NLL with fitted floor model curves
for idx, param in enumerate(selected_params_combined):
    ax = axes[0, idx]
    N_val = param_to_N[param]

    for replica in valid_replicas:
        # Get data
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        t = subset["Token Index"].values
        nll = subset["mean_NLL"].values

        # Plot data points
        ax.scatter(
            t, nll, s=10, alpha=0.5, color=R_to_color[replica], label=f"{replica}"
        )

        # Get fitted parameters and plot curve
        fit_row = floor_fit_df[
            (floor_fit_df["Parameters"] == param) & (floor_fit_df["R"] == replica)
        ]
        if len(fit_row) == 1:
            NLL_inf = fit_row["NLL_inf"].values[0]
            A = fit_row["A"].values[0]
            alpha = fit_row["alpha"].values[0]

            t_smooth = np.linspace(max(1, t.min()), t.max(), 200)
            nll_fit = floor_model(t_smooth, NLL_inf, A, alpha)
            ax.plot(t_smooth, nll_fit, "-", color=R_to_color[replica], linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Negative Log Likelihood")
    ax.set_title(param)
    ax.grid(True, alpha=0.3, which="both")

# Row 2: Cumulative Probability
for idx, param in enumerate(selected_params_combined):
    ax = axes[1, idx]

    for replica in unique_replicas:
        subset = plot5_combined_df[
            (plot5_combined_df["Parameters"] == param)
            & (plot5_combined_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index + 1")

        if len(subset) == 0:
            continue

        color = replica_palette[str(replica)]
        ax.plot(
            subset["Token Index + 1"],
            subset["cumulative_prob"],
            color=color,
            label=f"{replica}",
        )
        # Add uncertainty bands
        ax.fill_between(
            subset["Token Index + 1"],
            subset["ci_lower_prob"],
            subset["ci_upper_prob"],
            alpha=0.2,
            color=color,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(param)
    ax.set_xlim(1, 800)
    ax.set_ylim(1e-4, 1)
    ax.grid(True, alpha=0.3, which="both")

# Create legend handles for ALL replicas that appear in the plot
legend_replicas = unique_replicas  # Include all replicas (0 to 3162)
handles = [
    plt.Line2D([0], [0], color=replica_palette[str(r)], marker="o", linestyle="-", markersize=5)
    for r in legend_replicas
]
fig.legend(
    handles,
    [str(r) for r in legend_replicas],
    title="Num. Replicas",
    loc="upper left",
    bbox_to_anchor=(1, 1),
)

plt.tight_layout()
plt.subplots_adjust(right=0.88)  # Make room for legend
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=nll_and_cumprob_x=token_index_hue=num_replicas_col=model_size_combined",
)
plt.close()

# %%
# Visualization 1: Parameters vs R (x) with model size (hue)
# Three columns for NLL_∞, A, α

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Color palette for model sizes (log scale) - use "flare" for model size
N_values = sorted(floor_fit_df["N"].unique())
colors_N = sns.color_palette("flare", len(N_values))
N_to_color = {N: colors_N[i] for i, N in enumerate(N_values)}

for i, (param_name, ylabel, use_log_y) in enumerate(
    [
        ("NLL_inf", r"$\mathrm{NLL}_\infty$", True),
        ("A", r"$A$", True),
        ("alpha", r"$\alpha$", False),
    ]
):
    ax = axes[i]

    for N_val in N_values:
        subset = floor_fit_df[floor_fit_df["N"] == N_val].sort_values("R")
        label = f"{N_val/1e6:.0f}M" if N_val >= 1e6 else f"{N_val/1e3:.0f}K"
        ax.plot(
            subset["R"],
            subset[param_name],
            "o-",
            color=N_to_color[N_val],
            label=label,
            markersize=8,
        )

    ax.set_xlabel(r"Num. Replicas $R$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("symlog", linthresh=1)
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

axes[2].legend(title=r"Model Size", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.subplots_adjust(right=0.88)  # Make room for legend

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit_params_x=num_replicas_hue=model_size",
)
# plt.show()

# %%
# Visualization 2: Parameters vs N (x, log scale) with replicas (hue)
# Three columns for NLL_∞, A, α

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Color palette for replicas - use SymLogNorm for consistent colors with notebooks 10/11
all_replicas = [0, 1, 3, 10, 32, 100, 316, 1000, 3162]
replica_sym_norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=max(all_replicas))
viridis_cmap = matplotlib.colormaps["viridis"]
R_to_color = {r: viridis_cmap(replica_sym_norm(r)) for r in all_replicas}

# Only plot the valid replicas that we actually fit
R_values = sorted(floor_fit_df["R"].unique())

for i, (param_name, ylabel, use_log_y) in enumerate(
    [
        ("NLL_inf", r"$\mathrm{NLL}_\infty$", True),
        ("A", r"$A$", True),
        ("alpha", r"$\alpha$", False),
    ]
):
    ax = axes[i]

    for R_val in R_values:
        subset = floor_fit_df[floor_fit_df["R"] == R_val].sort_values("N")
        ax.plot(
            subset["N"],
            subset[param_name],
            "o-",
            color=R_to_color[R_val],
            label=str(R_val),
            markersize=8,
        )

    ax.set_xlabel(r"Model Size $N$")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

axes[2].legend(title=r"Replicas $R$", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.subplots_adjust(right=0.88)  # Make room for legend

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit_params_x=model_size_hue=num_replicas",
)
# plt.show()
