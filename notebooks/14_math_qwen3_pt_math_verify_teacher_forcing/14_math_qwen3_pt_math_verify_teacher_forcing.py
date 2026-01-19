"""Analyze teacher-forced evaluation results: NLL by token position.

This notebook analyzes how negative log likelihood varies across token positions
in the solution, broken down by model size and contamination level (num replicas).
"""

import ast
import gc
import hashlib
import os
import re

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

# Create color palette mapping for the replica values (viridis for replicas)
n_replicas = len(unique_replicas)
viridis_colors = sns.color_palette("viridis", n_replicas)
replica_palette = {str(r): viridis_colors[i] for i, r in enumerate(unique_replicas)}

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
flare_cmap = plt.cm.get_cmap("flare")
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
    ylabel=r"Cumulative Sequence Probability",
    xscale="log",
    yscale="log",
    xlim=(1, 800),
    ylim=(1e-30, 1),  # Truncate y-axis to avoid very small probabilities
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
# Now try to find functional forms for how nll_inf and tau depend on R and N
print("\n" + "=" * 60)
print("FITTING PARAMETER DEPENDENCE ON R AND N")
print("=" * 60)

# For nll_inf(R, N): expect it to decrease with R (more contamination = lower floor)
# Candidate: nll_inf = nll_mem + (nll_base(N) - nll_mem) * exp(-R / R_0)
# Or power law: nll_inf = nll_mem + a(N) * R^(-b)

# For tau(R): expect it to decrease with R (faster recall with more contamination)
# Candidate: tau = tau_0 / (1 + R/R_1)^c

# Let's fit these across all data points
successful_fits = improved_fit_df[improved_fit_df["Success"]].copy()

# Fit nll_inf vs R for each model size
print("\nFitting nll_inf = a * R^(-b) + c for each model size:")
nll_inf_params = {}
for param in unique_params:
    subset = successful_fits[successful_fits["Parameters"] == param]
    if len(subset) < 3:
        continue

    R = subset["Num. Replicas"].values.astype(float)
    nll_inf = subset["nll_inf"].values

    try:
        # Model: nll_inf = c + a * R^(-b)  (power law decay to floor)
        def nll_inf_model(R, a, b, c):
            return c + a * (R + 1) ** (-b)

        popt, _ = curve_fit(
            nll_inf_model, R, nll_inf, p0=[1, 0.5, 0.01], bounds=([0, 0, 0], [10, 3, 1])
        )
        y_pred = nll_inf_model(R, *popt)
        r2 = compute_r_squared(nll_inf, y_pred)
        nll_inf_params[param] = {"a": popt[0], "b": popt[1], "c": popt[2], "r2": r2}
        print(
            f"  {param}: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.6f}, R²={r2:.4f}"
        )
    except Exception as e:
        print(f"  {param}: Fit failed - {e}")

# Fit tau vs R for each model size
print("\nFitting tau = tau_0 / (1 + R/R_1)^c for each model size:")
tau_params = {}
for param in unique_params:
    subset = successful_fits[successful_fits["Parameters"] == param]
    if len(subset) < 3:
        continue

    R = subset["Num. Replicas"].values.astype(float)
    tau = subset["tau"].values

    try:
        # Model: tau = tau_0 / (1 + R/R_1)^c
        def tau_model(R, tau_0, R_1, c):
            return tau_0 / (1 + R / R_1) ** c

        popt, _ = curve_fit(
            tau_model, R, tau, p0=[500, 10, 0.5], bounds=([1, 0.1, 0], [2000, 1000, 3])
        )
        y_pred = tau_model(R, *popt)
        r2 = compute_r_squared(tau, y_pred)
        tau_params[param] = {"tau_0": popt[0], "R_1": popt[1], "c": popt[2], "r2": r2}
        print(
            f"  {param}: tau_0={popt[0]:.1f}, R_1={popt[1]:.1f}, c={popt[2]:.4f}, R²={r2:.4f}"
        )
    except Exception as e:
        print(f"  {param}: Fit failed - {e}")

print("\nParameter fitting complete.")

# %%
# =============================================================================
# UNIFIED MODEL: Fit NLL(t, R, N) across ALL conditions simultaneously
# =============================================================================
print("\n" + "=" * 70)
print("UNIFIED MODEL FITTING: NLL(t, R, N) across all conditions")
print("=" * 70)

from scipy.optimize import minimize, differential_evolution

# Prepare pooled data
# Filter: exclude R=0 (baseline) and R>=1000 (NLL uptick due to hard-to-memorize sequences)
valid_replicas_unified = [r for r in unique_replicas if 0 < r < 1000]
pooled_data = []
for param in unique_params:
    for replica in valid_replicas_unified:
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        N = subset["Num. Parameters"].iloc[0]
        R = replica

        for _, row in subset.iterrows():
            pooled_data.append(
                {
                    "t": row["Token Index"],
                    "N": N,
                    "R": R,
                    "NLL": row["mean_NLL"],
                    "weight": 1.0 / row["sem_NLL"] if row["sem_NLL"] > 0 else 1.0,
                }
            )

pooled_df = pd.DataFrame(pooled_data)
print(
    f"Pooled data: {len(pooled_df)} points across {len(unique_params)} model sizes and {len(valid_replicas_unified)} replica levels (filtered)"
)
print(f"  Note: Excluded R=0 (baseline) and R>=1000 (NLL uptick due to hard-to-memorize sequences)")

# Normalize N for numerical stability
N_scale = 1e8  # Scale factor for N
pooled_df["N_scaled"] = pooled_df["N"] / N_scale

t_all = pooled_df["t"].values
N_all = pooled_df["N_scaled"].values
R_all = pooled_df["R"].values
NLL_all = pooled_df["NLL"].values
weights_all = pooled_df["weight"].values
weights_all = weights_all / weights_all.mean()  # Normalize weights


# %%
# Define unified model candidates


def unified_model_v1(t, N, R, params):
    """
    Model V1: Exponential decay with power-law parameter dependence

    NLL(t, R, N) = NLL_∞(R, N) + Δ(R, N) * exp(-t / τ(R, N))

    where:
    - NLL_∞(R, N) = c0 * N^(-α) * (1 + c1 * (R+1)^(-β))
    - τ(R, N) = τ0 * N^γ / (1 + R/R0)^δ
    - Δ(R, N) = d0 * N^(-ε) * (R+1)^ζ / (1 + (R/R1)^η)

    Parameters: [c0, α, c1, β, τ0, γ, R0, δ, d0, ε, ζ, R1, η]
    """
    c0, alpha, c1, beta, tau0, gamma, R0, delta, d0, eps, zeta, R1, eta = params

    # Ensure positive values
    c0, alpha, c1, beta = abs(c0), abs(alpha), abs(c1), abs(beta)
    tau0, gamma, R0, delta = abs(tau0), abs(gamma), abs(R0) + 0.1, abs(delta)
    d0, eps, zeta, R1, eta = abs(d0), abs(eps), abs(zeta), abs(R1) + 0.1, abs(eta)

    # NLL floor: decreases with N (larger models better), decreases with R (more contamination)
    NLL_inf = c0 * N ** (-alpha) * (1 + c1 * (R + 1) ** (-beta))

    # Decay timescale: increases with N (larger models slower?), decreases with R
    tau = tau0 * N**gamma / (1 + R / R0) ** delta
    tau = np.maximum(tau, 1.0)  # Prevent tau from being too small

    # Amplitude: depends on both N and R
    Delta = d0 * N ** (-eps) * (R + 1) ** zeta / (1 + (R / R1) ** eta)

    return NLL_inf + Delta * np.exp(-t / tau)


def unified_model_v2(t, N, R, params):
    """
    Model V2: Simpler separable model

    NLL(t, R, N) = NLL_base(N) * f(R) * g(t, R)

    where:
    - NLL_base(N) = a * N^(-b)
    - f(R) = (1 + c*(R+1)^(-d)) for floor adjustment
    - g(t, R) = e^(-e) + (1 - e^(-e)) * exp(-t / (τ0 / (1 + R/R0)^f))

    This says: NLL starts at NLL_base(N) and decays toward NLL_base(N)*floor_factor(R)

    Parameters: [a, b, c, d, τ0, R0, f, floor_min]
    """
    a, b, c, d, tau0, R0, f, floor_min = params

    a, b, c, d = abs(a), abs(b), abs(c), abs(d)
    tau0, R0, f, floor_min = abs(tau0), abs(R0) + 0.1, abs(f), abs(floor_min)

    # Base NLL (depends only on N)
    NLL_base = a * N ** (-b)

    # Floor factor (depends on R): goes from ~1 at R=0 to floor_min at R=∞
    floor_factor = floor_min + (1 - floor_min) * (R + 1) ** (-d)

    # Decay timescale (depends on R)
    tau = tau0 / (1 + R / R0) ** f
    tau = np.maximum(tau, 1.0)

    # NLL decays from NLL_base toward NLL_base * floor_factor
    NLL_inf = NLL_base * floor_factor
    Delta = NLL_base * (1 - floor_factor)

    return NLL_inf + Delta * np.exp(-t / tau)


def unified_model_v3(t, N, R, params):
    """
    Model V3: Log-linear relationships

    log(NLL) = a0 + a1*log(N) + a2*log(R+1) + a3*log(t+1)
               + a4*log(N)*log(R+1) + a5*log(R+1)*log(t+1) + a6*log(N)*log(t+1)

    Parameters: [a0, a1, a2, a3, a4, a5, a6]
    """
    a0, a1, a2, a3, a4, a5, a6 = params

    log_N = np.log(N)
    log_R = np.log(R + 1)
    log_t = np.log(t + 1)

    log_NLL = (
        a0
        + a1 * log_N
        + a2 * log_R
        + a3 * log_t
        + a4 * log_N * log_R
        + a5 * log_R * log_t
        + a6 * log_N * log_t
    )

    return np.exp(log_NLL)


def unified_model_v4(t, N, R, params):
    """
    Model V4: Exponential decay with explicit floor and rate dependence

    NLL(t, R, N) = NLL_∞(R, N) + [NLL_0(N) - NLL_∞(R, N)] * exp(-t / τ(R))

    where:
    - NLL_0(N) = a * N^(-b)  (initial NLL, depends only on model size)
    - NLL_∞(R, N) = NLL_0(N) * [c + (1-c) * exp(-R/R_floor)]  (floor depends on R)
    - τ(R) = τ_max * exp(-R/R_tau)  (decay rate depends on R)

    Parameters: [a, b, c, R_floor, τ_max, R_tau]
    """
    a, b, c, R_floor, tau_max, R_tau = params

    a, b = abs(a), abs(b)
    c = np.clip(abs(c), 0.001, 0.999)  # Floor fraction between 0 and 1
    R_floor, tau_max, R_tau = abs(R_floor) + 0.1, abs(tau_max), abs(R_tau) + 0.1

    # Initial NLL (depends only on N)
    NLL_0 = a * N ** (-b)

    # Floor NLL (fraction of initial, depends on R)
    floor_frac = c + (1 - c) * np.exp(-R / R_floor)
    NLL_inf = NLL_0 * floor_frac

    # Decay timescale (depends on R)
    tau = tau_max * np.exp(-R / R_tau)
    tau = np.maximum(tau, 1.0)

    return NLL_inf + (NLL_0 - NLL_inf) * np.exp(-t / tau)


def unified_model_v5(t, N, R, params):
    """
    Model V5: Most flexible - separate power laws for everything

    NLL(t, R, N) = NLL_∞ + Δ * exp(-t/τ)

    - NLL_∞ = a * N^(-b) * (R+1)^(-c)
    - Δ = d * N^(-e) * (1 - (R+1)^(-f))
    - τ = g * N^h / (R+1)^i

    Parameters: [a, b, c, d, e, f, g, h, i]
    """
    a, b, c, d, e, f, g, h, i = params

    a, b, c = abs(a), abs(b), abs(c)
    d, e, f = abs(d), abs(e), abs(f)
    g, h, i = abs(g), abs(h), abs(i)

    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    Delta = d * N ** (-e) * (1 - (R + 1) ** (-f))
    tau = g * N**h / (R + 1) ** i
    tau = np.maximum(tau, 1.0)

    return NLL_inf + Delta * np.exp(-t / tau)


# %%
# Fitting function
def compute_loss(params, model_func, t, N, R, NLL_true, weights):
    """Compute weighted MSE loss."""
    try:
        NLL_pred = model_func(t, N, R, params)
        if np.any(np.isnan(NLL_pred)) or np.any(np.isinf(NLL_pred)):
            return 1e10
        residuals = (NLL_true - NLL_pred) * np.sqrt(weights)
        return np.mean(residuals**2)
    except:
        return 1e10


def fit_unified_model(model_func, model_name, n_params, bounds, n_restarts=5):
    """Fit a unified model using differential evolution + local optimization."""
    print(f"\nFitting {model_name} ({n_params} parameters)...")

    best_result = None
    best_loss = np.inf

    for restart in range(n_restarts):
        try:
            # Global optimization with differential evolution
            result = differential_evolution(
                compute_loss,
                bounds,
                args=(model_func, t_all, N_all, R_all, NLL_all, weights_all),
                maxiter=1000,
                tol=1e-6,
                seed=42 + restart,
                workers=1,
                updating="deferred",
                polish=True,
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result

        except Exception as e:
            print(f"  Restart {restart+1} failed: {e}")
            continue

    if best_result is None:
        print(f"  FAILED: No successful optimization")
        return None

    # Compute R²
    NLL_pred = model_func(t_all, N_all, R_all, best_result.x)
    r2 = compute_r_squared(NLL_all, NLL_pred)
    rmse = compute_rmse(NLL_all, NLL_pred)

    print(f"  Loss: {best_loss:.6f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
    print(f"  Parameters: {best_result.x}")

    return {
        "model_name": model_name,
        "params": best_result.x,
        "loss": best_loss,
        "r2": r2,
        "rmse": rmse,
        "model_func": model_func,
    }


# %%
# Fit all model variants
print("\n" + "-" * 70)
print("Fitting multiple unified model variants...")
print("-" * 70)

unified_results = []

# Model V2: Simpler separable (8 params)
bounds_v2 = [
    (0.1, 50),
    (0, 2),
    (0, 10),
    (0, 3),
    (1, 2000),
    (0.1, 500),
    (0, 3),
    (0.001, 0.5),
]
result = fit_unified_model(unified_model_v2, "V2_separable", 8, bounds_v2, n_restarts=3)
if result:
    unified_results.append(result)

# Model V4: Explicit floor/rate (6 params)
bounds_v4 = [(0.1, 50), (0, 2), (0.001, 0.5), (0.1, 500), (1, 2000), (0.1, 500)]
result = fit_unified_model(
    unified_model_v4, "V4_floor_rate", 6, bounds_v4, n_restarts=3
)
if result:
    unified_results.append(result)

# Model V5: Separate power laws (9 params)
bounds_v5 = [
    (0.1, 50),
    (0, 2),
    (0, 2),
    (0.1, 50),
    (0, 2),
    (0, 2),
    (1, 2000),
    (0, 2),
    (0, 2),
]
result = fit_unified_model(
    unified_model_v5, "V5_power_laws", 9, bounds_v5, n_restarts=3
)
if result:
    unified_results.append(result)

# Model V3: Log-linear (7 params)
bounds_v3 = [(-10, 10), (-3, 3), (-3, 3), (-3, 3), (-1, 1), (-1, 1), (-1, 1)]
result = fit_unified_model(
    unified_model_v3, "V3_log_linear", 7, bounds_v3, n_restarts=3
)
if result:
    unified_results.append(result)

# Model V1: Full model (13 params) - try last since most complex
bounds_v1 = [
    (0.1, 50),
    (0, 2),
    (0, 10),
    (0, 3),
    (1, 2000),
    (0, 2),
    (0.1, 500),
    (0, 3),
    (0.1, 50),
    (0, 2),
    (0, 2),
    (0.1, 500),
    (0, 3),
]
result = fit_unified_model(unified_model_v1, "V1_full", 13, bounds_v1, n_restarts=2)
if result:
    unified_results.append(result)

# %%
# Compare results
print("\n" + "=" * 70)
print("UNIFIED MODEL COMPARISON")
print("=" * 70)

if unified_results:
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "N_params": len(r["params"]),
                "R²": r["r2"],
                "RMSE": r["rmse"],
            }
            for r in unified_results
        ]
    ).sort_values("R²", ascending=False)
    print("\nModel comparison (sorted by R²):")
    print(comparison_df.to_string(index=False))

    # Select best model
    best_unified = max(unified_results, key=lambda x: x["r2"])
    print(
        f"\nBest unified model: {best_unified['model_name']} (R² = {best_unified['r2']:.4f})"
    )
    print(f"Parameters: {best_unified['params']}")

print("\nUnified model fitting complete.")

# %%
# =============================================================================
# COMPREHENSIVE MODEL SEARCH: Ruthless hill-climbing
# =============================================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL SEARCH: Exhaustive fitting")
print("=" * 70)

# Define many more model candidates based on observations and physical intuition


def model_exp_power(t, N, R, params):
    """
    Exponential decay with power-law floor and rate.
    NLL = a*N^(-b)*(R+1)^(-c) + d*N^(-e)*exp(-t/(f*(R+1)^(-g)))
    """
    a, b, c, d, e, f, g = [abs(p) for p in params]
    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    tau = f * (R + 1) ** (-g)
    tau = np.maximum(tau, 1.0)
    Delta = d * N ** (-e)
    return NLL_inf + Delta * np.exp(-t / tau)


def model_double_exp(t, N, R, params):
    """
    Double exponential - fast initial decay + slow long-term decay.
    NLL = a*N^(-b)*(R+1)^(-c) + d*exp(-t/tau1) + e*exp(-t/tau2)
    """
    a, b, c, d, e, tau1, tau2 = [abs(p) for p in params]
    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    tau1 = np.maximum(tau1 / (R + 1) ** 0.5, 1.0)
    tau2 = np.maximum(tau2, 10.0)
    return NLL_inf + d * np.exp(-t / tau1) + e * np.exp(-t / tau2)


def model_logistic_decay(t, N, R, params):
    """
    Logistic decay from NLL_0 to NLL_inf.
    NLL = NLL_inf + (NLL_0 - NLL_inf) / (1 + exp((t - t_half)/k))
    """
    a, b, c, d, t_half_base, k_base = [abs(p) for p in params]
    NLL_0 = a * N ** (-b)
    NLL_inf = NLL_0 * c * (R + 1) ** (-d)
    t_half = t_half_base / (R + 1) ** 0.3
    k = k_base / (R + 1) ** 0.2
    k = np.maximum(k, 1.0)
    return NLL_inf + (NLL_0 - NLL_inf) / (1 + np.exp((t - t_half) / k))


def model_power_decay(t, N, R, params):
    """
    Power-law decay in t.
    NLL = a*N^(-b)*(R+1)^(-c) + d*N^(-e)*(t+1)^(-f*(R+1)^g)
    """
    a, b, c, d, e, f, g = [abs(p) for p in params]
    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    power = f * (R + 1) ** g
    power = np.clip(power, 0.01, 2.0)
    Delta = d * N ** (-e)
    return NLL_inf + Delta * (t + 1) ** (-power)


def model_stretched_exp_unified(t, N, R, params):
    """
    Stretched exponential decay.
    NLL = NLL_inf + Delta * exp(-(t/tau)^beta)
    """
    a, b, c, d, e, tau0, beta0, tau_R, beta_R = [abs(p) for p in params]
    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    Delta = d * N ** (-e)
    tau = tau0 / (1 + R / tau_R)
    tau = np.maximum(tau, 1.0)
    beta = beta0 + beta_R * np.log(R + 1)
    beta = np.clip(beta, 0.1, 2.0)
    return NLL_inf + Delta * np.exp(-((t / tau) ** beta))


def model_rational_decay(t, N, R, params):
    """
    Rational function decay.
    NLL = NLL_inf + Delta / (1 + (t/tau)^p)
    """
    a, b, c, d, e, tau0, p, tau_R = [abs(p) for p in params]
    NLL_inf = a * N ** (-b) * (R + 1) ** (-c)
    Delta = d * N ** (-e)
    tau = tau0 / (1 + R / tau_R)
    tau = np.maximum(tau, 1.0)
    p = np.clip(p, 0.5, 3.0)
    return NLL_inf + Delta / (1 + (t / tau) ** p)


def model_log_linear_extended(t, N, R, params):
    """
    Extended log-linear with more interaction terms.
    """
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = params
    log_N = np.log(N)
    log_R = np.log(R + 1)
    log_t = np.log(t + 1)
    log_NLL = (
        a0
        + a1 * log_N
        + a2 * log_R
        + a3 * log_t
        + a4 * log_N * log_R
        + a5 * log_R * log_t
        + a6 * log_N * log_t
        + a7 * log_N**2
        + a8 * log_R**2
    )
    return np.exp(log_NLL)


def model_exp_with_baseline(t, N, R, params):
    """
    Exponential decay that respects R=0 baseline.
    For R=0: NLL ≈ constant (baseline)
    For R>0: NLL decays from baseline toward floor
    """
    a, b, baseline_mult, floor_a, floor_b, tau0, tau_R = [abs(p) for p in params]
    # Baseline NLL (for R=0)
    NLL_baseline = a * N ** (-b)
    # Floor NLL (for R→∞)
    NLL_floor = floor_a * N ** (-floor_b)
    # Interpolate based on R
    R_effect = 1 - np.exp(-R / 10)  # Smooth transition
    NLL_inf = NLL_baseline * (1 - R_effect) + NLL_floor * R_effect
    # Delta is the gap that decays
    Delta = (NLL_baseline - NLL_floor) * R_effect
    # Tau depends on R
    tau = tau0 / (1 + R / tau_R)
    tau = np.maximum(tau, 1.0)
    return NLL_inf + Delta * np.exp(-t / tau)


def model_mixture(t, N, R, params):
    """
    Mixture model: probability of being in 'recall' state increases with t and R.
    NLL = p_recall * NLL_recall + (1-p_recall) * NLL_base
    where p_recall = 1 - exp(-lambda * t) and lambda depends on R
    """
    a, b, c, d, lambda0, lambda_R = [abs(p) for p in params]
    NLL_base = a * N ** (-b)  # Baseline NLL
    NLL_recall = c * N ** (-d)  # NLL when in recall mode (lower)
    # Rate of entering recall state
    lambda_rate = lambda0 * (R + 1) ** lambda_R
    # Probability of being in recall state by time t
    p_recall = 1 - np.exp(-lambda_rate * t / 100)
    return p_recall * NLL_recall + (1 - p_recall) * NLL_base


def model_neural_scaling_inspired(t, N, R, params):
    """
    Inspired by neural scaling laws.
    NLL = (A/N^alpha + B) * (C/(R+1)^beta + D) * (E/(t+1)^gamma + F)
    """
    A, alpha, B, C, beta, D, E, gamma, F = [abs(p) for p in params]
    N_term = A / (N**alpha) + B
    R_term = C / ((R + 1) ** beta) + D
    t_term = E / ((t + 1) ** gamma) + F
    return N_term * R_term * t_term


# Extended model dictionary
EXTENDED_MODELS = {
    "exp_power": {
        "func": model_exp_power,
        "bounds": [
            (0.1, 20),
            (0, 2),
            (0, 2),
            (0.1, 20),
            (0, 2),
            (1, 500),
            (0, 2),
        ],
        "n_params": 7,
    },
    "double_exp": {
        "func": model_double_exp,
        "bounds": [
            (0.01, 10),
            (0, 2),
            (0, 2),
            (0.1, 10),
            (0.1, 10),
            (1, 500),
            (10, 1000),
        ],
        "n_params": 7,
    },
    "logistic_decay": {
        "func": model_logistic_decay,
        "bounds": [(0.1, 20), (0, 2), (0.01, 1), (0, 2), (1, 500), (1, 200)],
        "n_params": 6,
    },
    "power_decay": {
        "func": model_power_decay,
        "bounds": [
            (0.01, 10),
            (0, 2),
            (0, 2),
            (0.1, 20),
            (0, 2),
            (0.01, 1),
            (0, 1),
        ],
        "n_params": 7,
    },
    "stretched_exp": {
        "func": model_stretched_exp_unified,
        "bounds": [
            (0.01, 10),
            (0, 2),
            (0, 2),
            (0.1, 20),
            (0, 2),
            (1, 500),
            (0.5, 2),
            (1, 100),
            (0, 0.5),
        ],
        "n_params": 9,
    },
    "rational_decay": {
        "func": model_rational_decay,
        "bounds": [
            (0.01, 10),
            (0, 2),
            (0, 2),
            (0.1, 20),
            (0, 2),
            (1, 500),
            (0.5, 3),
            (1, 100),
        ],
        "n_params": 8,
    },
    "log_linear_ext": {
        "func": model_log_linear_extended,
        "bounds": [
            (-10, 10),
            (-3, 3),
            (-3, 3),
            (-3, 3),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
            (-1, 1),
        ],
        "n_params": 9,
    },
    "exp_baseline": {
        "func": model_exp_with_baseline,
        "bounds": [
            (0.1, 20),
            (0, 2),
            (0.5, 2),
            (0.01, 5),
            (0, 2),
            (1, 500),
            (1, 100),
        ],
        "n_params": 7,
    },
    "mixture": {
        "func": model_mixture,
        "bounds": [(0.1, 20), (0, 2), (0.01, 5), (0, 2), (0.001, 1), (0, 2)],
        "n_params": 6,
    },
    "neural_scaling": {
        "func": model_neural_scaling_inspired,
        "bounds": [
            (0.01, 10),
            (0, 2),
            (0.01, 5),
            (0.01, 10),
            (0, 2),
            (0.01, 2),
            (0.01, 5),
            (0, 1),
            (0.1, 2),
        ],
        "n_params": 9,
    },
}

# %%
# Fit all extended models
print("\n" + "-" * 70)
print("Fitting extended model candidates...")
print("-" * 70)

all_results = unified_results.copy()  # Start with previous results

for model_name, model_info in EXTENDED_MODELS.items():
    result = fit_unified_model(
        model_info["func"],
        model_name,
        model_info["n_params"],
        model_info["bounds"],
        n_restarts=3,
    )
    if result:
        all_results.append(result)

# %%
# Comprehensive comparison
print("\n" + "=" * 70)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame(
    [
        {
            "Model": r["model_name"],
            "N_params": len(r["params"]),
            "R²": r["r2"],
            "RMSE": r["rmse"],
            "Loss": r["loss"],
        }
        for r in all_results
    ]
).sort_values("R²", ascending=False)

print("\nAll models ranked by R²:")
print(comparison_df.to_string(index=False))

# Compute AIC/BIC for model selection
n_data = len(NLL_all)
comparison_df["AIC"] = comparison_df.apply(
    lambda row: n_data * np.log(row["RMSE"] ** 2) + 2 * row["N_params"], axis=1
)
comparison_df["BIC"] = comparison_df.apply(
    lambda row: n_data * np.log(row["RMSE"] ** 2) + row["N_params"] * np.log(n_data),
    axis=1,
)

print("\nModels ranked by BIC (penalizes complexity):")
print(comparison_df.sort_values("BIC").to_string(index=False))

# %%
# Select best model (by R² and parsimony)
best_by_r2 = max(all_results, key=lambda x: x["r2"])
best = best_by_r2  # Use best R² model for subsequent analysis
print(f"\nBest model by R²: {best_by_r2['model_name']} (R² = {best_by_r2['r2']:.4f})")

# Find best model with reasonable complexity (< 8 params)
parsimonious = [r for r in all_results if len(r["params"]) <= 7]
if parsimonious:
    best_parsimonious = max(parsimonious, key=lambda x: x["r2"])
    print(
        f"Best parsimonious model (≤7 params): {best_parsimonious['model_name']} (R² = {best_parsimonious['r2']:.4f})"
    )

# %%
# =============================================================================
# DETAILED ANALYSIS OF TOP MODELS
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED ANALYSIS OF TOP 3 MODELS")
print("=" * 70)

top_3 = sorted(all_results, key=lambda x: x["r2"], reverse=True)[:3]

for rank, model in enumerate(top_3, 1):
    print(f"\n{'='*50}")
    print(f"RANK {rank}: {model['model_name']}")
    print(f"{'='*50}")
    print(f"R² = {model['r2']:.4f}")
    print(f"RMSE = {model['rmse']:.4f}")
    print(f"N_params = {len(model['params'])}")
    print(f"Parameters: {model['params']}")

# %%
# Compute local R² matrix for report
# Note: Uses valid_replicas_unified (excludes R=0 and R>=1000) to be consistent with fitting
local_r2_matrix = np.full((len(unique_params), len(valid_replicas_unified)), np.nan)
for i, param in enumerate(unique_params):
    for j, replica in enumerate(valid_replicas_unified):
        subset = nll_by_token_df[
            (nll_by_token_df["Parameters"] == param)
            & (nll_by_token_df["Num. MATH Test Set Replicas"] == replica)
        ].sort_values("Token Index")

        if len(subset) < 5:
            continue

        t = subset["Token Index"].values.astype(float)
        nll = subset["mean_NLL"].values
        N = subset["Num. Parameters"].iloc[0] / N_scale
        R = replica

        nll_pred = best["model_func"](t, N, R, best["params"])
        local_r2_matrix[i, j] = compute_r_squared(nll, nll_pred)

# %%
# =============================================================================
# SAVE MODEL FITTING REPORT (Markdown)
# =============================================================================
report_path = os.path.join(results_dir, "MODEL_FITTING_REPORT.md")

with open(report_path, "w") as f:
    f.write("# NLL vs Token Index Model Fitting Report\n\n")

    f.write("## Executive Summary\n\n")
    f.write("We tested 15 different functional forms for `NLL(t, R, N)` to model how per-token\n")
    f.write("negative log-likelihood varies with token position, contamination level, and model size.\n\n")
    f.write("**Key Finding:** Global R² can be misleading! A model can achieve high global R² by\n")
    f.write("capturing between-condition variance while poorly fitting within-condition decay shapes.\n\n")
    f.write("| Metric | Best Model | Value |\n")
    f.write("|--------|------------|-------|\n")
    f.write(f"| Global R² | {best['model_name']} | {best['r2']:.4f} |\n")
    f.write(f"| Mean Local R² | {best['model_name']} | {np.nanmean(local_r2_matrix):.2f} |\n\n")
    f.write(f"**Recommendation:** Use `{best['model_name']}` model.\n\n")
    f.write("---\n\n")

    f.write("## Data\n\n")
    f.write("| Property | Value |\n")
    f.write("|----------|-------|\n")
    f.write(f"| Total data points | {len(pooled_df):,} |\n")
    f.write(f"| Model sizes | {', '.join(unique_params)} |\n")
    f.write(f"| Replica levels (filtered) | {', '.join(map(str, valid_replicas_unified))} |\n")
    f.write(f"| Token index range | {int(pooled_df['t'].min())} to {int(pooled_df['t'].max())} |\n\n")
    f.write("**Note:** Excludes R=0 (uncontaminated baseline) and R>=1000 (NLL uptick due to\n")
    f.write("hard-to-memorize long sequences). See `NLL_UPTICK_ANALYSIS.md` for details.\n\n")
    f.write("---\n\n")

    f.write("## Models Tested\n\n")
    f.write("### Ranked by Global R²\n\n")
    f.write("| Model | R² | RMSE | Parameters |\n")
    f.write("|-------|-----|------|------------|\n")
    for _, row in comparison_df.sort_values("R²", ascending=False).iterrows():
        f.write(f"| {row['Model']} | {row['R²']:.4f} | {row['RMSE']:.3f} | {row['N_params']} |\n")
    f.write("\n---\n\n")

    f.write(f"## Best Model: {best['model_name']}\n\n")
    f.write(f"**Fitted Parameters ({len(best['params'])}):**\n")
    f.write("```\n")
    f.write(f"{best['params']}\n")
    f.write("```\n\n")
    f.write("**Local R² Statistics:**\n")
    f.write("| Statistic | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| Mean | {np.nanmean(local_r2_matrix):.2f} |\n")
    f.write(f"| Std | {np.nanstd(local_r2_matrix):.2f} |\n")
    f.write(f"| Min | {np.nanmin(local_r2_matrix):.2f} |\n")
    f.write(f"| Max | {np.nanmax(local_r2_matrix):.2f} |\n\n")
    f.write("---\n\n")

    f.write("## Generated Files\n\n")
    f.write("- `MODEL_FITTING_REPORT.md` - This report\n")
    f.write("- `parameter_visualization.pdf/png` - Model predictions across parameter space\n")
    f.write("- `y=fit_params_x=num_replicas_hue=model_size.pdf/png` - Floor model parameters vs R\n")
    f.write("- `y=fit_params_x=model_size_hue=num_replicas.pdf/png` - Floor model parameters vs N\n")

print(f"\nReport saved to: {report_path}")

# %%
# =============================================================================
# PARAMETER VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("PARAMETER VISUALIZATION")
print("=" * 70)

# Visualize model predictions across parameter space
plt.close()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: NLL vs t for different R (fixed N = median)
ax = axes[0, 0]
N_median = np.median([p / N_scale for p in [34e6, 62e6, 93e6, 153e6, 344e6]])
t_range = np.logspace(0, 3, 100)
for R in [0, 1, 10, 100, 1000]:
    nll_pred = best["model_func"](t_range, N_median, R, best["params"])
    ax.plot(t_range, nll_pred, label=f"R={R}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Token Index t")
ax.set_ylabel("NLL")
ax.set_title(f"NLL vs t for different R\n(N = {N_median*N_scale/1e6:.0f}M)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: NLL vs t for different N (fixed R = 100)
ax = axes[0, 1]
R_fixed = 100
for N_val in [34e6, 93e6, 344e6]:
    N_scaled = N_val / N_scale
    nll_pred = best["model_func"](t_range, N_scaled, R_fixed, best["params"])
    ax.plot(t_range, nll_pred, label=f"N={N_val/1e6:.0f}M")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Token Index t")
ax.set_ylabel("NLL")
ax.set_title(f"NLL vs t for different N\n(R = {R_fixed})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: NLL at t=100 vs R for different N
ax = axes[0, 2]
R_range = np.logspace(0, 3.5, 50)
t_fixed = 100
for N_val in [34e6, 93e6, 344e6]:
    N_scaled = N_val / N_scale
    nll_pred = best["model_func"](t_fixed, N_scaled, R_range, best["params"])
    ax.plot(R_range, nll_pred, label=f"N={N_val/1e6:.0f}M")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Num. Replicas R")
ax.set_ylabel("NLL")
ax.set_title(f"NLL at t={t_fixed} vs R")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Decay rate (derivative) vs R
ax = axes[1, 0]
t_test = 50
dt = 1
for N_val in [34e6, 93e6, 344e6]:
    N_scaled = N_val / N_scale
    nll_t = best["model_func"](t_test, N_scaled, R_range, best["params"])
    nll_t_plus = best["model_func"](t_test + dt, N_scaled, R_range, best["params"])
    decay_rate = -(nll_t_plus - nll_t) / dt
    ax.plot(R_range, decay_rate, label=f"N={N_val/1e6:.0f}M")
ax.set_xscale("log")
ax.set_xlabel("Num. Replicas R")
ax.set_ylabel("-dNLL/dt")
ax.set_title(f"Decay Rate at t={t_test}")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Asymptotic floor vs R
ax = axes[1, 1]
t_asymp = 800  # Large t to approximate floor
for N_val in [34e6, 93e6, 344e6]:
    N_scaled = N_val / N_scale
    nll_floor = best["model_func"](t_asymp, N_scaled, R_range, best["params"])
    ax.plot(R_range, nll_floor, label=f"N={N_val/1e6:.0f}M")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Num. Replicas R")
ax.set_ylabel("NLL (floor)")
ax.set_title(f"Asymptotic NLL (t={t_asymp})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Model comparison (top 3 models)
ax = axes[1, 2]
N_test = 153e6 / N_scale
R_test = 100
for model in top_3:
    nll_pred = model["model_func"](t_range, N_test, R_test, model["params"])
    ax.plot(t_range, nll_pred, label=f"{model['model_name']} (R²={model['r2']:.3f})")
# Add actual data for this condition
subset = nll_by_token_df[
    (nll_by_token_df["Parameters"] == "153M")
    & (nll_by_token_df["Num. MATH Test Set Replicas"] == R_test)
]
if len(subset) > 0:
    ax.scatter(
        subset["Token Index"],
        subset["mean_NLL"],
        s=20,
        c="black",
        alpha=0.5,
        label="Data",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Token Index t")
ax.set_ylabel("NLL")
ax.set_title(f"Top 3 Models Comparison\n(N=153M, R={R_test})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="parameter_visualization",
)
plt.close()

print("\nParameter visualization complete.")
print(f"\nAll results saved to: {results_dir}")

# %%
# =============================================================================
# ENHANCED REPORT: Compare Global vs Local Fit Quality
# =============================================================================
print("\n" + "=" * 70)
print("ENHANCED ANALYSIS: Global vs Local Fit Quality")
print("=" * 70)


def compute_local_r2_for_model(model_result):
    """Compute local R² for each condition."""
    local_r2s = []
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
            N = subset["Num. Parameters"].iloc[0] / N_scale
            R = replica

            nll_pred = model_result["model_func"](t, N, R, model_result["params"])
            local_r2 = compute_r_squared(nll, nll_pred)
            local_r2s.append(
                {
                    "Parameters": param,
                    "Replicas": replica,
                    "local_r2": local_r2,
                }
            )
    return pd.DataFrame(local_r2s)


# Compute local R² for top models
enhanced_comparison = []
for model_result in all_results:
    local_df = compute_local_r2_for_model(model_result)
    enhanced_comparison.append(
        {
            "Model": model_result["model_name"],
            "N_params": len(model_result["params"]),
            "Global_R2": model_result["r2"],
            "Mean_Local_R2": local_df["local_r2"].mean(),
            "Median_Local_R2": local_df["local_r2"].median(),
            "Min_Local_R2": local_df["local_r2"].min(),
            "Pct_Positive_Local": (local_df["local_r2"] > 0).mean() * 100,
        }
    )

enhanced_df = pd.DataFrame(enhanced_comparison)

# Sort by mean local R² (more meaningful for actual fit quality)
print("\nModels ranked by MEAN LOCAL R² (actual fit quality):")
print(
    enhanced_df.sort_values("Mean_Local_R2", ascending=False)[
        ["Model", "N_params", "Global_R2", "Mean_Local_R2", "Pct_Positive_Local"]
    ]
    .round(4)
    .to_string(index=False)
)

# Find best model by local R²
best_by_local = enhanced_df.loc[enhanced_df["Mean_Local_R2"].idxmax()]
print(f"\nBest model by MEAN LOCAL R²: {best_by_local['Model']}")
print(f"  Global R²: {best_by_local['Global_R2']:.4f}")
print(f"  Mean Local R²: {best_by_local['Mean_Local_R2']:.4f}")
print(f"  % Positive Local R²: {best_by_local['Pct_Positive_Local']:.1f}%")

# Note: Enhanced report removed - all info now in MODEL_FITTING_REPORT.md
print("\n" + "=" * 70)
print("OVERNIGHT FITTING COMPLETE")
print("=" * 70)

# %%
# =============================================================================
# SUMMARY OF MODEL FITTING ANALYSIS (2025-01-19)
# =============================================================================
#
# BEST MODEL (per-condition fits):
#
#    NLL(t) = NLL_∞ + A * t^(-α)
#
#    where:
#        - NLL_∞: irreducible error floor (asymptote as t→∞)
#        - A: prefactor (amplitude of decay)
#        - α: decay exponent
#
# FITTING PROCEDURE:
#    1. For each (N, R) condition, fit three parameters: NLL_∞, A, α
#    2. Used scipy.optimize.curve_fit with bounds:
#       - NLL_∞ ∈ [0, max(NLL)]
#       - A ∈ [0, 10 × max(NLL)]
#       - α ∈ [0.001, 2.0]
#    3. Initial guesses:
#       - NLL_∞ = mean of last 10 data points
#       - A = NLL[0] - NLL_∞
#       - α = 0.1
#
# RESULTS (0 < R < 1000):
#    - Mean R² = 0.89
#    - 30/30 conditions have R² > 0.5
#    - 28/30 conditions have R² > 0.8
#
# PARAMETER DEPENDENCIES ON N AND R:
#    log(NLL_∞) = 6.24 - 1.03*log(N) - 0.73*log(R), R² = 0.80
#    log(A) = 4.62 - 0.70*log(N) - 0.57*log(R), R² = 0.57
#    α = 0.16 + 0.05*log(N) + 0.08*log(R), R² = 0.26
#
# EXCLUDED DATA:
#    - R=0: No recall effect (different regime)
#    - R≥1000: Anomalous behavior (saturation)
#
# KEY OUTPUT:
#    y=nll_x=token_index_hue=num_replicas_col=model_size_fit.pdf
#
# =============================================================================

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

# Color palette for replicas
colors_R = sns.color_palette("viridis", len(valid_replicas))
R_to_color = {r: colors_R[i] for i, r in enumerate(valid_replicas)}

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

# Remove unused subplot
axes[5].axis("off")

# Add legend to last used subplot
handles = [
    plt.Line2D([0], [0], color=R_to_color[r], marker="o", linestyle="-", markersize=5)
    for r in valid_replicas
]
axes[4].legend(
    handles, [str(r) for r in valid_replicas], title="Num. Replicas", loc="lower left"
)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=nll_x=token_index_hue=num_replicas_col=model_size_fit",
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

axes[2].legend(title=r"Model Size", loc="best")
plt.tight_layout()

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit_params_x=num_replicas_hue=model_size",
)
# plt.show()

# %%
# Visualization 2: Parameters vs N (x, log scale) with replicas (hue)
# Three columns for NLL_∞, A, α

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Color palette for replicas (symlog scale matching other plots) - use "viridis" for replicas
# Use all replicas for consistent colors
all_replicas = [0, 1, 3, 10, 32, 100, 316, 1000, 3162]
colors_R = sns.color_palette("viridis", len(all_replicas))
R_to_color = {r: colors_R[i] for i, r in enumerate(all_replicas)}

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

axes[2].legend(title=r"Replicas $R$", loc="best")
plt.tight_layout()

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit_params_x=model_size_hue=num_replicas",
)
# plt.show()
