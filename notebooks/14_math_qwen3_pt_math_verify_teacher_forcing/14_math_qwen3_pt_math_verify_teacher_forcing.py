"""Analyze teacher-forced evaluation results: NLL by token position.

This notebook analyzes how negative log likelihood varies across token positions
in the solution, broken down by model size and contamination level (num replicas).
"""

import ast
import gc
import hashlib
import os
import re

from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

import src.analyze
import src.globals
import src.plot


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

# Get log_prob columns - sample uniformly in log space for memory efficiency
all_log_prob_cols = sorted(
    [c for c in all_columns if c.startswith("log_prob_token_")],
    key=lambda x: int(x.replace("log_prob_token_", "")),
)
max_token_idx = max(int(c.replace("log_prob_token_", "")) for c in all_log_prob_cols)
print(f"Max token index in data: {max_token_idx}")

# Sample ~200 token positions uniformly in log space (after adding 1 for log scaling)
num_samples = 200
log_spaced_indices = np.unique(
    np.geomspace(1, max_token_idx + 1, num_samples).astype(int) - 1
).tolist()
# Always include token 0
if 0 not in log_spaced_indices:
    log_spaced_indices = [0] + log_spaced_indices
log_spaced_indices = sorted(log_spaced_indices)

log_prob_cols = [
    f"log_prob_token_{i}"
    for i in log_spaced_indices
    if f"log_prob_token_{i}" in all_log_prob_cols
]
print(
    f"Using {len(log_prob_cols)} token positions (log-spaced from 0 to {max_token_idx})"
)

# Process each row group
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

# Filter to Token Index <= 1000 so ylim is set correctly (but keep ALL replicas)
plot1_df = nll_by_token_df[nll_by_token_df["Token Index + 1"] <= 1000].copy()

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
    facet_kws={"sharey": True, "sharex": True},
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
    xlim=(1, 1e3),
    ylim=(1e-3, 1e1),
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
n_params = len(unique_params)
flare_colors = sns.color_palette("flare", n_params)
params_palette = {p: flare_colors[i] for i, p in enumerate(unique_params)}

# Filter to Token Index <= 1000 so ylim is set correctly, then compute CI bounds
plot2_df = nll_by_token_df[nll_by_token_df["Token Index + 1"] <= 1000].copy()
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
    xlim=(1, 1e3),
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
    xlim=(1, 1e3),
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
    xlim=(1, 1e3),
)
g.set_titles(r"Replicas: {col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title=r"Model Size")
enable_minor_gridlines(g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=prob_x=token_index_hue=model_size_col=num_replicas",
)
# plt.show()
