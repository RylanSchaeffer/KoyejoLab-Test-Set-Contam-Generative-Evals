"""Does SFT induce generalization or just preserve memorization?

Analyzes teacher-forced evaluation on *perturbed* MATH problems for
pre-SFT vs post-SFT models. If post-SFT cross-entropy on perturbed problems
drops relative to pre-SFT, that's direct evidence of generalization —
the model learned to solve math, not just memorize solutions.

This is the key experiment for Reviewer 6RQA, who identified memorization vs.
generalization disentanglement as "the difference between a borderline and
an excellent contribution."

Sweep: onaspopu (memorization-scoring-vs-sampling-eval-teacher-forcing)
  - 344M: 9 pre-SFT + 9 post-SFT = 18 runs on RylanSchaeffer/math_perturbed
  - 153M: 8 pre-SFT + 8 post-SFT = 16 runs on RylanSchaeffer/math_perturbed
"""

import ast
import gc
import hashlib
import os
import re

import matplotlib
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

import src.analyze
import src.globals
import src.plot

src.globals.MODEL_NAMES_TO_PARAMETERS_DICT["153M"] = 153e6

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# ===========================================================================
# 1. Load perturbed MATH teacher-forcing data (pre-SFT + post-SFT)
# ===========================================================================
perturbed_tf_sweep_ids = [
    "onaspopu",  # 344M + 153M, pre-SFT + post-SFT, on RylanSchaeffer/math_perturbed
]

configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
    data_dir=data_dir,
    sweep_ids=perturbed_tf_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

configs_df["Model"] = configs_df["model_config"].apply(
    lambda mc: ast.literal_eval(mc)["model"]
)
configs_df["Parameters"] = configs_df["Model"].apply(
    lambda m: re.search(r"Qwen3-([\d.]+[MB])", m).group(1)
)
configs_df["Num. Parameters"] = configs_df["Parameters"].apply(
    lambda p: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p]
)
configs_df["Num. Replicas Per Epoch"] = configs_df["Model"].apply(
    lambda m: int(re.search(r"rep_(\d+)_sbst", m).group(1))
)
configs_df["Num. Epochs"] = configs_df["Model"].apply(
    lambda m: int(re.search(r"epch_(\d+)_ot", m).group(1))
)
configs_df["Num. MATH Test Set Replicas"] = (
    configs_df["Num. Replicas Per Epoch"] * configs_df["Num. Epochs"]
)
configs_df["SFT"] = configs_df["Model"].str.endswith("_sft")
configs_df["Stage"] = configs_df["SFT"].map({False: "Pre-SFT", True: "Post-SFT"})

print(f"Loaded {len(configs_df)} runs")
print(configs_df.groupby(["Parameters", "Stage"]).size())

# ===========================================================================
# 2. Compute mean NLL per run from per-token log probs
# ===========================================================================
# Download histories (this creates the parquet file)
src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
    data_dir=data_dir,
    sweep_ids=perturbed_tf_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    filetype="parquet",
)

filename = "sweeps=" + ",".join(perturbed_tf_sweep_ids)
hashed_filename = hashlib.md5(filename.encode()).hexdigest()
histories_path = os.path.join(data_dir, hashed_filename + "_runs_histories.parquet")

print(f"\nLoading histories from {histories_path}")
parquet_file = pq.ParquetFile(histories_path)
all_columns = [field.name for field in parquet_file.schema]

# Get all log_prob columns
log_prob_cols = sorted(
    [c for c in all_columns if c.startswith("log_prob_token_")],
    key=lambda x: int(x.replace("log_prob_token_", "")),
)
print(f"Found {len(log_prob_cols)} log_prob columns")

run_id_to_config = configs_df.set_index("run_id")[
    ["Parameters", "Num. Parameters", "Num. MATH Test Set Replicas", "Stage"]
].to_dict("index")

# Compute mean NLL per run (average across all tokens and all problems)
num_row_groups = parquet_file.metadata.num_row_groups
mean_nll_results = []

for rg_idx in range(num_row_groups):
    print(f"  Processing row group {rg_idx + 1}/{num_row_groups}...")

    # Process in column batches to manage memory
    batch_size = 200
    table = parquet_file.read_row_group(rg_idx, columns=["run_id"])
    run_ids_df = table.to_pandas()
    del table
    gc.collect()

    for run_id in run_ids_df["run_id"].unique():
        if run_id not in run_id_to_config:
            continue

        config = run_id_to_config[run_id]
        run_mask = run_ids_df["run_id"] == run_id
        num_sequences = run_mask.sum()

        # Accumulate NLL sum and count across column batches
        total_nll_sum = 0.0
        total_count = 0

        for batch_start in range(0, len(log_prob_cols), batch_size):
            batch_end = min(batch_start + batch_size, len(log_prob_cols))
            batch_cols = log_prob_cols[batch_start:batch_end]

            table = parquet_file.read_row_group(
                rg_idx, columns=["run_id"] + batch_cols
            )
            batch_df = table.to_pandas()
            del table

            run_data = batch_df[batch_df["run_id"] == run_id][batch_cols]
            # NLL = -log_prob. Sum all non-NaN values.
            nll_vals = -run_data.values
            valid_mask = ~np.isnan(nll_vals)
            total_nll_sum += np.nansum(nll_vals)
            total_count += valid_mask.sum()

            del batch_df, run_data
            gc.collect()

        if total_count > 0:
            mean_nll_results.append(
                {
                    "run_id": run_id,
                    "Parameters": config["Parameters"],
                    "Num. Parameters": config["Num. Parameters"],
                    "Num. MATH Test Set Replicas": config[
                        "Num. MATH Test Set Replicas"
                    ],
                    "Stage": config["Stage"],
                    "Mean NLL": total_nll_sum / total_count,
                    "Total Tokens": total_count,
                }
            )

    del run_ids_df
    gc.collect()

mean_nll_df = pd.DataFrame(mean_nll_results)
print(f"\nComputed mean NLL for {len(mean_nll_df)} runs")
print(mean_nll_df.groupby(["Parameters", "Stage"])[["Mean NLL"]].describe())

# ===========================================================================
# 3. Also load original MATH teacher-forcing for comparison (best-effort)
# ===========================================================================
original_tf_sweep_ids = [
    "em23bzb7",  # 153M pre-SFT on original MATH
    "sy8h8i80",  # 344M pre-SFT on original MATH
]

orig_mean_nll_df = None
try:
    orig_configs_df = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
        data_dir=data_dir,
        sweep_ids=original_tf_sweep_ids,
        refresh=refresh,
        wandb_username="rylan",
        finished_only=True,
    )

    orig_configs_df["Model"] = orig_configs_df["model_config"].apply(
        lambda mc: ast.literal_eval(mc)["model"]
    )
    orig_configs_df["Parameters"] = orig_configs_df["Model"].apply(
        lambda m: re.search(r"Qwen3-([\d.]+[MB])", m).group(1)
    )
    orig_configs_df["Num. Parameters"] = orig_configs_df["Parameters"].apply(
        lambda p: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p]
    )
    orig_configs_df["Num. Replicas Per Epoch"] = orig_configs_df["Model"].apply(
        lambda m: int(re.search(r"rep_(\d+)_sbst", m).group(1))
    )
    orig_configs_df["Num. Epochs"] = orig_configs_df["Model"].apply(
        lambda m: int(re.search(r"epch_(\d+)_ot", m).group(1))
    )
    orig_configs_df["Num. MATH Test Set Replicas"] = (
        orig_configs_df["Num. Replicas Per Epoch"] * orig_configs_df["Num. Epochs"]
    )
    orig_configs_df["Stage"] = "Pre-SFT"

    src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
        data_dir=data_dir,
        sweep_ids=original_tf_sweep_ids,
        refresh=refresh,
        wandb_username="rylan",
        filetype="parquet",
    )

    orig_filename = "sweeps=" + ",".join(original_tf_sweep_ids)
    orig_hashed = hashlib.md5(orig_filename.encode()).hexdigest()
    orig_histories_path = os.path.join(
        data_dir, orig_hashed + "_runs_histories.parquet"
    )

    orig_run_id_to_config = orig_configs_df.set_index("run_id")[
        ["Parameters", "Num. Parameters", "Num. MATH Test Set Replicas", "Stage"]
    ].to_dict("index")

    print(f"\nLoading original MATH TF histories from {orig_histories_path}")
    orig_pq = pq.ParquetFile(orig_histories_path)
    orig_log_prob_cols = sorted(
        [
            c
            for c in [f.name for f in orig_pq.schema]
            if c.startswith("log_prob_token_")
        ],
        key=lambda x: int(x.replace("log_prob_token_", "")),
    )

    orig_mean_nll_results = []
    for rg_idx in range(orig_pq.metadata.num_row_groups):
        print(
            f"  Processing row group {rg_idx + 1}/{orig_pq.metadata.num_row_groups}..."
        )

        table = orig_pq.read_row_group(rg_idx, columns=["run_id"])
        run_ids_df = table.to_pandas()
        del table
        gc.collect()

        for run_id in run_ids_df["run_id"].unique():
            if run_id not in orig_run_id_to_config:
                continue

            config = orig_run_id_to_config[run_id]
            total_nll_sum = 0.0
            total_count = 0

            for batch_start in range(0, len(orig_log_prob_cols), batch_size):
                batch_end = min(batch_start + batch_size, len(orig_log_prob_cols))
                batch_cols = orig_log_prob_cols[batch_start:batch_end]

                table = orig_pq.read_row_group(
                    rg_idx, columns=["run_id"] + batch_cols
                )
                batch_df = table.to_pandas()
                del table

                run_data = batch_df[batch_df["run_id"] == run_id][batch_cols]
                nll_vals = -run_data.values
                total_nll_sum += np.nansum(nll_vals)
                total_count += (~np.isnan(nll_vals)).sum()

                del batch_df, run_data
                gc.collect()

            if total_count > 0:
                orig_mean_nll_results.append(
                    {
                        "run_id": run_id,
                        "Parameters": config["Parameters"],
                        "Num. Parameters": config["Num. Parameters"],
                        "Num. MATH Test Set Replicas": config[
                            "Num. MATH Test Set Replicas"
                        ],
                        "Stage": config["Stage"],
                        "Mean NLL": total_nll_sum / total_count,
                        "Total Tokens": total_count,
                    }
                )

        del run_ids_df
        gc.collect()

    orig_mean_nll_df = pd.DataFrame(orig_mean_nll_results)
    print(f"Loaded {len(orig_mean_nll_df)} original MATH TF runs")

except Exception as e:
    print(f"\nWARNING: Could not load original MATH TF data: {e}")
    print("Skipping comparison plot. Re-run when W&B API is available.")

# ===========================================================================
# 4. Plot 1: Pre-SFT vs Post-SFT mean NLL on perturbed MATH
# ===========================================================================
# Color palette: pre-SFT vs post-SFT
stage_palette = {"Pre-SFT": "#4C72B0", "Post-SFT": "#DD8452"}

plt.close()
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for idx, param_label in enumerate(["153M", "344M"]):
    ax = axes[idx]
    subset = mean_nll_df[mean_nll_df["Parameters"] == param_label]

    for stage, color in stage_palette.items():
        stage_data = subset[subset["Stage"] == stage].sort_values(
            "Num. MATH Test Set Replicas"
        )
        ax.plot(
            stage_data["Num. MATH Test Set Replicas"],
            stage_data["Mean NLL"],
            marker="o",
            color=color,
            label=stage,
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_title(f"Qwen3-{param_label}")
    ax.set_xlabel(r"Num. MATH Test Set Replicas ($R$)")
    ax.grid(True, alpha=0.3, which="both")

axes[0].set_ylabel(r"Mean NLL on Perturbed MATH")

handles = [
    plt.Line2D([0], [0], color=c, marker="o", linestyle="-", markersize=6)
    for c in stage_palette.values()
]
fig.legend(
    handles,
    list(stage_palette.keys()),
    loc="upper left",
    bbox_to_anchor=(1, 1),
)
plt.tight_layout()
plt.subplots_adjust(right=0.88)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=mean_nll_perturbed_x=num_replicas_hue=stage_col=model_size",
)
plt.close()

# ===========================================================================
# 5. Plot 2: Delta NLL (Post-SFT minus Pre-SFT) on perturbed MATH
# ===========================================================================
# Compute delta: negative means SFT reduced NLL (= improved = generalization)
delta_rows = []
for param_label in ["153M", "344M"]:
    subset = mean_nll_df[mean_nll_df["Parameters"] == param_label]
    pre = subset[subset["Stage"] == "Pre-SFT"].set_index("Num. MATH Test Set Replicas")
    post = subset[subset["Stage"] == "Post-SFT"].set_index(
        "Num. MATH Test Set Replicas"
    )
    common_R = pre.index.intersection(post.index)
    for R in common_R:
        delta_rows.append(
            {
                "Parameters": param_label,
                "Num. MATH Test Set Replicas": R,
                "Delta Mean NLL": post.loc[R, "Mean NLL"] - pre.loc[R, "Mean NLL"],
            }
        )

delta_df = pd.DataFrame(delta_rows)

plt.close()
fig, ax = plt.subplots(figsize=(10.67, 8))

param_colors = {"153M": "#4C72B0", "344M": "#DD8452"}
for param_label, color in param_colors.items():
    d = delta_df[delta_df["Parameters"] == param_label].sort_values(
        "Num. MATH Test Set Replicas"
    )
    ax.plot(
        d["Num. MATH Test Set Replicas"],
        d["Delta Mean NLL"],
        marker="o",
        color=color,
        label=f"Qwen3-{param_label}",
        linewidth=2,
        markersize=6,
    )

ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel(r"Num. MATH Test Set Replicas ($R$)")
ax.set_ylabel(r"$\Delta$ Mean NLL (Post-SFT $-$ Pre-SFT)")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# Annotate interpretation
ax.annotate(
    r"$\leftarrow$ SFT improves (generalization)",
    xy=(0.02, 0.02),
    xycoords="axes fraction",
    fontsize=16,
    color="green",
    alpha=0.7,
)
ax.annotate(
    r"SFT hurts (forgetting) $\rightarrow$",
    xy=(0.02, 0.95),
    xycoords="axes fraction",
    fontsize=16,
    color="red",
    alpha=0.7,
)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=delta_nll_perturbed_x=num_replicas_hue=model_size",
)
plt.close()

# ===========================================================================
# 6. Plot 3: Comparison — original MATH vs perturbed MATH (pre-SFT only)
# ===========================================================================
if orig_mean_nll_df is None:
    print("\nSkipping Plot 3 (original MATH comparison) — data not available.")
else:
    orig_mean_nll_df["Dataset"] = "Original MATH"
    perturbed_presft = mean_nll_df[mean_nll_df["Stage"] == "Pre-SFT"].copy()
    perturbed_presft["Dataset"] = "Perturbed MATH"
    comparison_df = pd.concat(
        [
            orig_mean_nll_df[
                ["Parameters", "Num. MATH Test Set Replicas", "Mean NLL", "Dataset"]
            ],
            perturbed_presft[
                ["Parameters", "Num. MATH Test Set Replicas", "Mean NLL", "Dataset"]
            ],
        ],
        ignore_index=True,
    )

    dataset_palette = {"Original MATH": "#4C72B0", "Perturbed MATH": "#DD8452"}

    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, param_label in enumerate(["153M", "344M"]):
        ax = axes[idx]
        subset = comparison_df[comparison_df["Parameters"] == param_label]

        for dataset_label, color in dataset_palette.items():
            d = subset[subset["Dataset"] == dataset_label].sort_values(
                "Num. MATH Test Set Replicas"
            )
            ax.plot(
                d["Num. MATH Test Set Replicas"],
                d["Mean NLL"],
                marker="o",
                color=color,
                label=dataset_label,
                linewidth=2,
                markersize=6,
            )

        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_title(f"Qwen3-{param_label}")
        ax.set_xlabel(r"Num. MATH Test Set Replicas ($R$)")
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel(r"Mean NLL (Pre-SFT)")

    handles = [
        plt.Line2D([0], [0], color=c, marker="o", linestyle="-", markersize=6)
        for c in dataset_palette.values()
    ]
    fig.legend(
        handles,
        list(dataset_palette.keys()),
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename="y=mean_nll_x=num_replicas_hue=dataset_col=model_size",
    )
    plt.close()

# ===========================================================================
# 7. Print summary statistics for the rebuttal
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY FOR REVIEWER 6RQA REBUTTAL")
print("=" * 70)

for param_label in ["153M", "344M"]:
    print(f"\n--- Qwen3-{param_label} ---")
    subset = mean_nll_df[mean_nll_df["Parameters"] == param_label]

    pre = subset[subset["Stage"] == "Pre-SFT"].set_index("Num. MATH Test Set Replicas")
    post = subset[subset["Stage"] == "Post-SFT"].set_index(
        "Num. MATH Test Set Replicas"
    )

    print(f"{'R':>6s}  {'Pre-SFT NLL':>12s}  {'Post-SFT NLL':>13s}  {'Delta':>8s}  {'Direction':>12s}")
    common_R = sorted(pre.index.intersection(post.index))
    for R in common_R:
        pre_nll = pre.loc[R, "Mean NLL"]
        post_nll = post.loc[R, "Mean NLL"]
        delta = post_nll - pre_nll
        direction = "IMPROVED" if delta < 0 else "WORSENED"
        print(f"{R:>6d}  {pre_nll:>12.4f}  {post_nll:>13.4f}  {delta:>+8.4f}  {direction:>12s}")

    # Compare with original MATH
    if orig_mean_nll_df is None:
        print("  (Original MATH comparison not available)")
        continue
    orig_subset = orig_mean_nll_df[orig_mean_nll_df["Parameters"] == param_label]
    if len(orig_subset) > 0:
        print(f"\n  Original MATH NLL (pre-SFT) for comparison:")
        for _, row in orig_subset.sort_values("Num. MATH Test Set Replicas").iterrows():
            print(f"    R={int(row['Num. MATH Test Set Replicas']):>5d}: {row['Mean NLL']:.4f}")

print(f"\n{'=' * 70}")
print("Key question: Does SFT reduce NLL on perturbed problems?")
print("If Delta < 0 at any R: evidence of generalization (not just memorization)")
print("If Delta > 0 at all R: SFT only helped via memorization, no transfer")
print("=" * 70)

print("\nFinished 16_sft_generalization_teacher_forcing_perturbed.py")
