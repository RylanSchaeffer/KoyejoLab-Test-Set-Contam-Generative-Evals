import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "9auakwfg",
]

sft_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-sft",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

sft_runs_configs_df = (
    src.analyze.add_pretraining_quantities_to_supervised_finetuning_runs_configs_df(
        sft_runs_configs_df=sft_runs_configs_df
    )
)

sft_runs_configs_df["Loss Difference"] = (
    sft_runs_configs_df["eval_after/eval_loss"]
    - sft_runs_configs_df["eval_before/eval_loss"]
)
sft_runs_configs_df["Loss Ratio"] = (
    sft_runs_configs_df["eval_after/eval_loss"]
    / sft_runs_configs_df["eval_before/eval_loss"]
)

num_replicas_sym_norm = SymLogNorm(
    linthresh=1.0,
    vmin=sft_runs_configs_df["Num. MATH Test Set Replicas"].min(),
    vmax=sft_runs_configs_df["Num. MATH Test Set Replicas"].max(),
)

num_parameters_log_norm = LogNorm(
    vmin=sft_runs_configs_df["Num. Parameters"].min(),
    vmax=sft_runs_configs_df["Num. Parameters"].max(),
)

# Easier than changing the legend title.
sft_runs_configs_df["Num. Replicas"] = sft_runs_configs_df[
    "Num. MATH Test Set Replicas"
]
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="eval_after/eval_loss",
    hue="Num. Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    # marker="o",  # Keeps markers as circles, style will vary dashes
    legend=False,
    ax=axes[0],
)
sns.scatterplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="eval_after/eval_loss",
    hue="Num. Replicas",
    hue_norm=num_replicas_sym_norm,
    style="Parameters",
    style_order=["34M", "62M", "93M", "153M", "344M"],
    palette="viridis",
    legend=False,
    ax=axes[0],
    s=125,
)
axes[0].plot([0, 7.5], [0, 7.5], linestyle="--", color="black")
axes[0].set(
    xlim=(0, 7.5),
    xlabel="Loss on MATH Test Before SFT",
    ylim=(0, 7.5),
    ylabel="Loss on MATH Test After SFT",
)
axes[0].text(
    x=3.6,
    y=1.2,
    s="SFT on Train\nHelps on Test",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,
    fontweight="bold",
)
axes[0].text(
    x=1.6,
    y=3.5,
    s="SFT on Train\nHurts on Test",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,
)
# Position the text slightly above the line to avoid overlap
axes[0].text(
    x=5.8,
    y=5.8,
    s="SFT On Train Has No Effect",
    rotation=45,
    rotation_mode="anchor",
    transform_rotates_text=True,  # Ensures rotation stays relative to the line
    verticalalignment="bottom",
    horizontalalignment="center",
    fontsize=16,
    color="black",
)
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="Loss Ratio",
    hue="Num. Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    ax=axes[1],
    legend=False,
)
g = sns.scatterplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="Loss Ratio",
    hue="Num. Replicas",
    hue_norm=num_replicas_sym_norm,
    style="Parameters",
    style_order=["34M", "62M", "93M", "153M", "344M"],
    palette="viridis",
    ax=axes[1],
    s=125,
    legend="full",
)
axes[1].text(
    x=0.15,
    y=1.2,
    s="SFT on Train Hurts on Test",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,
)
axes[1].text(
    x=0.15,
    y=0.8,
    s="SFT on Train Helps on Test",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,
)
axes[1].plot([0.03, 7.5], [1.0, 1.00], linestyle="--", color="black")
axes[1].set(
    xlabel="Loss on MATH Test Before SFT",
    ylabel=r"Pre-SFT Test Loss / Post-SFT Test Loss",
    yscale="log",
    xscale="log",
)
axes[1].yaxis.set_label_coords(-0.12, 0.4)
all_replicas = ["0", "1", "3", "10", "32", "100", "316", "1000", "3162"]
keep_replicas = ["0", "10", "100", "1000", "3162"]
# Create a set of replicas to HIDE (anything in 'all' but not in 'keep')
remove_replicas = set([r for r in all_replicas if r not in keep_replicas])
# Get the handles from the scatterplot (which now has ALL data points)
handles, labels = axes[1].get_legend_handles_labels()
subset_handles = []
subset_labels = []
for h, l in zip(handles, labels):
    # If this label is one of the replicas we want to hide, skip it
    if l in remove_replicas:
        continue
    # Otherwise keep it. This preserves:
    # 1. "Num. Replicas" (title)
    # 2. The replicas we want (0, 10, 100...)
    # 3. "Num. Parameters" (title)
    # 4. The parameter labels (7B, etc.)
    subset_handles.append(h)
    subset_labels.append(l)
# Recreate the legend with the filtered list
# Use axes[1].legend instead of sns.move_legend for manual control
axes[1].legend(
    subset_handles,
    subset_labels,
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=True,
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1.03))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="combined_loss_analysis_hue=replicas_style=params",
)
plt.show()

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="Loss Difference",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    # style="Parameters",
    legend="full",
)
plt.plot([0.03, 7.5], [0, 0], linestyle="--", color="black")
g.set(
    # xscale="log",
    xlabel="Pre-SFT Loss on MATH Test",
    # ylabel="Pre-SFT Test Loss - Post-SFT Test Loss",
    ylabel="$\Delta$ in Test Loss: Pre-SFT - Post-SFT",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title="Num. Replicas")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=loss-diff_x=loss-before_hue=replicas"
)
# plt.show()

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="Loss Ratio",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    # style="Parameters",
    legend="full",
)
plt.plot([0.03, 7.5], [1.0, 1.00], linestyle="--", color="black")
g.set(
    # xscale="log",
    xlabel="Loss on MATH Test Before SFT",
    # ylabel=r"$\frac{\text{Test Loss After SFT}}{\text{Test Loss Before SFT}}$",
    # ylabel="Loss After SFT / Loss Before SFT",
    ylabel="$\Delta$ in Test Loss: Pre-SFT / Post-SFT",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title="Num. Replicas")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=loss-ratio_x=loss-before_hue=replicas"
)
# plt.show()

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="eval_after/eval_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    # style="Parameters",
    legend="full",
)
plt.plot(
    [0, 7.5],
    [0, 7.5],
    linestyle="--",
    color="black",
)
g.set(
    xlim=(0, 7.5),
    xlabel="Loss on MATH Test Before SFT",
    ylim=(0, 7.5),
    ylabel="Loss on MATH Test After SFT",
)
plt.text(
    x=3.6,  # Horizontal position
    y=1.2,  # Vertical position (slightly below the y=x line)
    s="SFTing on Train\nHelps on Test",  # The text string
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,  # Adjust based on your preference
    fontweight="bold",
)
plt.text(
    x=1.6,  # Horizontal position
    y=3.5,  # Vertical position (slightly below the y=x line)
    s="SFTing on Train\nHurts on Test",  # The text string
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,  # Adjust based on your preference
)
sns.move_legend(g, "upper left", title="Num. Replicas", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="y=loss-after_x=loss-before_hue=replicas"
)
plt.show()


plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=sft_runs_configs_df,
    x="eval_before/eval_loss",
    y="eval_after/eval_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    # style="Parameters",
    legend="full",
)
plt.plot(
    [0, 7.5],
    [0, 7.5],
    linestyle="--",
    color="black",
)
g.set(
    xlim=(0, 7.5),
    xlabel="MATH Test Cross Entropy Before SFT",
    ylim=(0, 7.5),
    ylabel="MATH Test Cross Entropy After SFT",
)
plt.text(
    x=5.0,  # Horizontal position
    y=2.5,  # Vertical position (slightly below the y=x line)
    s="SFTing on Train Helps on Test",  # The text string
    rotation=45,  # Rotates text to match the y=x slope
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,  # Adjust based on your preference
)
plt.text(
    x=2.5,  # Horizontal position
    y=5.0,  # Vertical position (slightly below the y=x line)
    s="SFTing on Train Hurts on Test",  # The text string
    rotation=45,  # Rotates text to match the y=x slope
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=16,  # Adjust based on your preference
)
sns.move_legend(g, "upper left", title="Num. Replicas", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss-after_x=loss-before_hue=replicas_text=diagonal",
)
plt.show()


print("Finished 12_math_qwen3_pt_cross_entropy!")
