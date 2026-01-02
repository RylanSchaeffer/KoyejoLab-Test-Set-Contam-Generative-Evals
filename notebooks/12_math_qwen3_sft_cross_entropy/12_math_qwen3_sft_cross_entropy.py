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
