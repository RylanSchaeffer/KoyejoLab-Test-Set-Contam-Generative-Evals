import ast

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
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
    "rkx5xfde",  # Qwen 3  34M
    "g31f7bsb",  # Qwen 3  34M
    "u7dxxphm",  # Qwen 3  62M
    "o6aoejzc",  # Qwen 3  62M
    "ho49sshi",  # Qwen 3  93M
    "x8gmmzlo",  # Qwen 3  93M
    "sl086kx0",  # Qwen 3 153M
    "09c432gh",  # Qwen 3 344M
    "gsx7gisg",  # Qwen 3 344M
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)
pretrain_run_configs_df["Model"] = pretrain_run_configs_df["model_config"].apply(
    lambda model_config: ast.literal_eval(model_config)["model_name"]
)
pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
    "model/num_parameters"
]
pretrain_run_configs_df["Benchmark Subset Fraction"] = pretrain_run_configs_df[
    "data_config"
].apply(lambda data_config: ast.literal_eval(data_config)["benchmark_subset_fraction"])
pretrain_run_configs_df["Overtrain Multiplier"] = pretrain_run_configs_df[
    "trainer_config"
].apply(lambda trainer_config: ast.literal_eval(trainer_config)["overtrain_multiplier"])
pretrain_run_configs_df["Num. Tokens"] = (
    20.0
    * pretrain_run_configs_df["Overtrain Multiplier"]
    * pretrain_run_configs_df["Num. Parameters"]
)
pretrain_run_configs_df["FLOP (6ND)"] = (
    6
    * pretrain_run_configs_df["Num. Parameters"]
    * pretrain_run_configs_df["Num. Tokens"]
)
pretrain_run_configs_df["Num. Replicas Per Epoch"] = pretrain_run_configs_df[
    "data_config"
].apply(
    lambda data_config: ast.literal_eval(data_config)[
        "num_benchmark_replicas_per_epoch"
    ]
)
pretrain_run_configs_df["Num. Epochs"] = pretrain_run_configs_df[
    "trainer_config"
].apply(lambda trainer_config: ast.literal_eval(trainer_config)["num_train_epochs"])
pretrain_run_configs_df["Num. Replicas"] = (
    pretrain_run_configs_df["Num. Replicas Per Epoch"]
    * pretrain_run_configs_df["Num. Epochs"]
)

# Keep only runs with Benchmark Subset Fraction == 1.0
pretrain_run_configs_df = pretrain_run_configs_df[
    pretrain_run_configs_df["Benchmark Subset Fraction"] == 1.0
].copy()

pretrain_run_1xOT_configs_df = pretrain_run_configs_df[
    pretrain_run_configs_df["Overtrain Multiplier"] == 1.0
]

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_run_1xOT_configs_df,
    x="Num. Replicas",
    y="eval_after/eval_benchmark_loss",
    hue="FLOP (6ND)",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="cool",
    # col="Overtrain Multiplier",
    marker="o",
    legend="full",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3162),
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.format_g_legend_in_scientific_notation(g=g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_replicas_hue=flop",
)
# plt.show()

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_run_1xOT_configs_df,
    x="Num. Replicas",
    y="eval_after/eval_benchmark_loss",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="flare",
    # col="Overtrain Multiplier",
    marker="o",
    legend="full",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3162),
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.format_g_legend_in_scientific_notation(g=g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_replicas_hue=num_parameters",
)
plt.show()

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_run_1xOT_configs_df,
    x="FLOP (6ND)",
    y="eval_after/eval_benchmark_loss",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    marker="o",
    legend="full",
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.format_g_legend_in_scientific_notation(g=g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=flop_hue=num_replicas",
)
# plt.show()

plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_run_1xOT_configs_df,
    x="Num. Parameters",
    y="eval_after/eval_benchmark_loss",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    marker="o",
    legend="full",
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.format_g_legend_in_scientific_notation(g=g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_parameters_hue=num_replicas",
)
# plt.show()


print("Finished 10_gen_eval_math_qwen3_pt.py")
