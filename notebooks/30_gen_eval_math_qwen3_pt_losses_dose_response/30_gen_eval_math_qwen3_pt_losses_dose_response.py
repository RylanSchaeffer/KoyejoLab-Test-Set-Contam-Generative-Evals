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

# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "rkx5xfde",  # Qwen 3  34M
    "g31f7bsb",  # Qwen 3  34M
    "ehxxzk5n",  # Qwen 3  34M
    "u7dxxphm",  # Qwen 3  62M
    "o6aoejzc",  # Qwen 3  62M
    "1nwyun1z",  # Qwen 3  62M
    "xbiu535y",  # Qwen 3  62M
    "ho49sshi",  # Qwen 3  93M
    "x8gmmzlo",  # Qwen 3  93M
    "u5xcf726",  # Qwen 3  93M
    "sl086kx0",  # Qwen 3 153M
    "09c432gh",  # Qwen 3 344M
    "09c432gh",  # Qwen 3 344M
    "gsx7gisg",  # Qwen 3 344M
    "6f9ah90l",  # Qwen 3 344M
    "r9fixoce",  # Qwen 3 344M
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

# Extract basic quantities.
pretrain_run_configs_df["Model"] = pretrain_run_configs_df["model_config"].apply(
    lambda model_config: ast.literal_eval(model_config)["model_name"]
)
pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
    "model/num_parameters"
]
pretrain_run_configs_df["Parameters"] = pretrain_run_configs_df[
    "Num. Parameters"
].apply(lambda n: f"{n / 1_000_000:.0f}M")
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
pretrain_run_configs_df["Num. MATH Test Set Replicas"] = (
    pretrain_run_configs_df["Num. Replicas Per Epoch"]
    * pretrain_run_configs_df["Num. Epochs"]
)

num_replicas_sym_norm = SymLogNorm(
    linthresh=1.0,
    vmin=pretrain_run_configs_df["Num. MATH Test Set Replicas"].min(),
    vmax=pretrain_run_configs_df["Num. MATH Test Set Replicas"].max(),
)

num_parameters_log_norm = LogNorm(
    vmin=pretrain_run_configs_df["Num. Parameters"].min(),
    vmax=pretrain_run_configs_df["Num. Parameters"].max(),
)

overtrain_multiplier_log_norm = LogNorm(
    vmin=pretrain_run_configs_df["Overtrain Multiplier"].min(),
    vmax=pretrain_run_configs_df["Overtrain Multiplier"].max(),
)

# Gadre et al. (2024) Figure 2. https://arxiv.org/pdf/2403.08540
pretrain_run_configs_melted_df = pretrain_run_configs_df[
    [
        "FLOP (6ND)",
        # "eval_after/eval_eval_loss",
        "eval_after/eval_benchmark_loss",
        "Overtrain Multiplier",
        "Parameters",
        "Num. Parameters",
        "Num. MATH Test Set Replicas",
    ]
].melt(
    id_vars=[
        "FLOP (6ND)",
        "Overtrain Multiplier",
        "Parameters",
        "Num. Parameters",
        "Num. MATH Test Set Replicas",
    ],
    value_vars=[
        # "eval_after/eval_eval_loss",
        "eval_after/eval_benchmark_loss",
    ],
    var_name="Data",
    value_name="Cross Entropy on MATH Test Set",
)
pretrain_run_configs_melted_df["Data"] = pretrain_run_configs_melted_df["Data"].map(
    {
        # "eval_after/eval_eval_loss": "FineWebEdu",
        "eval_after/eval_benchmark_loss": "MATH",
    },
)

plt.close()
g = sns.relplot(
    data=pretrain_run_configs_melted_df,
    kind="scatter",
    x="FLOP (6ND)",
    y="Cross Entropy on MATH Test Set",
    col="Num. MATH Test Set Replicas",
    col_order=[0, 1, 10, 32, 100, 316],
    col_wrap=3,
    style="Parameters",
    style_order=["34M", "63M", "93M", "153M", "344M"],
    hue="Overtrain Multiplier",
    hue_norm=LogNorm(),
    palette="copper",
    facet_kws={"margin_titles": True, "sharey": "row"},
    legend="full",
    s=100,
    linewidth=0,
    height=6,
    aspect=0.75,
)
g.map_dataframe(
    sns.lineplot,
    x="FLOP (6ND)",
    y="Cross Entropy on MATH Test Set",
    hue="Overtrain Multiplier",
    hue_norm=LogNorm(),
    palette="copper",
    legend=False,  # keep axes clean
)
g.set(xscale="log", yscale="log")
g.set_titles(
    row_template="{row_name} Test Set", col_template="{col_name} MATH Test Set Replicas"
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=compute_hue=ot_row=data_col=num-replicas_lines=ot",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=pretrain_run_configs_melted_df[
        pretrain_run_configs_melted_df["Data"] == "MATH"
    ],
    kind="scatter",
    x="FLOP (6ND)",
    y="Cross Entropy on MATH Test Set",
    col="Num. MATH Test Set Replicas",
    col_order=[0, 1, 10, 32, 100, 316],
    col_wrap=3,
    style="Overtrain Multiplier",
    hue="Parameters",
    hue_order=["34M", "63M", "93M", "153M", "344M"],
    # hue_norm=LogNorm(),
    palette="flare",
    facet_kws={"margin_titles": True, "sharey": "row"},
    legend="full",
    s=100,
    linewidth=0,
    height=6,
    aspect=0.75,
)
g.map_dataframe(
    sns.lineplot,
    x="FLOP (6ND)",
    y="Cross Entropy on MATH Test Set",
    hue="Parameters",
    hue_order=["34M", "63M", "93M", "153M", "344M"],
    # hue_norm=LogNorm(),
    palette="flare",
    legend=False,  # keep axes clean
)
g.set(xscale="log", yscale="log")
g.set_titles(
    row_template="{row_name} Test Set", col_template="{col_name} MATH Test Set Replicas"
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=compute_hue=ot_row=data_col=num-replicas_lines=params",
)
plt.show()

print("Finished 20_gen_eval_math_qwen3_pt_losses_dose_response")
