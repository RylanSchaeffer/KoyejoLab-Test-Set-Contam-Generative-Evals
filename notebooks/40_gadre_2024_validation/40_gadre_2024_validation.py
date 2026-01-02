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

pretrain_run_configs_df = (
    src.analyze.add_pretraining_quantities_to_pretrain_runs_configs_df(
        pretrain_run_configs_df=pretrain_run_configs_df
    )
)
# Alias "Num. Replicas Per Epoch" for successful merge.
pretrain_run_configs_df["Num. MATH Test Set Replicas"] = pretrain_run_configs_df[
    "Num. Replicas Per Epoch"
]

# Gadre et al. (2024) Figure 2. https://arxiv.org/pdf/2403.08540
pretrain_runs_no_contam_configs_df = pretrain_run_configs_df[
    pretrain_run_configs_df["Num. MATH Test Set Replicas"] == 0.0
]
pretrain_runs_no_contam_configs_melted_df = pretrain_runs_no_contam_configs_df[
    [
        "FLOP (6ND)",
        "eval_after/eval_eval_loss",
        "eval_after/eval_benchmark_loss",
        "Overtrain Multiplier",
        "Parameters",
        "Num. Parameters",
    ]
].melt(
    id_vars=["FLOP (6ND)", "Overtrain Multiplier", "Parameters", "Num. Parameters"],
    value_vars=[
        "eval_after/eval_eval_loss",
        "eval_after/eval_benchmark_loss",
    ],
    var_name="Data",
    value_name="Cross Entropy",
)
pretrain_runs_no_contam_configs_melted_df[
    "Data"
] = pretrain_runs_no_contam_configs_melted_df["Data"].map(
    {
        "eval_after/eval_eval_loss": "FineWebEdu",
        "eval_after/eval_benchmark_loss": "MATH",
    },
)

plt.close()
g = sns.relplot(
    data=pretrain_runs_no_contam_configs_melted_df,
    kind="scatter",
    x="FLOP (6ND)",
    y="Cross Entropy",
    col="Data",
    col_order=["FineWebEdu", "MATH"],
    style="Parameters",
    style_order=["34M", "63M", "93M", "153M", "344M"],
    hue="Overtrain Multiplier",
    hue_norm=LogNorm(),
    palette="copper",
    facet_kws={"margin_titles": True},
    legend="full",
    s=100,
    linewidth=0,
    height=6,
    aspect=0.75,
)
g.map_dataframe(
    sns.lineplot,
    x="FLOP (6ND)",
    y="Cross Entropy",
    hue="Overtrain Multiplier",
    hue_norm=LogNorm(),
    palette="copper",
    legend=False,  # keep axes clean
)
g.set(xscale="log", yscale="log")
g.set_titles(col_template="{col_name} Test Set")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=compute_hue=ot_col=data_lines=ot_setting=no-contam",
)
# plt.show()


# Fit power law scaling to each model's inference scaling w.r.t. k.
power_law_fits_df = pd.DataFrame(
    [
        src.analyze.fit_neural_scaling_law(
            model_df,
            x_col="FLOP (6ND)",
            y_col="eval_after/eval_benchmark_loss",
            additional_columns_to_add=[
                "Num. MATH Test Set Replicas",
            ],
        )
        for model_nickname, model_df in pretrain_runs_1xOT_configs_df.groupby(
            ["Num. MATH Test Set Replicas"]
        )
    ]
)
