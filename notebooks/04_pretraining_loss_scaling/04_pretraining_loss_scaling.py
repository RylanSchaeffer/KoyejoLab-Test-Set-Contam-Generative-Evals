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

# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "66oov9k1",  # Qwen 3   34M     Finished.
    # "",  # Qwen 3   48M
    "2cdw9bff",  # Qwen 3   62M     Running.
    # "",  # Qwen 3   93M
    "t6opvpa3",  # Qwen 3  153M     Running.
    "ro4lmq0x",  # Qwen 3  342M     Running.
]

# TODO: Check whether there are NaNs in the data, and if so, debug why.
pt_run_configs_df: pd.DataFrame = src.analyze.download_wandb_pretraining_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)


metric_columns = ["benchmark_loss", "eval_loss", "train_loss"]
metric_column_to_nice_string_dict = {
    "benchmark_loss": "Loss (Benchmark)",
    "eval_loss": "Loss (Eval)",
    "train_loss": "Loss (Train)",
}

for metric_column in metric_columns:
    plt.close()
    g = sns.relplot(
        data=pt_run_configs_df,
        kind="line",
        x="FLOP (6ND)",
        y=metric_column,
        col="Benchmark Subset Fraction",
        row="Num. Replicas",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
    )
    g.set(
        xscale="log",
        yscale="log",
        ylabel=metric_column_to_nice_string_dict[metric_column],
    )
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y={metric_column}_x=flop_row=num_replicas_col=subset_fraction",
    )
    # plt.show()

print("Finished 04_pretraining_loss_scaling.py!")
