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
    "6y9dy2ow",  # Qwen 3   34M     1xOT    Subset Fraction=1.0
    "5oo55o9s",  # Qwen 3   62M     1xOT    Subset Fraction=1.0
    "q5uoy1eu",  # Qwen 3   93M     1xOT    Subset Fraction=1.0
    "vnz1h147",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    # "vnz1h147",  # Qwen 3  343M     1xOT    Subset Fraction=1.0
]

eval_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

eval_runs_configs_df["Model"] = eval_runs_configs_df["model_config"].apply(
    lambda model_config: ast.literal_eval(model_config)["model"]
)
eval_runs_configs_df["Parameters"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: re.search(r"(Qwen3-[\d.]+[MB])", model_name).group(1)
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
eval_runs_configs_df["Overtrain Multiplier"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"ot_(\d+)", model_name).group(1))
)
eval_runs_configs_df["Num. Replicas"] = (
    eval_runs_configs_df["Num. Replicas Per Epoch"]
    * eval_runs_configs_df["Num. Epochs"]
)
eval_runs_configs_df["Num. Tokens"] = 20 * eval_runs_configs_df["Num. Parameters"]
eval_runs_configs_df["FLOP (6ND)"] = (
    6 * eval_runs_configs_df["Num. Parameters"] * eval_runs_configs_df["Num. Tokens"]
)
eval_runs_configs_df.rename(columns={"temperature": "Temp."}, inplace=True)

eval_runs_histories_df: pd.DataFrame = (
    src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
        filetype="parquet",
    )
)

extended_eval_runs_histories_df = eval_runs_histories_df.merge(
    eval_runs_configs_df[
        [
            "run_id",
            "Num. Parameters",
            "Num. Replicas",
            "Num. Tokens",
            "FLOP (6ND)",
            "Overtrain Multiplier",
            "Temp.",
        ]
    ],
    how="outer",
    on=["run_id"],
)

for (temperature,), math_verify_by_temp_df in extended_eval_runs_histories_df.groupby(
    ["Temp."]
):
    plt.close()
    plt.figure(figsize=(10, 6))
    g = sns.lineplot(
        data=math_verify_by_temp_df,
        x="FLOP (6ND)",
        y="math_verify_score",
        hue="Num. Replicas",
        hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
        palette="viridis",
        style="Temp.",
        legend="full",
    )
    g.set(
        xscale="log",
        ylabel="Math Verify Score",
        yscale="log",
        ylim=(None, 1.05),
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y=math_verify_x=compute_hue=num_replicas_style=temp={temperature}",
    )
    # plt.show()

plt.close()
g = sns.relplot(
    data=extended_eval_runs_histories_df,
    kind="line",
    x="Num. Replicas",
    y="math_verify_score",
    hue="FLOP (6ND)",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="cool",
    col="Temp.",
    col_wrap=3,
)
g.set(
    xscale="symlog",
    xlim=(-0.1, None),
    ylabel="Math Verify Score",
    yscale="log",
    ylim=(None, 1.05),
)
src.plot.format_g_legend_in_scientific_notation(g=g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=compute_col=temp",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=extended_eval_runs_histories_df,
    kind="line",
    x="FLOP (6ND)",
    y="math_verify_score",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    col="Temp.",
    col_wrap=3,
    legend="full",
)
g.set(
    xscale="log",
    ylabel="Math Verify Score",
    yscale="log",
    ylim=(None, 1.05),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=compute_hue=num_replicas_col=temp",
)
# plt.show()


print("Finished 11_gen_eval_math_qwen3_pt_eval.py")
