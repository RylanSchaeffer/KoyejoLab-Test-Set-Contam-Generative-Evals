import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
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

temperatures_to_display = np.round(
    [
        0.0,
        # 0.1,
        # 0.1778,
        # 0.3162,
        0.5623,
        # 0.75,
        0.93804187,
        1.0,
        1.29154967,
        1.5,
    ],
    decimals=2,
)

sweep_ids = [
    "6y9dy2ow",  # Qwen 3   34M     1xOT    Subset Fraction=1.0
    "lnrpy3ed",  # Qwen 3   34M     1xOT    Subset Fraction=1.0     More temperatures.
    "5oo55o9s",  # Qwen 3   62M     1xOT    Subset Fraction=1.0
    "10q465ij",  # Qwen 3   62M     1xOT    Subset Fraction=1.0     More temperatures.
    "q5uoy1eu",  # Qwen 3   93M     1xOT    Subset Fraction=1.0
    "f5djvfth",  # Qwen 3   93M     1xOT    Subset Fraction=1.0     More temperatures.
    "vnz1h147",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "xkzfmbhk",  # Qwen 3  153M     1xOT    Subset Fraction=1.0
    "39rugx2e",  # Qwen 3  343M     1xOT    Subset Fraction=1.0     More temperatures.
]

eval_runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

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
eval_runs_configs_df["Overtrain Multiplier"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"ot_(\d+)", model_name).group(1))
)
eval_runs_configs_df["Num. MATH Test Set Replicas"] = (
    eval_runs_configs_df["Num. Replicas Per Epoch"]
    * eval_runs_configs_df["Num. Epochs"]
)
eval_runs_configs_df["Num. Tokens"] = 20 * eval_runs_configs_df["Num. Parameters"]
eval_runs_configs_df["FLOP (6ND)"] = (
    6 * eval_runs_configs_df["Num. Parameters"] * eval_runs_configs_df["Num. Tokens"]
)
eval_runs_configs_df.rename(columns={"temperature": "Temp."}, inplace=True)
eval_runs_configs_df["Temp."] = np.round(eval_runs_configs_df["Temp."], decimals=2)

num_replicas_sym_norm = SymLogNorm(
    linthresh=1.0,
    vmin=eval_runs_configs_df["Num. MATH Test Set Replicas"].min(),
    vmax=eval_runs_configs_df["Num. MATH Test Set Replicas"].max(),
)

num_parameters_log_norm = LogNorm(
    vmin=eval_runs_configs_df["Num. Parameters"].min(),
    vmax=eval_runs_configs_df["Num. Parameters"].max(),
)

eval_runs_histories_df: pd.DataFrame = (
    src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
        filetype="parquet",
        cols_to_drop=["response", "solution", "_step", "_runtime", "_timestamp"],
    )
)

avg_math_verify_scores_by_exp_condition_df = (
    eval_runs_histories_df.groupby(["run_id"])["math_verify_score"]
    .mean()
    .reset_index()
    .merge(
        eval_runs_configs_df[
            [
                "run_id",
                "Parameters",
                "Num. Parameters",
                "Num. MATH Test Set Replicas",
                "Num. Tokens",
                "FLOP (6ND)",
                "Overtrain Multiplier",
                "Temp.",
            ]
        ],
        how="inner",
        on=["run_id"],
    )
    .drop(columns=["run_id"])
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
# Pretraining results use exact numbers of parameters. Eval results extract from the name.
pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
    "Num. Parameters"
].map(
    {
        34061856.0: 34e6,
        62873440.0: 62e6,
        153110464.0: 153e6,
        93069280.0: 93e6,
        344023616.0: 344e6,
    }
)

avg_math_verify_and_cross_entropies_by_exp_condition_df = (
    avg_math_verify_scores_by_exp_condition_df.merge(
        pretrain_run_configs_df[
            [
                "Num. Parameters",
                "Num. MATH Test Set Replicas",
                "Overtrain Multiplier",
                "eval_after/eval_benchmark_loss",
            ]
        ],
        on=[
            "Num. Parameters",
            "Num. MATH Test Set Replicas",
            "Overtrain Multiplier",
        ],
        how="outer",
    )
)

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=avg_math_verify_and_cross_entropies_by_exp_condition_df,
    x="eval_after/eval_benchmark_loss",
    y="math_verify_score",
    hue="Temp.",
    palette="YlOrBr_r",
    marker="o",
)
g.set(
    xlabel="Cross Entropy on MATH Test Set",
    xscale="log",
    xlim=(None, 1e0),
    ylabel="Math Verification Score",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=cross_entropy_hue=temp",
)
# plt.show()


plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=avg_math_verify_and_cross_entropies_by_exp_condition_df,
    x="math_verify_score",
    y="eval_after/eval_benchmark_loss",
    hue="Temp.",
    palette="YlOrBr_r",
    marker="o",
)
g.set(
    # xscale="log",
    xlabel="Math Verification Score",
    ylabel="Cross Entropy on MATH Test Set",
    yscale="log",
    ylim=(None, 1e0),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cross_entropy_x=math_verify_hue=temp",
)
plt.show()


plt.close()
g = sns.relplot(
    data=avg_math_verify_and_cross_entropies_by_exp_condition_df,
    kind="line",
    x="eval_after/eval_benchmark_loss",
    y="math_verify_score",
    hue="Temp.",
    col="Num. Parameters",
    palette="YlOrBr_r",
    marker="o",
)
g.set(
    xlabel="Cross Entropy on MATH Test Set",
    xscale="log",
    ylabel="Math Verification Score",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=cross_entropy_hue=temp_col=params",
)
plt.show()


extended_eval_runs_histories_df = eval_runs_histories_df.merge(
    eval_runs_configs_df[
        [
            "run_id",
            "Num. Parameters",
            "Num. MATH Test Set Replicas",
            "Num. Tokens",
            "FLOP (6ND)",
            "Overtrain Multiplier",
            "Temp.",
        ]
    ],
    how="outer",
    on=["run_id"],
)


extended_eval_runs_histories_ot_1_temp_0_df = extended_eval_runs_histories_df[
    (extended_eval_runs_histories_df["Temp."] == 0.0)
    & (extended_eval_runs_histories_df["Overtrain Multiplier"] == 1)
]

plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
ax = axes[0]
g = sns.lineplot(
    data=extended_eval_runs_histories_ot_1_temp_0_df,
    x="Num. MATH Test Set Replicas",
    y="math_verify_score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    marker="o",
    # legend="full",
    legend=False,
    ax=ax,
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    yscale="log",
    ylim=(1e-3, 1.05),
    ylabel="Math Verify Score",
)
sm_left = ScalarMappable(cmap="flare", norm=num_parameters_log_norm)
sm_left.set_array([])
cbarL = fig.colorbar(
    sm_left, ax=axes[0], label="Num. Parameters", fraction=0.05, pad=0.02
)

ax = axes[1]
g = sns.lineplot(
    data=extended_eval_runs_histories_ot_1_temp_0_df,
    x="Num. Parameters",
    y="math_verify_score",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    legend=False,
    ax=ax,
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(1e-3, 1.05),
    ylabel="",
)
sm = ScalarMappable(cmap="viridis", norm=num_replicas_sym_norm)
sm.set_array([])
fig.colorbar(
    sm, ax=axes[1], label="Num. MATH Test Set Replicas", fraction=0.05, pad=0.02
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_by_num_parameters_by_num_replicas",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=extended_eval_runs_histories_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="math_verify_score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Temp.",
    col_wrap=3,
    col_order=temperatures_to_display,
    marker="o",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylabel="Math Verify Score",
    yscale="log",
    ylim=(1e-3, 1.05),
)
src.plot.format_g_legend_in_scientific_notation(g=g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=compute_col=temp",
)
plt.show()

plt.close()
g = sns.relplot(
    data=extended_eval_runs_histories_df,
    kind="line",
    x="FLOP (6ND)",
    y="math_verify_score",
    hue="Num. MATH Test Set Replicas",
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

# for (temperature,), math_verify_by_temp_df in extended_eval_runs_histories_df.groupby(
#     ["Temp."]
# ):
#     plt.close()
#     plt.figure(figsize=(10, 6))
#     g = sns.lineplot(
#         data=math_verify_by_temp_df,
#         x="FLOP (6ND)",
#         y="math_verify_score",
#         hue="Num. MATH Test Set Replicas",
#         hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
#         palette="viridis",
#         style="Temp.",
#         legend="full",
#     )
#     g.set(
#         xscale="log",
#         ylabel="Math Verify Score",
#         yscale="log",
#         ylim=(None, 1.05),
#     )
#     sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
#     src.plot.save_plot_with_multiple_extensions(
#         plot_dir=results_dir,
#         plot_filename=f"y=math_verify_x=compute_hue=num_replicas_style=temp={temperature}",
#     )
#     # plt.show()


print("Finished 11_gen_eval_math_qwen3_pt_eval.py")
