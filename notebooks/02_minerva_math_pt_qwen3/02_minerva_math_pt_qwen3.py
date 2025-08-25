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
    "j8b3kqbm",  # Qwen 3 34M   Train: Finished     Eval: Finished
]

eval_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)
eval_run_configs_df["Model"] = eval_run_configs_df["model_config"].apply(
    src.analyze.extract_hf_model_name_or_path,
)
eval_run_configs_df["Num. Parameters"] = eval_run_configs_df["Model"].apply(
    src.analyze.extract_num_model_parameters
)
eval_run_configs_df["Num. Tokens"] = 20 * eval_run_configs_df["Num. Parameters"]
eval_run_configs_df["Pretraining FLOP (6ND)"] = (
    6 * eval_run_configs_df["Num. Parameters"] * eval_run_configs_df["Num. Tokens"]
)
eval_run_configs_df["Num. Replicas Per Epoch"] = eval_run_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"replicas_(\d+)_epch", model_name).group(1))
)
eval_run_configs_df["Num. Epochs"] = eval_run_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
)
eval_run_configs_df["Num. Replicas"] = (
    eval_run_configs_df["Num. Replicas Per Epoch"] * eval_run_configs_df["Num. Epochs"]
)

solution_token_length_columns = [
    col
    for col in eval_run_configs_df.columns.values
    if col.startswith("custom/solution_token_length_")
]

eval_solution_token_length_columns_df = eval_run_configs_df[
    ["Pretraining FLOP (6ND)", "Num. Parameters", "temperature", "Num. Replicas"]
    + solution_token_length_columns
].melt(
    id_vars=[
        "Pretraining FLOP (6ND)",
        "Num. Parameters",
        "temperature",
        "Num. Replicas",
    ],
    value_vars=solution_token_length_columns,
    var_name="Problem Index",
    value_name="Solution Token Length",
)
eval_solution_token_length_columns_df[
    "Problem Index"
] = eval_solution_token_length_columns_df["Problem Index"].apply(
    lambda s: int(s.split("_")[-1])
)


math_verify_columns = [
    col
    for col in eval_run_configs_df.columns.values
    if col.startswith("custom/math_verify_")
    and all(
        [
            not col.endswith(ending)
            for ending in ["mean", "max", "stddev", "min", "median"]
        ]
    )
]

eval_math_verify_columns_df = eval_run_configs_df[
    ["Pretraining FLOP (6ND)", "Num. Parameters", "temperature", "Num. Replicas"]
    + math_verify_columns
].melt(
    id_vars=[
        "Pretraining FLOP (6ND)",
        "Num. Parameters",
        "temperature",
        "Num. Replicas",
    ],
    value_vars=math_verify_columns,
    var_name="Problem Index",
    value_name="Math Verify",
)
eval_math_verify_columns_df["Problem Index"] = eval_math_verify_columns_df[
    "Problem Index"
].apply(lambda s: int(s.split("_")[-1]))

eval_math_verify_and_solution_token_length_df = (
    eval_solution_token_length_columns_df.merge(
        eval_math_verify_columns_df,
        how="inner",
        on=[
            "Pretraining FLOP (6ND)",
            "Num. Parameters",
            "temperature",
            "Num. Replicas",
            "Problem Index",
        ],
    )
)

eval_math_verify_and_solution_token_length_df[
    "Solution Token Length (Binned)"
] = pd.cut(
    eval_math_verify_and_solution_token_length_df["Solution Token Length"],
    bins=np.percentile(
        eval_math_verify_and_solution_token_length_df["Solution Token Length"],
        np.linspace(0, 100, 25),
    ),
)

eval_math_verify_and_solution_token_length_df[
    "Solution Token Length (Bin Midpoint)"
] = eval_math_verify_and_solution_token_length_df[
    "Solution Token Length (Binned)"
].apply(
    lambda x: x.mid
)

plt.close()
g = sns.relplot(
    data=eval_math_verify_and_solution_token_length_df,
    kind="line",
    x="Solution Token Length (Bin Midpoint)",
    y="Math Verify",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    col="temperature",
    row="Num. Parameters",
    facet_kws={"sharey": True, "margin_titles": True, "sharex": True},
    palette="viridis",
)
g.set(
    xscale="log",
    # yscale="log",
)
g.set_titles(col_template="Temperature: {col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_custom_mean_x=solution_token_length_bin_midpoint_hue=num_replicas_col=temp_row=params",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=eval_math_verify_columns_df,
    kind="line",
    x="Pretraining FLOP (6ND)",
    y="Math Verify",
    col="temperature",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    marker="o",
    palette="magma",
)
g.set(
    ylim=(0.001, 1.01),
    xscale="log",
    yscale="log",
    ylabel="Math Verify",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
# for t, l in zip(g.legend.texts, ["34M"]):
#     t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_custom_mean_x=flop_hue=num_replicas_col=temp",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=eval_run_configs_df,
    kind="line",
    x="Pretraining FLOP (6ND)",
    y="lm_eval_harness/math_verify_none",
    col="temperature",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    marker="o",
    palette="magma",
)
g.set(
    ylim=(0.001, 1.01),
    xscale="log",
    yscale="log",
    ylabel="Math Verify",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
# for t, l in zip(g.legend.texts, ["34M"]):
#     t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_harness_mean_x=flop_hue=num_replicas_col=temp",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=eval_math_verify_columns_df,
    kind="line",
    x="Num. Replicas",
    y="Math Verify",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    marker="o",
    palette="viridis",
)
g.set(
    xlim=(-0.05, 1000),
    ylim=(0.001, 1.01),
    xscale="symlog",
    yscale="log",
    ylabel="Math Verify",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
for t, l in zip(g.legend.texts, ["34M"]):
    t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_custom_mean_x=num_replicas_col=temp_hue=params",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=eval_run_configs_df,
    kind="line",
    x="Num. Replicas",
    y="lm_eval_harness/math_verify_none",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    marker="o",
    palette="viridis",
)
g.set(
    xlim=(-0.05, 1000),
    ylim=(0.001, 1.01),
    xscale="symlog",
    yscale="log",
    ylabel="Math Verify",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
for t, l in zip(g.legend.texts, ["34M"]):
    t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_harness_mean_x=num_replicas_col=temp_hue=params",
)
# plt.show()

sweep_ids = [
    "6958x53a",  # Qwen 3 34M   Train: Running
    "gws5uy57",  # Qwen 3 93M   Train: Running
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)
pretrain_run_configs_df["Num. Replicas Per Epoch"] = pretrain_run_configs_df[
    "data_config"
].apply(lambda x: ast.literal_eval(x)["num_benchmark_replicas_per_epoch"])
pretrain_run_configs_df["Num. Epochs"] = pretrain_run_configs_df[
    "trainer_config"
].apply(lambda x: ast.literal_eval(x)["num_train_epochs"])
pretrain_run_configs_df["Num. Replicas"] = (
    pretrain_run_configs_df["Num. Replicas Per Epoch"]
    * pretrain_run_configs_df["Num. Epochs"]
)
pretrain_run_configs_df["Num. Parameters"] = pretrain_run_configs_df[
    "model/num_parameters"
]
pretrain_run_configs_df["Num. Tokens"] = (
    20
    * pretrain_run_configs_df["Num. Parameters"]
    * pretrain_run_configs_df["trainer_config"].apply(
        lambda x: ast.literal_eval(x)["overtrain_multiplier"]
    )
)
pretrain_run_configs_df["Pretraining FLOP (6ND)"] = (
    6
    * pretrain_run_configs_df["Num. Parameters"]
    * pretrain_run_configs_df["Num. Tokens"]
)

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=pretrain_run_configs_df,
    x="Pretraining FLOP (6ND)",
    y="eval/loss",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1),
    palette="viridis",
    marker="o",
)
g.set(xscale="log", yscale="log", ylabel="Cross Entropy of MATH")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=flop_hue=num_replicas",
)
# plt.show()

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=pretrain_run_configs_df,
    x="Num. Replicas",
    y="eval/loss",
    hue="Pretraining FLOP (6ND)",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="magma",
    legend=False,
    marker="o",
)

g.set(
    xscale="symlog",
    xlim=(-0.1, 1000),
    yscale="log",
    ylabel="Cross Entropy on MATH",
)

# Move the legend into a colorbar and apply scientific notation
norm = matplotlib.colors.LogNorm(
    vmin=pretrain_run_configs_df["Pretraining FLOP (6ND)"].min(),
    vmax=pretrain_run_configs_df["Pretraining FLOP (6ND)"].max(),
)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
cbar = g.get_figure().colorbar(sm, ax=g)
cbar.set_label("Pretraining FLOP (6ND)")
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.get_offset_text().set_fontsize(10)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_replicas_hue=flop",
)
# plt.show()


print("Finished 02_minerva_math_pt_qwen3.py")
