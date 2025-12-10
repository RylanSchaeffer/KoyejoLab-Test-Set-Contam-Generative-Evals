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
        "eval_after/eval_eval_loss",
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
        "eval_after/eval_eval_loss",
        "eval_after/eval_benchmark_loss",
    ],
    var_name="Data",
    value_name="Cross Entropy",
)
pretrain_run_configs_melted_df["Data"] = pretrain_run_configs_melted_df["Data"].map(
    {
        "eval_after/eval_eval_loss": "FineWebEdu",
        "eval_after/eval_benchmark_loss": "MATH",
    },
)

plt.close()
g = sns.relplot(
    data=pretrain_run_configs_melted_df,
    kind="scatter",
    x="FLOP (6ND)",
    y="Cross Entropy",
    row="Data",
    row_order=["FineWebEdu", "MATH"],
    col="Num. MATH Test Set Replicas",
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
    y="Cross Entropy",
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
    data=pretrain_run_configs_melted_df,
    kind="scatter",
    x="FLOP (6ND)",
    y="Cross Entropy",
    row="Data",
    row_order=["FineWebEdu", "MATH"],
    col="Num. MATH Test Set Replicas",
    style="Overtrain Multiplier",
    hue="Num. Parameters",
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
    y="Cross Entropy",
    hue="Num. Parameters",
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
    plot_filename="y=loss_x=compute_hue=ot_row=data_col=num-replicas_lines=params",
)
# plt.show()


# Keep only runs with Benchmark Subset Fraction == 1.0
pretrain_run_configs_df = pretrain_run_configs_df[
    pretrain_run_configs_df["Benchmark Subset Fraction"] == 1.0
].copy()

plt.close()
g = sns.relplot(
    data=pretrain_run_configs_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="eval_after/eval_benchmark_loss",
    hue="Overtrain Multiplier",
    hue_norm=overtrain_multiplier_log_norm,
    palette="copper",
    col="Parameters",
    col_order=["34M", "63M", "93M"],  # , "153M", "344M"
    col_wrap=3,
    facet_kws={"sharex": False, "sharey": False},
    marker="o",
    markeredgecolor="none",
    legend="full",
    height=5,
)
g.set(
    # xscale="symlog",
    # xlim=(-0.1, 3500),
    yscale="log",
    ylabel="",
)
g.axes.flat[0].set_ylabel("Cross Entropy on MATH Test Set")
for ax in g.axes.flat:
    ax.set_xscale("symlog", linthresh=1e0)  # or smaller
    ax.set_xlim(-1e-1, 3500)
sns.move_legend(
    g,
    "upper left",
    bbox_to_anchor=(1.0, 1.0),
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_replicas_hue=ot_col=params",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=pretrain_run_configs_df,
    kind="line",
    x="Overtrain Multiplier",
    y="eval_after/eval_benchmark_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    marker="o",
    legend="full",
    # hue_norm=overtrain_multiplier_log_norm,
    # palette="copper",
    col="Parameters",
    col_order=["34M", "63M", "93M"],  # , "153M", "344M"
    col_wrap=3,
    facet_kws={"sharex": False, "sharey": True},
    markeredgecolor="none",
    height=5,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(
    g,
    "upper left",
    bbox_to_anchor=(1.0, 1.0),
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=ot_hue=num_replicas_col=params",
)
# plt.show()

# Subsample runs with Overtrain Multiplier == 1.0.
pretrain_runs_1xOT_configs_df = pretrain_run_configs_df[
    pretrain_run_configs_df["Overtrain Multiplier"] == 1.0
]

plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
ax = axes[0]
g = sns.lineplot(
    data=pretrain_runs_1xOT_configs_df,
    x="Num. MATH Test Set Replicas",
    y="eval_after/eval_benchmark_loss",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    marker="o",
    legend="full",
    # legend=False,
    ax=ax,
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
src.plot.format_g_legend_to_millions_and_billions(g=g)
ax = axes[1]
g = sns.lineplot(
    data=pretrain_runs_1xOT_configs_df,
    x="Num. Parameters",
    y="eval_after/eval_benchmark_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    legend="full",
    # legend=False,
    ax=ax,
)
# 1. Define which labels you want to keep (as strings)
#    Looking at your plot, these seem like good ones:
labels_to_keep = {"0", "10", "100", "1000", "3162"}
#    (You can, of course, add "1", "3", "32", etc., if you want)

# 2. Get the old legend and its contents
old_legend = g.get_legend()
all_handles = old_legend.legend_handles
all_labels = [t.get_text() for t in old_legend.get_texts()]

# 3. Filter to create new lists
new_handles = []
new_labels = []
for handle, label in zip(all_handles, all_labels):
    if label in labels_to_keep:
        new_handles.append(handle)
        new_labels.append(label)

# 4. Remove the old legend
old_legend.remove()

# 5. Add the new, filtered legend
g.legend(
    handles=new_handles,
    labels=new_labels,
    title="Num. Replicas",
    loc="lower left",
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="",
)
# g.get_legend().set_title("Num. Replicas")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_by_num_parameters_by_num_replicas",
)
# plt.show()


plt.close()
plt.figure(figsize=(10, 6))
g = sns.lineplot(
    data=pretrain_runs_1xOT_configs_df,
    x="Num. MATH Test Set Replicas",
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
    xlim=(-0.1, 3500),
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
    data=pretrain_runs_1xOT_configs_df,
    x="Num. MATH Test Set Replicas",
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
    xlim=(-0.1, 3500),
    yscale="log",
    ylabel="Cross Entropy on MATH Test Set",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.format_g_legend_in_scientific_notation(g=g)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=num_replicas_hue=num_parameters",
)
# plt.show()


print("Finished 10_gen_eval_math_qwen3_pt.py")
