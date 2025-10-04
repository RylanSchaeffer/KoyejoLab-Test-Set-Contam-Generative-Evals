import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import math
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot

# refresh = True
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Not displacing any data
# sweep_ids = [
#     "cqbazyfu",
#     "c1ncnak3",
#     "0h5q5wh1"  
# ]

# displacing data and following Hernandez
# sweep_ids = [
#     "ctv46x81"
# ]

# displacing data on larger model
sweep_ids = [
    "0afaovk5"
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username="jkazdan",#wandb.api.default_entity,
    finished_only=False,
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

print(pretrain_run_configs_df.head()['eval_after/eval_eval_loss'])

plt.scatter(pretrain_run_configs_df['Benchmark Subset Fraction'], pretrain_run_configs_df['eval_after/eval_eval_loss'])
plt.xlabel('Fraction of Dataset Replicated')
plt.ylabel('Eval Loss')
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="repeats_v_eval_loss.png",
)
plt.close()
# # Gadre et al. (2024) Figure 2. https://arxiv.org/pdf/2403.08540
# pretrain_runs_no_contam_configs_df = pretrain_run_configs_df[
#     pretrain_run_configs_df["Num. MATH Test Set Replicas"] == 0.0
# ]
# pretrain_runs_no_contam_configs_melted_df = pretrain_runs_no_contam_configs_df[
#     [
#         "FLOP (6ND)",
#         "eval_after/eval_eval_loss",
#         "eval_after/eval_benchmark_loss",
#         "Overtrain Multiplier",
#         "Parameters",
#     ]
# ].melt(
#     id_vars=["FLOP (6ND)", "Overtrain Multiplier", "Parameters"],
#     value_vars=[
#         "eval_after/eval_eval_loss",
#         "eval_after/eval_benchmark_loss",
#     ],
#     var_name="Data",
#     value_name="Cross Entropy",
# )
# pretrain_runs_no_contam_configs_melted_df[
#     "Data"
# ] = pretrain_runs_no_contam_configs_melted_df["Data"].map(
#     {
#         "eval_after/eval_eval_loss": "FineWebEdu",
#         "eval_after/eval_benchmark_loss": "MATH",
#     },
# )

# print(pretrain_run_configs.head())
# plt.close()
# g = sns.relplot(
#     data=pretrain_runs_no_contam_configs_melted_df,
#     kind="scatter",
#     x="FLOP (6ND)",
#     y="Cross Entropy",
#     col="Data",
#     col_order=["FineWebEdu", "MATH"],
#     style="Parameters",
#     style_order=["34M", "63M", "93M", "153M", "344M"],
#     hue="Overtrain Multiplier",
#     hue_norm=LogNorm(),
#     palette="copper",
#     facet_kws={"margin_titles": True},
#     legend="full",
#     s=100,
#     linewidth=0,
#     height=6,
#     aspect=0.75,
# )
# g.map_dataframe(
#     sns.lineplot,
#     x="FLOP (6ND)",
#     y="Cross Entropy",
#     hue="Overtrain Multiplier",
#     hue_norm=LogNorm(),
#     palette="copper",
#     legend=False,  # keep axes clean
# )
# g.set(xscale="log", yscale="log")
# g.set_titles(col_template="{col_name} Test Set")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=compute_hue=ot_col=data_setting=no-contam",
# )
# plt.show()


# # Keep only runs with Benchmark Subset Fraction == 1.0
# pretrain_run_configs_df = pretrain_run_configs_df[
#     pretrain_run_configs_df["Benchmark Subset Fraction"] == 1.0
# ].copy()

# plt.close()
# g = sns.relplot(
#     data=pretrain_run_configs_df,
#     kind="line",
#     x="Num. MATH Test Set Replicas",
#     y="eval_after/eval_benchmark_loss",
#     hue="Overtrain Multiplier",
#     hue_norm=overtrain_multiplier_log_norm,
#     palette="copper",
#     col="Parameters",
#     col_order=["34M", "63M", "93M"],  # , "153M", "344M"],
#     col_wrap=3,
#     facet_kws={"sharex": False},
#     marker="o",
# )
# g.set(
#     xscale="symlog",
#     xlim=(-0.1, 3500),
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=num_replicas_hue=ot_col=params",
# )
# plt.show()


# # Subsample runs with Overtrain Multiplier == 1.0.
# pretrain_run_1xOT_configs_df = pretrain_run_configs_df[
#     pretrain_run_configs_df["Overtrain Multiplier"] == 1.0
# ]

# plt.close()
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
# ax = axes[0]
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="Num. MATH Test Set Replicas",
#     y="eval_after/eval_benchmark_loss",
#     hue="Num. Parameters",
#     hue_norm=num_parameters_log_norm,
#     palette="flare",
#     marker="o",
#     # legend="full",
#     legend=False,
#     ax=ax,
# )
# g.set(
#     xscale="symlog",
#     xlim=(-0.1, 3500),
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sm_left = ScalarMappable(cmap="flare", norm=num_parameters_log_norm)
# sm_left.set_array([])
# cbarL = fig.colorbar(
#     sm_left, ax=axes[0], label="Num. Parameters", fraction=0.05, pad=0.02
# )

# ax = axes[1]
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="Num. Parameters",
#     y="eval_after/eval_benchmark_loss",
#     hue="Num. MATH Test Set Replicas",
#     hue_norm=num_replicas_sym_norm,
#     palette="viridis",
#     marker="o",
#     legend=False,
#     ax=ax,
# )
# g.set(
#     xscale="log",
#     yscale="log",
#     ylabel="",
# )
# sm = ScalarMappable(cmap="viridis", norm=num_replicas_sym_norm)
# sm.set_array([])
# fig.colorbar(
#     sm, ax=axes[1], label="Num. MATH Test Set Replicas", fraction=0.05, pad=0.02
# )
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_by_num_parameters_by_num_replicas",
# )
# plt.show()


# plt.close()
# plt.figure(figsize=(10, 6))
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="Num. Parameters",
#     y="eval_after/eval_benchmark_loss",
#     hue="Num. MATH Test Set Replicas",
#     hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
#     palette="viridis",
#     marker="o",
#     legend="full",
# )
# g.set(
#     xscale="log",
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# # src.plot.format_g_legend_in_scientific_notation(g=g)
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=num_parameters_hue=num_replicas",
# )
# # plt.show()

# plt.close()
# plt.figure(figsize=(10, 6))
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="Num. MATH Test Set Replicas",
#     y="eval_after/eval_benchmark_loss",
#     hue="FLOP (6ND)",
#     hue_norm=matplotlib.colors.LogNorm(),
#     palette="cool",
#     # col="Overtrain Multiplier",
#     marker="o",
#     legend="full",
# )
# g.set(
#     xscale="symlog",
#     xlim=(-0.1, 3500),
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.format_g_legend_in_scientific_notation(g=g)
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=num_replicas_hue=flop",
# )
# # plt.show()

# plt.close()
# plt.figure(figsize=(10, 6))
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="Num. MATH Test Set Replicas",
#     y="eval_after/eval_benchmark_loss",
#     hue="Num. Parameters",
#     hue_norm=matplotlib.colors.LogNorm(),
#     palette="flare",
#     # col="Overtrain Multiplier",
#     marker="o",
#     legend="full",
# )
# g.set(
#     xscale="symlog",
#     xlim=(-0.1, 3500),
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# src.plot.format_g_legend_in_scientific_notation(g=g)
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=num_replicas_hue=num_parameters",
# )
# # plt.show()

# plt.close()
# plt.figure(figsize=(10, 6))
# g = sns.lineplot(
#     data=pretrain_run_1xOT_configs_df,
#     x="FLOP (6ND)",
#     y="eval_after/eval_benchmark_loss",
#     hue="Num. MATH Test Set Replicas",
#     hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
#     palette="cool",
#     marker="o",
#     legend="full",
# )
# g.set(
#     xscale="log",
#     yscale="log",
#     ylabel="Cross Entropy on MATH Test Set",
# )
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# # src.plot.format_g_legend_in_scientific_notation(g=g)
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=loss_x=flop_hue=num_replicas",
# )
# plt.show()
# plt.close()

# # Create a heatmap of the different parameter sizes
# param_sizes = sorted(pretrain_run_configs_df['Num. Parameters'].unique()[:2])
# fig, axes = plt.subplots(nrows=1, ncols=len(param_sizes), figsize=(5*(len(param_sizes)+1), 5))
# axes = [axes] if len(param_sizes) == 1 else axes
# pretrain_run_configs_df['ce_loss_difference'] = pretrain_run_configs_df['eval_after/eval_benchmark_loss']/pretrain_run_configs_df['eval_after/eval_eval_loss']
# vmin, vmax = pretrain_run_configs_df['ce_loss_difference'].min(), pretrain_run_configs_df['ce_loss_difference'].max()
# all_overtrain, all_replicas = sorted(pretrain_run_configs_df['Overtrain Multiplier'].unique(), reverse=True), sorted(pretrain_run_configs_df['Num. Replicas Per Epoch'].unique())

# for idx, param_size in enumerate(param_sizes):
#     hm_data = pretrain_run_configs_df[pretrain_run_configs_df['Num. Parameters'] == param_size].pivot_table(
#         index='Overtrain Multiplier', columns="Num. Replicas Per Epoch", values='ce_loss_difference', aggfunc='mean'
#     ).reindex(index=all_overtrain, columns=all_replicas)
    
#     sns.heatmap(hm_data, ax=axes[idx], cbar=(idx == len(param_sizes) - 1), cbar_kws={'label': 'Eval Loss', 'aspect': 20} if idx == len(param_sizes) - 1 else None, vmin=vmin, vmax=vmax, cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax))
#     print_parameters = f"{param_size / 1_000_000:.0f}M"
#     axes[idx].set_title(f'{print_parameters} Parameters')
#     axes[idx].set(xlabel = 'Num. Replicas')
# plt.suptitle('MATH Cross Entropy/Eval Cross Entropy')
# plt.tight_layout()
# src.plot.save_plot_with_multiple_extensions(plot_dir=results_dir, plot_filename="heatmap_loss_by_overtrain_and_replicas_by_model_size")
# plt.show()
# plt.close()

# print("Finished 10_gen_eval_math_qwen3_pt.py")
