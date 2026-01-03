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


pretrain_runs_1xOT_configs_df = pretrain_run_configs_df[
    (pretrain_run_configs_df["Overtrain Multiplier"] == 1)
    # Higher number of replicas don't have sufficient points to fit.
    & (pretrain_run_configs_df["Num. MATH Test Set Replicas"] < 1000)
]

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


# Visualize the power law fits.
plt.close()
plt.figure(figsize=(10, 6))
g = sns.scatterplot(
    data=pretrain_runs_1xOT_configs_df,
    x="FLOP (6ND)",
    y="eval_after/eval_benchmark_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    marker="o",
    # legend="full",
    legend=False,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="Loss on MATH Test Set",
)
x_vals = np.geomspace(
    start=pretrain_runs_1xOT_configs_df["FLOP (6ND)"].min() / 1.1,
    stop=pretrain_runs_1xOT_configs_df["FLOP (6ND)"].max() * 1.1,
    num=100,
)
for row_idx, row in power_law_fits_df.iterrows():
    yhat_vals = row["fit_param_E_0"] + row["fit_param_C_0"] * np.power(
        x_vals, -row["fit_param_alpha"]
    )
    sns.lineplot(
        x=x_vals,
        y=yhat_vals,
        ax=g,
        hue=row["Num. MATH Test Set Replicas"],
        hue_norm=num_replicas_sym_norm,
        palette="viridis",
        linestyle="--",
        # legend=False,
    )
# Add the Uncontaminated Irreducible error
sns.lineplot(
    x=x_vals,
    y=np.full_like(
        x_vals,
        fill_value=power_law_fits_df[
            power_law_fits_df["Num. MATH Test Set Replicas"] == 0
        ]["fit_param_E_0"].values[0],
    ),
    ax=g,
    hue=0,
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    linestyle="-",
    legend=False,
)
plt.text(1e18, 2.8, "Uncontaminated Irreducible Error", size="x-small")
g.legend(
    title="Num. Replicas",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=loss_x=flop_hue=num_replicas",
)
plt.show()

plt.close()
fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharex=False, sharey=False)
sns.scatterplot(
    data=power_law_fits_df,
    x="Num. MATH Test Set Replicas",
    y="fit_param_E_0",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    legend=False,
    ax=axes[0],
    s=150,
)
axes[0].set(
    ylabel=r"Irreducible Error $E$",
    ylim=(-0.5, 5),
)
sns.scatterplot(
    data=power_law_fits_df,
    x="Num. MATH Test Set Replicas",
    y="fit_param_C_0",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    legend=False,
    ax=axes[1],
    s=150,
)
axes[1].set(
    yscale="log",
    ylabel=r"Compute Prefactor $C_0$",
)
sns.scatterplot(
    data=power_law_fits_df,
    x="Num. MATH Test Set Replicas",
    y="fit_param_alpha",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    legend=False,
    ax=axes[2],
    s=150,
)
axes[2].set(
    ylim=(0.0, 1.5e0),
    ylabel=r"Compute Exponent $\alpha$",
)
axes[2].grid(which="minor", axis="y")
sns.scatterplot(
    data=power_law_fits_df,
    x="Num. MATH Test Set Replicas",
    y="fit_loss",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    legend=False,
    ax=axes[3],
    s=150,
)
axes[3].set(
    # ylim=(0.0, 1.5e0),
    yscale="log",
    ylabel="Fit Loss (Avg)",
)
for ax in axes:
    ax.set_xscale("symlog", linthresh=1e0)  # or smaller
    ax.set_xlim(-1e-1, 500)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fit-params_x=num-replicas_col=param_setting=full-fits",
)
# plt.show()


# power_law_no_contam_fit_row = power_law_fits_df[
#     power_law_fits_df["Num. MATH Test Set Replicas"] == 0
# ]
# irreducible_error = power_law_no_contam_fit_row["fit_param_E_0"].values[0]
# compute_prefactor = power_law_no_contam_fit_row["fit_param_C_0"].values[0]
# compute_exponent = power_law_no_contam_fit_row["fit_param_alpha"].values[0]
#
# pretrain_runs_1xOT_configs_df["No Contam FLOP (6ND)"] = pretrain_runs_1xOT_configs_df[
#     "eval_after/eval_benchmark_loss"
# ].apply(
#     lambda loss: src.analyze.calculate_compute_contamination_exchange_rate(
#         loss,
#         irreducible_error=irreducible_error,
#         prefactor=compute_prefactor,
#         exponent=compute_exponent,
#     )
# )
# pretrain_runs_1xOT_configs_df["Compute Exchange Rate"] = (
#     pretrain_runs_1xOT_configs_df["No Contam FLOP (6ND)"]
#     / pretrain_runs_1xOT_configs_df["FLOP (6ND)"]
# )
# pretrain_runs_1xOT_configs_df["Compute Exchange Rate"] = pretrain_runs_1xOT_configs_df[
#     "Compute Exchange Rate"
# ].apply(lambda x: np.inf if np.isnan(x) else x)

print("Finished 12_gen_eval_contamination_vs_compute.py")
