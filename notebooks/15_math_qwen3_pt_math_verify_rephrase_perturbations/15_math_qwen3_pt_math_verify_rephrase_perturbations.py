import ast
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
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


def load_and_prepare_sweep_data(sweep_ids, condition_label):
    """Load eval runs from W&B sweeps and compute mean math_verify_score per run."""
    configs_df = src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username="rylan",
        finished_only=True,
    )

    configs_df["Model"] = configs_df["model_config"].apply(
        lambda model_config: ast.literal_eval(model_config)["model"]
    )
    configs_df["Parameters"] = configs_df["Model"].apply(
        lambda model_name: re.search(r"Qwen3-([\d.]+[MB])", model_name).group(1)
    )
    configs_df["Num. Parameters"] = configs_df["Parameters"].apply(
        lambda parameters: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[parameters]
    )
    configs_df["Num. Replicas Per Epoch"] = configs_df["Model"].apply(
        lambda model_name: int(re.search(r"rep_(\d+)_sbst", model_name).group(1))
    )
    configs_df["Num. Epochs"] = configs_df["Model"].apply(
        lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
    )
    configs_df["Num. MATH Test Set Replicas"] = (
        configs_df["Num. Replicas Per Epoch"] * configs_df["Num. Epochs"]
    )
    configs_df.rename(columns={"temperature": "Temp."}, inplace=True)
    configs_df["Temp."] = np.round(configs_df["Temp."], decimals=2)

    histories_df = src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
        filetype="parquet",
        cols_to_drop=["response", "solution", "_step", "_runtime", "_timestamp"],
    )

    # Average math_verify_score per run, then merge with configs.
    avg_scores_df = (
        histories_df.groupby("run_id")["math_verify_score"]
        .mean()
        .reset_index()
        .merge(
            configs_df[
                [
                    "run_id",
                    "Parameters",
                    "Num. Parameters",
                    "Num. MATH Test Set Replicas",
                    "Temp.",
                ]
            ],
            how="inner",
            on="run_id",
        )
        .drop(columns=["run_id"])
    )
    avg_scores_df["Condition"] = condition_label
    return avg_scores_df


# Load original MATH eval (344M only, temp=0 only from P1 sweep).
original_df = load_and_prepare_sweep_data(
    sweep_ids=["mprek7pj"],  # Qwen 3 344M 1xOT 4-shot
    condition_label="Original",
)
# Filter to temp=0 to match perturbed/rephrased (which only have temp=0).
original_df = original_df[original_df["Temp."] == 0.0]

# Load perturbed and rephrased eval data.
perturbed_df = load_and_prepare_sweep_data(
    sweep_ids=["w8j3qnru"],  # Qwen 3 344M 1xOT perturbed 4-shot
    condition_label="Perturbed",
)

rephrased_df = load_and_prepare_sweep_data(
    sweep_ids=["25xeednq"],  # Qwen 3 344M 1xOT rephrased 4-shot
    condition_label="Rephrased",
)

# Combine all conditions.
combined_df = pd.concat([original_df, perturbed_df, rephrased_df], ignore_index=True)

# Rename for plotting.
combined_df.rename(
    columns={"math_verify_score": "Math Verify Score"}, inplace=True
)

num_parameters_log_norm = LogNorm(
    vmin=combined_df["Num. Parameters"].min(),
    vmax=combined_df["Num. Parameters"].max(),
)


def _format_facetgrid_legend(g):
    """Format numeric legend labels to M/B suffix on a FacetGrid."""
    for txt in g._legend.texts:
        try:
            num = float(txt.get_text())
            if 1e6 <= num < 1e9:
                txt.set_text(f"{int(num / 1e6)}M")
            elif num >= 1e9:
                txt.set_text(f"{int(num / 1e9)}B")
        except ValueError:
            pass


# Plot 1: All three conditions (Original, Rephrased, Perturbed).
plt.close()
g = sns.relplot(
    data=combined_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=["Original", "Rephrased", "Perturbed"],
    marker="o",
    legend="full",
    facet_kws={"sharey": True},
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylim=(-0.05, 1.05),
)
_format_facetgrid_legend(g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=num_params_col=condition",
)
plt.close()

# Plot 2: Only Rephrased and Perturbed conditions.
rephrase_perturbed_df = combined_df[
    combined_df["Condition"].isin(["Rephrased", "Perturbed"])
]

plt.close()
g = sns.relplot(
    data=rephrase_perturbed_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=["Rephrased", "Perturbed"],
    marker="o",
    legend="full",
    facet_kws={"sharey": True},
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylim=(-0.05, 1.05),
)
_format_facetgrid_legend(g)
sns.move_legend(g, loc="upper right", bbox_to_anchor=(0.98, 0.88), frameon=True)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=num_params_col=condition_rephrase_perturbed",
)
plt.close()

# Plot 3: Same as Plot 2 but with col_wrap=1 for single-column layout.
plt.close()
g = sns.relplot(
    data=rephrase_perturbed_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=["Rephrased", "Perturbed"],
    col_wrap=1,
    marker="o",
    legend="full",
    facet_kws={"sharey": True},
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylim=(-0.05, 1.05),
)
_format_facetgrid_legend(g)
sns.move_legend(g, loc="upper right", bbox_to_anchor=(0.95, 0.45), frameon=True)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=num_params_col=condition_rephrase_perturbed_col_wrap=1",
)
plt.close()

print("Finished 15_math_qwen3_pt_math_verify_rephrase_perturbations.py")
