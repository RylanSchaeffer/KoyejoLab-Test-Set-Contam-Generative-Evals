import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
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
    "kmx1c22e",  # Qwen 2.5 0.5B    Train: Finished     Eval: Finished
    "0bvui2wk",  # Qwen 2.5 1.5B    Train: Finished     Eval: Running
    "2rf538jp",  # Qwen 2.5 3B      Train: Finished     Eval: Finished
    "eoegnwb8",  # Qwen 2.5 7B      Train: Running      Eval:
]

run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)


run_configs_df["Model"] = run_configs_df["model_config"].apply(
    src.analyze.extract_hf_model_name_or_path,
)
run_configs_df["Num. Parameters"] = run_configs_df["Model"].apply(
    src.analyze.extract_num_model_parameters
)
run_configs_df["Num. Train Epochs"] = run_configs_df["Model"].apply(
    src.analyze.extract_num_train_epochs
)

palette = "cool"
new_labels = ["0.5B", "1.5B", "3B", "7B"]

plt.close()
g = sns.relplot(
    data=run_configs_df,
    kind="line",
    x="Num. Train Epochs",
    y="lm_eval_harness/math_verify_none",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    # markers=True,
    # dashes=False,
    # style="Num. Parameters",
    marker="o",
    # linestyle="-",
    palette=palette,
    # legend=False,
)
g.set(
    xlim=(0.0, 101.0),
    ylim=(0.0, 1.01),
    xscale="symlog",
    # yscale="log",
    xlabel="Num. Train Epochs",
    ylabel="Exact Match",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_harness_mean_x=num_train_epochs_col=temp_hue=params",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=run_configs_df,
    kind="line",
    x="Num. Train Epochs",
    y="custom/math_verify_mean",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    marker="o",
    palette=palette,
    # legend=False,
)
g.set(
    xlim=(0.0, 101.0),
    ylim=(0.0, 1.01),
    xscale="symlog",
    # yscale="log",
    xlabel="Num. Train Epochs",
    ylabel="Exact Match",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_custom_mean_x=num_train_epochs_col=temp_hue=params",
)
# plt.show()

plt.close()
g = sns.relplot(
    data=run_configs_df,
    kind="line",
    x="Num. Train Epochs",
    y="custom/edit_distance_mean",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    marker="o",
    palette=palette,
    # legend=False,
)
g.set(
    xlim=(0.0, 101.0),
    ylim=(0.0, None),
    xscale="symlog",
    xlabel="Num. Train Epochs",
    ylabel="Edit Distance to Gold Solution",
)
g.set_titles(col_template="Temperature: {col_name}")
# Overwrite the default numerical names in the legend.
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=edit_distance_custom_mean_x=num_train_epochs_col=temp_hue=params",
)
g.set(yscale="symlog")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=edit_distance_log_custom_mean_x=num_train_epochs_col=temp_hue=params",
)
# plt.show()

print("Finished 01_minerva_math")
