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


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "qfjs0h5u",  # Qwen 2.5 3B.
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

plt.close()
g = sns.relplot(
    data=run_configs_df,
    kind="line",
    x="Num. Train Epochs",
    y="math_verify_mean",
    col="temperature",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="cool",
    legend=False,
)
g.set(
    xlim=(0.0, 100.0),
    ylim=(1.0 / 5000, 1.05),
    xscale="symlog",
    yscale="log",
    xlabel="Num. Train Epochs",
    ylabel="Exact Match",
)
g.set_titles(col_template="Temperature: {col_name}")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_mean_x=num_train_epochs_col=temp_hue=params",
)
plt.show()

print("Finished 01_minerva_math")
