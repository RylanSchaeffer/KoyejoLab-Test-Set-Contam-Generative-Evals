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

default_figsize = (8, 6)

# Data from Bordt et al. ICML 2025 How much can we forget about Data Contamination?
fig1a_df = pd.DataFrame(
    {
        "Parameters": ["124M", "350M", "774M", "1.6B"],
        "Num. Parameters": [124e6, 350e6, 774e6, 1558e6],
        "0x": [44.16, 44.72, 45.78, 46.90],
        "4x": [49.54, 55.69, 67.30, 75.48],
        "12x": [54.98, 69.90, 85.16, 91.04],
        "32x": [73.05, 89.20, 94.65, 95.70],
        "144x": [93.20, 95.50, 97.25, 97.55],
    }
)

melted_fig1a_df = fig1a_df.melt(
    id_vars=["Num. Parameters", "Parameters"],
    var_name="Num. Replicas",
    value_name="Average Accuracy (7 Benchmarks)",
)
melted_fig1a_df["Num. Replicas"] = melted_fig1a_df["Num. Replicas"].apply(
    lambda x: int(x.replace("x", ""))
)
melted_fig1a_df.head()

plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1a_df,
    x="Num. Replicas",
    y="Average Accuracy (7 Benchmarks)",
    hue="Parameters",
    palette="tab10",
    marker="o",
    markersize=15,
)
g.set(
    title="Parameter Scaling (Fig 1a)",
    xlabel="Num. Replicas",
    xscale="symlog",
    xlim=(-0.25, None),
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1a_y=accuracy_x=multiplier_hue=params",
)
# plt.show()

plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1a_df,
    x="Num. Parameters",
    y="Average Accuracy (7 Benchmarks)",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    marker="o",
    markersize=15,
)
g.set(
    title="Parameter Scaling (Fig 1a)",
    xscale="log",
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1a_y=accuracy_x=params_hue=multiplier",
)
# plt.show()

fig1b_df = pd.DataFrame(
    {
        r"$\times$ Chinchilla Tokens": [2, 4, 8, 15],
        "0x": [43.31, 44.52, 45.14, 46.45],
        "4x": [50.40, 50.75, 49.16, 48.51],
        "12x": [59.84, 58.10, 51.84, 47.88],
        "32x": [80.85, 78.35, 64.15, 51.20],
        "144x": [94.85, 93.65, 85.15, 67.10],
    }
)

melted_fig1b_df = fig1b_df.melt(
    id_vars=r"$\times$ Chinchilla Tokens",
    var_name="Num. Replicas",
    value_name="Average Accuracy (7 Benchmarks)",
)
melted_fig1b_df["Num. Replicas"] = melted_fig1b_df["Num. Replicas"].apply(
    lambda x: int(x.replace("x", ""))
)
melted_fig1b_df[r"$\times$ Chinchilla\\nTokens"] = melted_fig1b_df[
    r"$\times$ Chinchilla Tokens"
]

plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1b_df,
    x="Num. Replicas",
    y="Average Accuracy (7 Benchmarks)",
    hue=r"$\times$ Chinchilla\\nTokens",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="tab10",
    marker="o",
    markersize=15,
)
g.set(
    title="Data Scaling (Fig 1b)",
    # xlabel="Num. Replicas",
    xscale="symlog",
    xlim=(-0.25, None),
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
g.legend_.set_title(r"$\times$\\Chinchilla\\Tokens")
g.legend_.get_title().set_multialignment("left")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1b_y=accuracy_x=multiplier_hue=overtrain",
)
# plt.show()

plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1b_df,
    x=r"$\times$ Chinchilla Tokens",
    y="Average Accuracy (7 Benchmarks)",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    palette="viridis",
    marker="o",
    markersize=15,
)
g.set(
    title="Data Scaling (Fig 1b)",
    # xlabel=r"$\times$ Chinchilla Tokens",
    xlim=(0, 16),
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1b_y=accuracy_x=overtrain_hue=multiplier",
)
# plt.show()

fig1c_df = pd.DataFrame(
    {
        "Parameters": ["124M", "350M", "774M", "1.6B"],
        "Num. Parameters": [124e6, 350e6, 774e6, 1558e6],
        "0x": [42.22, 44.72, 49.16, 52.06],
        "4x": [48.14, 55.69, 64.76, 67.61],
        "12x": [56.92, 69.90, 81.30, 82.32],
        "32x": [80.70, 89.20, 92.95, 91.85],
        "144x": [96.45, 95.50, 96.05, 95.40],
    }
)


melted_fig1c_df = fig1c_df.melt(
    id_vars=["Num. Parameters", "Parameters"],
    var_name="Num. Replicas",
    value_name="Average Accuracy (7 Benchmarks)",
)
melted_fig1c_df["Num. Replicas"] = melted_fig1c_df["Num. Replicas"].apply(
    lambda x: int(x.replace("x", ""))
)
melted_fig1c_df.head()


plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1c_df,
    x="Num. Replicas",
    y="Average Accuracy (7 Benchmarks)",
    hue="Parameters",
    palette="tab10",
    # palette="viridis",
    marker="o",
    markersize=15,
)
g.set(
    title="Chinchilla Scaling (Fig 1c)",
    xlabel="Num. Replicas",
    xscale="symlog",
    xlim=(-0.25, None),
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1c_y=accuracy_x=multiplier_hue=params",
)
# plt.show()

plt.close()
plt.figure(figsize=default_figsize)
g = sns.lineplot(
    data=melted_fig1c_df,
    x="Num. Parameters",
    y="Average Accuracy (7 Benchmarks)",
    hue="Num. Replicas",
    hue_norm=matplotlib.colors.SymLogNorm(linthresh=1.0),
    # palette="tab10",
    palette="viridis",
    marker="o",
    markersize=15,
)
g.set(
    title="Chinchilla Scaling (Fig 1c)",
    xscale="log",
    ylabel="Average Accuracy (7 Benchmarks)",
    ylim=(40, 100),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="fig1c_y=accuracy_x=params_hue=multiplier",
)
# plt.show()
