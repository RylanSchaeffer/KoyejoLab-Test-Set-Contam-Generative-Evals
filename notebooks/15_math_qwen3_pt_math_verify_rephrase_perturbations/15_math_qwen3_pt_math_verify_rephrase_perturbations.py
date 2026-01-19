import os

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.analyze
import src.globals
import src.plot

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Load Stella's data
stella_df = pd.read_csv(os.path.join(data_dir, "stella_data.csv"), index_col=0)

# Map model names to parameter counts
stella_df["Num. Parameters"] = stella_df["Model"].map(
    src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
)

# Rename Replicas column for consistency
stella_df.rename(columns={"Replicas": "Num. MATH Test Set Replicas"}, inplace=True)

# Reshape from wide to long format for faceting
stella_long_df = stella_df.melt(
    id_vars=["Model", "Num. MATH Test Set Replicas", "Num. Parameters"],
    value_vars=["Original", "Rephrased", "Perturbed"],
    var_name="Condition",
    value_name="Math Verify Score",
)

# Set up normalization for hue
num_parameters_log_norm = LogNorm(
    vmin=stella_long_df["Num. Parameters"].min(),
    vmax=stella_long_df["Num. Parameters"].max(),
)

# Plot 1: All three conditions (Original, Rephrased, Perturbed)
plt.close()
g = sns.relplot(
    data=stella_long_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=["Original", "Rephrased", "Perturbed"],
    marker="o",
    facet_kws={"sharey": True},
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylim=(-0.05, 1.05),
)
# Format legend labels to show M/B suffix
for txt in g._legend.texts:
    try:
        num = float(txt.get_text())
        if 1e6 <= num < 1e9:
            txt.set_text(f"{int(num / 1e6)}M")
        elif num >= 1e9:
            txt.set_text(f"{int(num / 1e9)}B")
    except ValueError:
        pass
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=num_params_col=condition",
)
# plt.show()

# Plot 2: Only Rephrased and Perturbed conditions
stella_rephrase_perturbed_df = stella_long_df[
    stella_long_df["Condition"].isin(["Rephrased", "Perturbed"])
]

plt.close()
g = sns.relplot(
    data=stella_rephrase_perturbed_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=["Rephrased", "Perturbed"],
    marker="o",
    facet_kws={"sharey": True},
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylim=(-0.05, 1.05),
)
# Format legend labels to show M/B suffix
for txt in g._legend.texts:
    try:
        num = float(txt.get_text())
        if 1e6 <= num < 1e9:
            txt.set_text(f"{int(num / 1e6)}M")
        elif num >= 1e9:
            txt.set_text(f"{int(num / 1e9)}B")
    except ValueError:
        pass
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=num_params_col=condition_rephrase_perturbed",
)
# plt.show()

print("Finished 15_math_qwen3_pt_math_verify_rephrase_perturbations.py")
