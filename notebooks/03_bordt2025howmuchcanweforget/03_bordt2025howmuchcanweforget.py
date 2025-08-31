import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
from scipy.optimize import least_squares
import seaborn as sns

import src.analyze
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

num_questions_per_num_replicas = {
    0: 0,
    4: 8000,
    12: 5000,
    36: 2000,
    144: 2000,
}

num_questions_per_benchmark = {
    "winogrande": 7987,
    "piqa": 8709,
    "social_i_qa": 3604,
    "boolq": 2849,
    "hellaswag": 8612,
    "mmlu": 9591,
    "arc-easy": 2612,
}
total_num_questions = sum(num_questions_per_benchmark.values())

average_tokens_per_benchmark_question = {
    "winogrande": 23.80,
    "piqa": 44.58,
    "social_i_qa": 37.825,
    "boolq": 139.18,
    "hellaswag": 82.77,
    "mmlu": 73.01,
    "arc-easy": 40.91,
}
total_num_tokens_per_benchmark = sum(
    [
        num_questions_per_benchmark[benchmark]
        * average_tokens_per_benchmark_question[benchmark]
        for benchmark in num_questions_per_benchmark
    ]
)

num_unique_benchmark_tokens_per_num_replicas = {}
num_total_benchmark_tokens_per_num_replicas = {}
for num_replicas, num_questions in num_questions_per_num_replicas.items():
    num_unique_benchmark_tokens_per_num_replicas[num_replicas] = (
        total_num_tokens_per_benchmark * num_questions / total_num_questions
    )
    num_total_benchmark_tokens_per_num_replicas[num_replicas] = (
        num_replicas * num_unique_benchmark_tokens_per_num_replicas[num_replicas]
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
fig1a_df["Num. Tokens"] = 7e9

melted_fig1a_df = fig1a_df.melt(
    id_vars=["Num. Parameters", "Parameters", "Num. Tokens"],
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
fig1b_df["Num. Parameters"] = 124e6
fig1b_df["Parameters"] = "124M"
fig1b_df["Num. Tokens"] = (
    fig1b_df[r"$\times$ Chinchilla Tokens"] * 20.0 * fig1b_df["Num. Parameters"]
)

melted_fig1b_df = fig1b_df.melt(
    id_vars=[
        r"$\times$ Chinchilla Tokens",
        "Num. Parameters",
        "Parameters",
        "Num. Tokens",
    ],
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
fig1c_df["Num. Tokens"] = 20 * fig1c_df["Num. Parameters"]

melted_fig1c_df = fig1c_df.melt(
    id_vars=["Num. Parameters", "Parameters", "Num. Tokens"],
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

melted_all_df = pd.concat(
    [melted_fig1a_df, melted_fig1b_df, melted_fig1c_df]
).reset_index(drop=True)

melted_all_df["Num. Total Benchmark Tokens"] = melted_all_df["Num. Replicas"].map(
    num_total_benchmark_tokens_per_num_replicas
)
melted_all_df["Proportion Benchmark Tokens"] = (
    melted_all_df["Num. Total Benchmark Tokens"] / melted_all_df["Num. Tokens"]
)

plt.close()
plt.figure(figsize=default_figsize)
g = sns.relplot(
    data=melted_all_df,
    kind="line",
    x="Proportion Benchmark Tokens",
    y="Average Accuracy (7 Benchmarks)",
    col="Parameters",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="viridis",
    marker="o",
    legend=False,
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=accuracy_x=proportion_benchmark_tokens_hue=params_col=params",
)
# plt.show()

# ----- Hill family with N-dependent parameters -----
#   y_min(N)  = 100 * sigmoid(a0 + a1*logN)                          in (0, 100)
#   y_max(N)  = y_min + (100 - y_min) * sigmoid(b0 + b1*logN)        in (y_min, 100)
#   p50(N)    = exp(c0 - gamma*logN)                                  > 0
#   h         = exp(logh)                                             > 0
#   mu(p, N)  = y_min + (y_max - y_min) * p^h / (p^h + p50(N)^h)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hill4_family_hN(p, logN, theta, logN_ref):
    # theta = [a0,a1,b0,b1,c0,gamma,h0,h1]
    a0, a1, b0, b1, c0, gamma, h0, h1 = theta
    ymin = 100.0 * _sigmoid(a0 + a1 * logN)
    frac = _sigmoid(b0 + b1 * logN)
    ymax = ymin + (100.0 - ymin) * frac
    p50 = np.exp(c0 - gamma * logN)
    h = np.exp(h0 + h1 * (logN - logN_ref))  # <-- h varies with N
    z = (p**h) / (p**h + p50**h)
    return ymin + (ymax - ymin) * z


def fit_hill4_family_varying_h(df):
    df = df.dropna(
        subset=[
            "Proportion Benchmark Tokens",
            "Average Accuracy (7 Benchmarks)",
            "Num. Parameters",
        ]
    ).copy()
    p = df["Proportion Benchmark Tokens"].to_numpy(float)
    y_obs = df["Average Accuracy (7 Benchmarks)"].to_numpy(float)
    logN = np.log(df["Num. Parameters"].to_numpy(float))
    logN_ref = np.mean(logN)  # center for stability

    # ---- inits ----
    y10, y90 = np.percentile(y_obs, [10, 90]) / 100.0
    y10 = np.clip(y10, 1e-3, 1 - 1e-3)
    a0 = np.log(y10 / (1 - y10))
    a1 = 0.0
    b0, b1 = np.log(0.9 / 0.1), 0.0
    ppos = p[p > 0]
    c0 = np.log(np.median(ppos)) if ppos.size else -8.0
    gamma = 0.4

    # let small models have smaller h (more bend-up): start with h0≈ln(0.8), h1>0
    h0 = np.log(0.8)
    h1 = 0.15 / (logN.max() - logN.min() + 1e-9)  # small slope

    theta0 = np.array([a0, a1, b0, b1, c0, gamma, h0, h1], dtype=float)

    def residuals(theta):
        return hill4_family_hN(p, logN, theta, logN_ref) - y_obs

    res = least_squares(residuals, theta0, loss="soft_l1", f_scale=3.0, max_nfev=20000)
    return res, logN_ref


# ---- fit and build overlay curves (colors from your existing mapping) ----
res, logN_ref = fit_hill4_family_varying_h(melted_all_df)
theta = res.x

curves = []
for N in np.sort(melted_all_df["Num. Parameters"].unique()):
    sub = melted_all_df[melted_all_df["Num. Parameters"] == N]
    p_grid = np.linspace(
        0.0, max(1.05 * melted_all_df["Proportion Benchmark Tokens"].max(), 1e-6), 200
    )
    y_hat = hill4_family_hN(p_grid, np.log(N) * np.ones_like(p_grid), theta, logN_ref)
    curves.append(
        pd.DataFrame(
            {
                "Proportion Benchmark Tokens": p_grid,
                "Average Accuracy (7 Benchmarks) (fit)": y_hat,
                "Num. Parameters": N,
                "Parameters": sub["Parameters"].iloc[0],
            }
        )
    )
fit_df = pd.concat(curves, ignore_index=True)


# Helpers to inspect the learned N-dependencies
def summarize_params_at_N(N):
    logN = np.log(np.asarray(N, dtype=float))
    a0, a1, b0, b1, c0, gamma, logh = theta
    ymin = 100.0 * _sigmoid(a0 + a1 * logN)
    frac = _sigmoid(b0 + b1 * logN)
    ymax = ymin + (100.0 - ymin) * frac
    p50 = np.exp(c0 - gamma * logN)
    h = np.exp(logh)
    return dict(ymin=ymin, ymax=ymax, p50=p50, h=h)


# # Example: see parameters for each model size present in your data
# for N in np.sort(melted_all_df["Num. Parameters"].unique()):
#     print(f"N={N:g} ->", summarize_params_at_N(N))


# --- overlay the fitted family on each facet (NO 'col' param here) ---
plt.close()
g = sns.relplot(
    data=melted_all_df,
    kind="scatter",
    x="Proportion Benchmark Tokens",
    y="Average Accuracy (7 Benchmarks)",
    col="Parameters",
    hue="Num. Parameters",
    hue_norm=matplotlib.colors.LogNorm(),
    palette="viridis",
    marker="o",
    legend=False,
)

# Draw fitted curves on the matching facet.
cmap = matplotlib.cm.get_cmap("viridis")
norm = matplotlib.colors.LogNorm(
    vmin=melted_all_df["Num. Parameters"].min(),
    vmax=melted_all_df["Num. Parameters"].max(),
)
for col_val, ax in zip(g.col_names, g.axes.flat):
    sub_fit = fit_df[fit_df["Parameters"] == col_val]
    if not sub_fit.empty:
        sns.lineplot(
            data=sub_fit,
            x="Proportion Benchmark Tokens",
            y="Average Accuracy (7 Benchmarks) (fit)",
            hue="Num. Parameters",
            hue_order=[124e6, 350e6, 774e6, 1.6e9],
            hue_norm=norm,
            palette=cmap,
            errorbar=None,
            linewidth=2,
            linestyle="--",
            legend=False,
            ax=ax,
        )

# # optional cosmetics
# for ax in g.axes.flat:
#     ax.set_ylim(0, 100)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=accuracy_x=proportion_benchmark_tokens_hue=params_col=params_overlay=fits",
)
# plt.show()


plt.close()
colors = [matplotlib.cm.viridis(i) for i in np.linspace(0, 1, 4)]
plt.figure(figsize=default_figsize)
g = sns.scatterplot(
    data=melted_all_df,
    x="Proportion Benchmark Tokens",
    y="Average Accuracy (7 Benchmarks)",
    hue="Parameters",
    palette=colors,  # Pass the generated list of colors
    marker="o",
)
g = sns.lineplot(
    data=fit_df,
    x="Proportion Benchmark Tokens",
    y="Average Accuracy (7 Benchmarks) (fit)",
    hue="Parameters",
    palette=colors,  # Pass the same list here
    linestyle="--",
    legend=False,
)
src.plot.format_g_legend_in_scientific_notation(g=g, num_decimal_digits=2)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=accuracy_x=proportion_benchmark_tokens_hue=params_fit",
)
plt.show()
print("Finished 03_bordt2025howmuchcanweforget.py")
