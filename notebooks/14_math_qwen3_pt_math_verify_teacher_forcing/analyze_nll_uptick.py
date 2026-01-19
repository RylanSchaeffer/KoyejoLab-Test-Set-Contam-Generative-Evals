"""Analyze when NLL increases with token index.

Creates ONE plot showing mean NLL vs token index for different contamination levels.
"""

import ast
import hashlib
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

import src.analyze
import src.globals
import src.plot


# Setup
data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Load configs
eval_teacher_forcing_sweep_ids = [
    "9vtnq3bd",  # 34M
    "ovps81c2",  # 62M
    "oi9x67mh",  # 93M
    "em23bzb7",  # 153M
    "sy8h8i80",  # 344M
]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-eval-teacher-forcing",
    data_dir=data_dir,
    sweep_ids=eval_teacher_forcing_sweep_ids,
    refresh=False,
    wandb_username="rylan",
    finished_only=True,
)

eval_runs_configs_df["Model"] = eval_runs_configs_df["model_config"].apply(
    lambda x: ast.literal_eval(x)["model"]
)
eval_runs_configs_df["Parameters"] = eval_runs_configs_df["Model"].apply(
    lambda x: re.search(r"Qwen3-([\d.]+[MB])", x).group(1)
)
eval_runs_configs_df["Num. Parameters"] = eval_runs_configs_df["Parameters"].apply(
    lambda x: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[x]
)
eval_runs_configs_df["Num. Replicas"] = eval_runs_configs_df["Model"].apply(
    lambda x: int(re.search(r"rep_(\d+)_sbst", x).group(1))
    * int(re.search(r"epch_(\d+)_ot", x).group(1))
)

run_id_to_config = eval_runs_configs_df.set_index("run_id")[
    ["Parameters", "Num. Parameters", "Num. Replicas"]
].to_dict("index")

# Load data
filename = "sweeps=" + ",".join(eval_teacher_forcing_sweep_ids)
hashed_filename = hashlib.md5(filename.encode()).hexdigest()
histories_path = os.path.join(data_dir, hashed_filename + "_runs_histories.parquet")

parquet_file = pq.ParquetFile(histories_path)
all_columns = [field.name for field in parquet_file.schema]
all_log_prob_cols = sorted(
    [c for c in all_columns if c.startswith("log_prob_token_")],
    key=lambda x: int(x.replace("log_prob_token_", "")),
)

# Token positions 0-800
MAX_TOKEN = 800
log_prob_cols = [
    c for c in all_log_prob_cols
    if int(c.replace("log_prob_token_", "")) <= MAX_TOKEN
]

print("Loading data...")
table = parquet_file.read_row_group(0, columns=["run_id"] + log_prob_cols)
df = table.to_pandas()

# Compute mean NLL at each token position for each (model_size, num_replicas)
results = []
for run_id in df["run_id"].unique():
    if run_id not in run_id_to_config:
        continue
    config = run_id_to_config[run_id]
    run_data = df[df["run_id"] == run_id]
    for col in log_prob_cols:
        token_idx = int(col.replace("log_prob_token_", ""))
        nlls = -run_data[col].dropna()
        if len(nlls) > 0:
            results.append({
                "Model Size": config["Parameters"],
                "Num. Parameters": config["Num. Parameters"],
                "Num. Replicas": config["Num. Replicas"],
                "Token Index": token_idx,
                "Mean NLL": nlls.mean(),
            })

results_df = pd.DataFrame(results)

# Aggregate across model sizes (average over all models)
agg_df = results_df.groupby(["Num. Replicas", "Token Index"])["Mean NLL"].mean().reset_index()

# Normalize: divide by NLL at token 0 for each replica count
normalized_data = []
for replicas in agg_df["Num. Replicas"].unique():
    subset = agg_df[agg_df["Num. Replicas"] == replicas].copy()
    nll_at_0 = subset[subset["Token Index"] == 0]["Mean NLL"].values[0]
    subset["Normalized NLL"] = subset["Mean NLL"] / nll_at_0
    normalized_data.append(subset)

norm_df = pd.concat(normalized_data)

# Plot: Normalized NLL vs Token Index
print("Creating plot...")
plt.close()
fig, ax = plt.subplots(figsize=(8, 6))

# Only show key replica values
key_replicas = [0, 10, 100, 1000]
palette = sns.color_palette("viridis", len(key_replicas))

for i, replicas in enumerate(key_replicas):
    subset = norm_df[norm_df["Num. Replicas"] == replicas].sort_values("Token Index")
    ax.plot(
        subset["Token Index"],
        subset["Normalized NLL"],
        label=f"{replicas}",
        color=palette[i],
        linewidth=2,
    )

ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xscale("log")
ax.set_xlabel(r"Token Index")
ax.set_ylabel(r"Normalized NLL (relative to token 0)")
ax.legend(title="Num. Replicas")
ax.set_xlim(1, 800)

plt.tight_layout()
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=normalized_nll_x=token_index_hue=replicas",
)
plt.close()

print(f"Plot saved to {results_dir}")
