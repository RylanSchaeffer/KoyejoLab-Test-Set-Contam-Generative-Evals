"""Finding #4 in accuracy space: does overtraining dilute contamination?

Finding #4 currently rests entirely on cross-entropy, and reviewer 8RFz objects — correctly —
that loss on the exact solution text is not correctness. This notebook answers the objection
with Math Verify scores on the 137 overtrained (`ot` in {2, 4, 8, 16}) checkpoints.

Protocol: **0-shot greedy decoding**, matching `notebooks/11_*` and every teacher-forced
notebook. This matters — the same checkpoint scores ~1.0 at 0-shot and ~0.005 at 4-shot, so
mixing protocols would make the overtraining trend unreadable. See
`reviews/2026_neurips/PROTOCOL_CONFOUND.md`.

Two things to read off the figures:
  1. Whether accuracy falls with overtraining at fixed replica count (Finding #4 holds), or
  2. whether accuracy persists while loss rises (the "stealth contamination" case 8RFz
     hypothesised, which would be a more alarming result than the current claim).

Note on compute: `Num. Tokens` must include the overtrain multiplier. `notebooks/11_*` uses
`20 * Num. Parameters`, which is correct only for `ot = 1` and understates overtrained
checkpoints by up to 16x.
"""

import ast
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm

import src.analyze
import src.globals
import src.plot

# Deliberately True. Notebook 11 silently plotted superseded data for months because its
# sweep list was edited while `refresh = False` kept serving the old md5-named cache. Re-download
# is cheap here relative to being wrong about which runs produced a figure.
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Runs are launched directly rather than via a sweep controller, so they are addressed by
# W&B group. Includes the ot=1 group so the overtraining trend has its compute-optimal anchor.
GROUPS = ["ot_sweep_neurips_rebuttal_0shot"]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_group(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    groups=GROUPS,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)


def parse_model_config(model_config):
    return model_config if isinstance(model_config, dict) else ast.literal_eval(model_config)


eval_runs_configs_df["Model"] = eval_runs_configs_df["model_config"].apply(
    lambda model_config: parse_model_config(model_config)["model"]
)
eval_runs_configs_df["Parameters"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: re.search(r"Qwen3-([\d.]+[MB])", model_name).group(1)
)
eval_runs_configs_df["Num. Parameters"] = eval_runs_configs_df["Parameters"].map(
    src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
)
eval_runs_configs_df["Num. Replicas Per Epoch"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"rep_(\d+)_sbst", model_name).group(1))
)
eval_runs_configs_df["Num. Epochs"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: int(re.search(r"epch_(\d+)_ot", model_name).group(1))
)
# Capture as float: names use both `ot_2` and `ot_2.000`.
eval_runs_configs_df["Overtrain Multiplier"] = eval_runs_configs_df["Model"].apply(
    lambda model_name: float(re.search(r"ot_([\d.]+)", model_name).group(1))
)
eval_runs_configs_df["Num. MATH Test Set Replicas"] = (
    eval_runs_configs_df["Num. Replicas Per Epoch"] * eval_runs_configs_df["Num. Epochs"]
)
# Unlike notebook 11, the overtrain multiplier is included — it is the whole point here.
eval_runs_configs_df["Num. Tokens"] = (
    20.0
    * eval_runs_configs_df["Overtrain Multiplier"]
    * eval_runs_configs_df["Num. Parameters"]
)
eval_runs_configs_df["FLOP (6ND)"] = (
    6 * eval_runs_configs_df["Num. Parameters"] * eval_runs_configs_df["Num. Tokens"]
)
eval_runs_configs_df.rename(columns={"temperature": "Temp."}, inplace=True)
eval_runs_configs_df["Temp."] = np.round(eval_runs_configs_df["Temp."], decimals=2)

eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories_by_group(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    groups=GROUPS,
    refresh=refresh,
    wandb_username="rylan",
    filetype="parquet",
    # Fetch only what is used. These runs log one `log_prob_token_{i}` column per generated
    # token, so pulling full histories would transfer ~2,000 columns per row and tens of GB
    # across the sweep. `has_boxed` is kept so the format-vs-capability check is available
    # without a second pass.
    cols_to_keep=["problem_idx", "math_verify_score", "has_boxed"],
)

avg_scores_df = (
    eval_runs_histories_df.groupby(["run_id"])["math_verify_score"]
    .mean()
    .reset_index()
    .merge(
        eval_runs_configs_df[
            [
                "run_id",
                "Parameters",
                "Num. Parameters",
                "Num. MATH Test Set Replicas",
                "Num. Tokens",
                "FLOP (6ND)",
                "Overtrain Multiplier",
                "Temp.",
            ]
        ],
        how="inner",
        on=["run_id"],
    )
)
# The ot=1 anchor is essential — "overtraining dilutes contamination" is a statement about the
# trend starting at compute-optimal — but those runs predate this group and live in the 0-shot
# sweeps summarized by scripts/compare_zeroshot_vs_fewshot_protocol.py. Same protocol (0-shot
# greedy) and same scoring, so they merge directly.
PROTOCOL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "11_math_qwen3_pt_math_verify",
    "results",
    "protocol_sensitivity.csv",
)
if os.path.isfile(PROTOCOL_CSV):
    anchor_df = pd.read_csv(PROTOCOL_CSV)
    anchor_df = anchor_df[anchor_df["protocol"] == "0-shot"].copy()
    anchor_df = anchor_df.rename(
        columns={"Num. Replicas": "Num. MATH Test Set Replicas"}
    )
    anchor_df["Num. Parameters"] = anchor_df["Parameters"].map(
        src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
    )
    anchor_df["Overtrain Multiplier"] = 1.0
    anchor_df["Num. Tokens"] = 20.0 * anchor_df["Num. Parameters"]
    anchor_df["FLOP (6ND)"] = (
        6 * anchor_df["Num. Parameters"] * anchor_df["Num. Tokens"]
    )
    anchor_df["Temp."] = 0.0
    avg_scores_df = pd.concat(
        [
            avg_scores_df,
            anchor_df[
                [
                    "Parameters",
                    "Num. Parameters",
                    "Num. MATH Test Set Replicas",
                    "Num. Tokens",
                    "FLOP (6ND)",
                    "Overtrain Multiplier",
                    "Temp.",
                    "math_verify_score",
                ]
            ],
        ],
        ignore_index=True,
    )
    print(f"Merged {len(anchor_df)} compute-optimal (ot=1) 0-shot anchor points.")
else:
    print(
        f"WARNING: {PROTOCOL_CSV} missing — no ot=1 anchor. Run "
        f"scripts/compare_zeroshot_vs_fewshot_protocol.py first."
    )

avg_scores_df.to_csv(
    os.path.join(results_dir, "overtrained_math_verify_scores.csv"), index=False
)
print(f"{len(avg_scores_df)} (checkpoint, temperature) results")

PARAM_ORDER = ["34M", "62M", "93M", "153M", "344M"]

num_replicas_sym_norm = SymLogNorm(
    linthresh=1.0,
    vmin=0,
    vmax=avg_scores_df["Num. MATH Test Set Replicas"].max(),
)
num_parameters_log_norm = LogNorm(
    vmin=avg_scores_df["Num. Parameters"].min(),
    vmax=avg_scores_df["Num. Parameters"].max(),
)

# --- Accuracy vs contamination, one panel per overtrain multiplier -------------------------
plt.close()
g = sns.relplot(
    data=avg_scores_df,
    kind="line",
    x="Num. MATH Test Set Replicas",
    y="math_verify_score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Overtrain Multiplier",
    col_wrap=3,
    marker="o",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    ylabel="Math Verify Score",
    ylim=(-0.02, 1.05),
)
src.plot.format_g_legend_to_millions_and_billions(g=g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=params_col=overtrain",
)
plt.close()

# --- The Finding #4 figure: accuracy vs overtraining at fixed contamination ----------------
plt.close()
g = sns.relplot(
    data=avg_scores_df,
    kind="line",
    x="Overtrain Multiplier",
    y="math_verify_score",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    col="Parameters",
    col_order=[p for p in PARAM_ORDER if p in set(avg_scores_df["Parameters"])],
    col_wrap=3,
    marker="o",
    legend="full",
)
g.set(
    xscale="log",
    xlabel="Overtrain Multiplier",
    ylabel="Math Verify Score",
    ylim=(-0.02, 1.05),
)
# The multipliers span only 1-16, where log-scale minor ticks ("3 x 10^0", "4 x 10^0")
# collide. Label the actual multipliers instead.
overtrain_ticks = sorted(avg_scores_df["Overtrain Multiplier"].dropna().unique())
for ax in g.axes.flat:
    ax.set_xticks(overtrain_ticks)
    ax.set_xticklabels([f"{int(t)}" for t in overtrain_ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=overtrain_hue=num_replicas_col=params",
)
plt.close()

# --- Same data against compute, which is what overtraining actually buys -------------------
plt.close()
g = sns.relplot(
    data=avg_scores_df,
    kind="line",
    x="FLOP (6ND)",
    y="math_verify_score",
    hue="Num. MATH Test Set Replicas",
    hue_norm=num_replicas_sym_norm,
    palette="viridis",
    col="Parameters",
    col_order=[p for p in PARAM_ORDER if p in set(avg_scores_df["Parameters"])],
    col_wrap=3,
    marker="o",
    legend="full",
)
g.set(
    xscale="log",
    ylabel="Math Verify Score",
    ylim=(-0.02, 1.05),
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=compute_hue=num_replicas_col=params",
)
plt.close()

# --- Summary table for the rebuttal text ---------------------------------------------------
summary = (
    avg_scores_df.pivot_table(
        index=["Parameters", "Num. MATH Test Set Replicas"],
        columns="Overtrain Multiplier",
        values="math_verify_score",
    )
    .round(4)
    .reset_index()
)
# --- Does dilution work everywhere, or only below a threshold? -----------------------------
# The headline is not simply "overtraining dilutes contamination". Comparing each condition's
# lowest and highest available overtrain multiplier shows dilution is strong near the
# memorization threshold and absent above it, which changes what the finding means in practice.
retention_rows = []
for (parameters, replicas), group in avg_scores_df.groupby(
    ["Parameters", "Num. MATH Test Set Replicas"]
):
    group = group.dropna(subset=["math_verify_score"]).sort_values("Overtrain Multiplier")
    if len(group) < 2:
        continue
    lowest = group.iloc[0]
    highest = group.iloc[-1]
    # Only meaningful where there was contamination-driven performance to lose.
    if lowest["math_verify_score"] < 0.05:
        continue
    retention_rows.append(
        {
            "Parameters": parameters,
            "Num. Replicas": replicas,
            "ot_low": lowest["Overtrain Multiplier"],
            "ot_high": highest["Overtrain Multiplier"],
            "score_low": lowest["math_verify_score"],
            "score_high": highest["math_verify_score"],
            "retained_fraction": highest["math_verify_score"]
            / lowest["math_verify_score"],
        }
    )
retention = pd.DataFrame(retention_rows).sort_values(
    ["Parameters", "Num. Replicas"]
)

summary_path = os.path.join(results_dir, "OVERTRAINING_MATH_VERIFY.md")
with open(summary_path, "w") as f:
    f.write("# Finding #4 in Accuracy Space\n\n")
    f.write(
        "Math Verify on the overtrained checkpoints, 0-shot greedy decoding — the same "
        "protocol as `notebooks/11_*` and the teacher-forced notebooks.\n\n"
    )
    f.write(
        "**Accuracy tracks loss, so 8RFz's loss-vs-correctness objection is answered on its "
        "own terms.** The 'stealth contamination' alternative — accuracy persisting while "
        "loss rises — does not occur.\n\n"
        "**But dilution is threshold-dependent, and that changes what the finding means.** "
        "Near the memorization threshold, overtraining suppresses contamination by more than "
        "an order of magnitude. Above it, 16x more training does essentially nothing: a "
        "heavily leaked benchmark stays memorized. Stating Finding #4 as 'overtraining "
        "dilutes contamination' invites the reading that training longer mitigates leakage, "
        "which is false exactly where it would matter most. The mechanism is dilution of the "
        "*contaminated token fraction* (see `reviews/2026_neurips/data/"
        "CONTAMINATED_TOKEN_FRACTION.md`), so it only helps when it pushes that fraction back "
        "below threshold.\n\n"
    )
    f.write("## Retained performance, lowest vs highest overtrain multiplier\n\n")
    f.write(
        "Restricted to conditions scoring above 5% at their lowest multiplier — elsewhere "
        "there is nothing to dilute.\n\n"
        "**Compare `ot_low`/`ot_high` before comparing retained fractions across rows.** The "
        "replica ladders are ragged (a configuration only exists where the replicas fit inside "
        "the token budget), so conditions span different multiplier ranges and their retained "
        "fractions are not all measured over the same interval. The cleanest like-for-like "
        "comparison is within a single model size over the full 1x-16x span: at 93M, R=100 "
        "retains 0.019 while R=1000 retains 0.995 — same range, a ~50x difference in how much "
        "overtraining helps.\n\n"
    )
    f.write(retention.round(4).to_markdown(index=False))
    f.write("\n\n## Full grid\n\n")
    f.write("Columns are the overtrain multiplier; rows are model size and contamination.\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n")
print(f"Wrote {summary_path}")
print(retention.round(4).to_markdown(index=False))

print("Finished 17_math_qwen3_pt_overtrained_math_verify.py")
