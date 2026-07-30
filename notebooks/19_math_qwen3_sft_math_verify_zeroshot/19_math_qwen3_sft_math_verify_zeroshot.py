"""Finding #5 at matched protocol: what does SFT do to contamination-driven accuracy?

`REBUTTAL_PLAN.md` P0.1 claims SFT collapses contaminated accuracy ~60x, from ~100% pretrained
to ~1-2% post-SFT. That figure is an artifact of comparing **0-shot pretrained** numbers against
**4-shot SFT** numbers: notebook 11's cache is 0-shot, notebook 13's sweeps are 4-shot, and the
same checkpoint scores ~1.0 versus ~0.005 across that difference alone. Matched at 4-shot the
two stages are 0.40% and 0.20% — a factor of two, not sixty.

This measures both stages at **0-shot**, so the comparison isolates the effect of SFT. The
pretrained side is reused from the 0-shot sweeps rather than re-run: same checkpoints before
fine-tuning, same protocol, same scoring.

Note the SFT checkpoints live under `jkazdan/` on the Hub, not `RylanSchaeffer/` — the repos were
transferred after the original evaluations, so W&B run configs record paths that now 404.
"""

import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

import src.analyze
import src.globals
import src.plot

# See notebook 17 for why this is not left False.
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

GROUP = "sft_rerun_zeroshot"
PARAM_ORDER = ["34M", "62M", "93M", "153M", "344M"]

PRETRAINED_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "11_math_qwen3_pt_math_verify",
    "results",
    "protocol_sensitivity_rescored.csv",
)


def parse_model_config(model_config):
    return model_config if isinstance(model_config, dict) else ast.literal_eval(model_config)


configs = src.analyze.download_wandb_project_runs_configs_by_group(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    groups=[GROUP],
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)
histories = src.analyze.download_wandb_project_runs_histories_by_group(
    wandb_project_path="memorization-scoring-vs-sampling-eval",
    data_dir=data_dir,
    groups=[GROUP],
    refresh=refresh,
    wandb_username="rylan",
    filetype="parquet",
    cols_to_keep=["problem_idx", "math_verify_score", "has_boxed"],
)

configs["Model"] = configs["model_config"].apply(lambda c: parse_model_config(c)["model"])
configs["Parameters"] = configs["Model"].apply(
    lambda m: re.search(r"Qwen3-([\d.]+[MB])", m).group(1)
)
configs["Num. Replicas"] = configs["Model"].apply(
    lambda m: int(re.search(r"rep_(\d+)_sbst", m).group(1))
)

# Raw accuracy conflates two different failures: the model no longer emits `\boxed{}` at all,
# versus it emits one and is wrong. Scoring counts both as incorrect. At 344M R=1000 the
# post-SFT boxed rate is ~2%, so the raw drop there is mostly the former. Accuracy *conditioned*
# on emitting the format separates them.
histories["scored_correct"] = histories["math_verify_score"].fillna(0) > 0
boxed_only = histories[histories["has_boxed"] > 0]
conditional = (
    boxed_only.groupby("run_id")["scored_correct"]
    .mean()
    .rename("sft_score_given_boxed")
    .reset_index()
)

sft = (
    histories.groupby("run_id")
    .agg(sft_score=("math_verify_score", "mean"), sft_boxed_rate=("has_boxed", "mean"))
    .reset_index()
    .merge(conditional, on="run_id", how="left")
    .merge(configs[["run_id", "Parameters", "Num. Replicas"]], on="run_id", how="inner")
)
print(f"{len(sft)} SFT checkpoints evaluated at 0-shot")

pretrained = pd.read_csv(PRETRAINED_CSV)
# Rescored (boxed-required) column, so the pretrained baseline is scored the same way the SFT
# runs are. The originally logged 0-shot scores were lenient and would inflate the baseline.
pretrained = pretrained[pretrained["protocol"] == "0-shot"][
    ["Parameters", "Num. Replicas", "strict_score"]
].rename(columns={"strict_score": "pretrained_score"})

# The superseded 0-shot sweeps have no 344M run at R=0 or R=316, so the left merge below leaves
# `pretrained_score` NaN there and `informative` silently drops both rows. That is not neutral:
# 344M R=316 is pretrained 99.84% -> 0.14% post-SFT, one of the largest collapses in the grid,
# and omitting it biases the headline mean upward. Those two checkpoints were re-run 0-shot into
# their own group for notebook 18; reuse them here.
GAP_GROUP = "zeroshot_original_gap_344m"
try:
    gap_configs = src.analyze.download_wandb_project_runs_configs_by_group(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        groups=[GAP_GROUP],
        refresh=refresh,
        wandb_username="rylan",
        finished_only=True,
    )
    gap_histories = src.analyze.download_wandb_project_runs_histories_by_group(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        groups=[GAP_GROUP],
        refresh=refresh,
        wandb_username="rylan",
        filetype="parquet",
        cols_to_keep=["problem_idx", "math_verify_score"],
    )
    gap_configs["Model"] = gap_configs["model_config"].apply(
        lambda c: parse_model_config(c)["model"]
    )
    gap_configs["Parameters"] = gap_configs["Model"].apply(
        lambda m: re.search(r"Qwen3-([\d.]+[MB])", m).group(1)
    )
    gap_configs["Num. Replicas"] = gap_configs["Model"].apply(
        lambda m: int(re.search(r"rep_(\d+)_sbst", m).group(1))
    )
    gap = (
        gap_histories.groupby("run_id")["math_verify_score"]
        .mean()
        .rename("pretrained_score")
        .reset_index()
        .merge(
            gap_configs[["run_id", "Parameters", "Num. Replicas"]],
            on="run_id",
            how="inner",
        )[["Parameters", "Num. Replicas", "pretrained_score"]]
    )
    have = set(zip(pretrained["Parameters"], pretrained["Num. Replicas"]))
    gap = gap[
        [(p, r) not in have for p, r in zip(gap["Parameters"], gap["Num. Replicas"])]
    ]
    if not gap.empty:
        pretrained = pd.concat([pretrained, gap], ignore_index=True)
        print(
            f"Merged {len(gap)} gap-filled pretrained baseline(s): "
            f"{list(zip(gap['Parameters'], gap['Num. Replicas']))}"
        )
except Exception as e:
    print(f"Gap-fill group unavailable ({type(e).__name__}: {e}); two cells stay blank.")

# The merge is one-to-one by construction; assert it rather than trust it, since a duplicated
# key would silently inflate `sft` rows and reweight every mean below.
assert not pretrained.duplicated(["Parameters", "Num. Replicas"]).any()
assert not sft.duplicated(["Parameters", "Num. Replicas"]).any()

n_sft_rows = len(sft)
merged = sft.merge(pretrained, on=["Parameters", "Num. Replicas"], how="left")
assert len(merged) == n_sft_rows, "baseline merge changed the row count"
_unmatched = merged[merged["pretrained_score"].isna()]
if not _unmatched.empty:
    print(
        "WARNING: no pretrained baseline for "
        f"{list(zip(_unmatched['Parameters'], _unmatched['Num. Replicas']))} — "
        "these are dropped from the headline means."
    )
merged["retained_fraction"] = merged["sft_score"] / merged["pretrained_score"].replace(
    0, np.nan
)
merged["Num. Parameters"] = merged["Parameters"].map(
    src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
)
merged = merged.sort_values(
    ["Num. Parameters", "Num. Replicas"]
)
merged.to_csv(os.path.join(results_dir, "sft_zeroshot_vs_pretrained.csv"), index=False)

# --- Figure: both stages on one axis --------------------------------------------------------
long = pd.concat(
    [
        merged[["Parameters", "Num. Parameters", "Num. Replicas", "pretrained_score"]]
        .rename(columns={"pretrained_score": "Math Verify Score"})
        .assign(Stage="Pretrained"),
        merged[["Parameters", "Num. Parameters", "Num. Replicas", "sft_score"]]
        .rename(columns={"sft_score": "Math Verify Score"})
        .assign(Stage="After SFT"),
    ],
    ignore_index=True,
).dropna(subset=["Math Verify Score"])

num_parameters_log_norm = LogNorm(
    vmin=long["Num. Parameters"].min(), vmax=long["Num. Parameters"].max()
)

plt.close()
g = sns.relplot(
    data=long,
    kind="line",
    x="Num. Replicas",
    y="Math Verify Score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Stage",
    col_order=["Pretrained", "After SFT"],
    marker="o",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    xlabel="Num. MATH Test Set Replicas",
    ylim=(-0.02, 1.05),
)
src.plot.format_g_legend_to_millions_and_billions(g=g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=params_col=stage_zeroshot",
)
plt.close()

# --- Report ---------------------------------------------------------------------------------
# Only conditions with contamination-driven performance to lose are informative about what SFT
# removes; elsewhere both stages sit at the floor and the ratio is noise over noise.
informative = merged[merged["pretrained_score"] >= 0.05].dropna(
    subset=["retained_fraction"]
)

report_path = os.path.join(results_dir, "SFT_ZEROSHOT.md")
with open(report_path, "w") as f:
    f.write("# Finding #5 at Matched Protocol (0-Shot)\n\n")
    f.write(
        "Both stages measured at 0-shot greedy decoding, so the comparison isolates SFT rather "
        "than confounding it with a protocol change.\n\n"
        "**The ~60x collapse quoted in `REBUTTAL_PLAN.md` P0.1 is an artifact** of comparing "
        "0-shot pretrained against 4-shot SFT. Use the numbers below instead.\n\n"
    )
    if not informative.empty:
        f.write("## What SFT removes, where there was something to remove\n\n")
        f.write(
            f"Restricted to the {len(informative)} conditions scoring >= 5% before SFT.\n\n"
        )
        f.write(
            f"- Mean pretrained: **{100 * informative['pretrained_score'].mean():.2f}%**\n"
            f"- Mean after SFT: **{100 * informative['sft_score'].mean():.2f}%**\n"
            f"- Median retained fraction: **{informative['retained_fraction'].median():.3f}** "
            f"(range {informative['retained_fraction'].min():.3f}-"
            f"{informative['retained_fraction'].max():.3f})\n\n"
        )
        f.write(
            "The collapse is real and large, but it varies by more than an order of magnitude "
            "across conditions — quote the range, not a single multiplier.\n\n"
        )
        f.write("## How much of the drop is format loss?\n\n")
        f.write(
            "Raw accuracy scores a response incorrect both when the model emits no "
            "`\\boxed{}` and when it emits one that is wrong. Those are different failures, and "
            "post-SFT boxed rates differ enormously by model size, so the distinction changes "
            "the interpretation:\n\n"
        )
        format_view = informative[
            [
                "Parameters",
                "Num. Replicas",
                "pretrained_score",
                "sft_score",
                "sft_boxed_rate",
                "sft_score_given_boxed",
            ]
        ].copy()
        for column in ("pretrained_score", "sft_score", "sft_score_given_boxed"):
            format_view[column] = (100 * format_view[column]).round(2)
        format_view["sft_boxed_rate"] = format_view["sft_boxed_rate"].round(3)
        f.write(format_view.to_markdown(index=False))
        intact = informative[informative["sft_boxed_rate"] >= 0.5]
        collapsed = informative[informative["sft_boxed_rate"] < 0.2]
        f.write(
            f"\n\n- **{len(intact)} conditions keep the format** (boxed rate >= 0.5). There the "
            f"accuracy drop is genuine loss of memorized content: the model still answers in the "
            f"expected form and is simply wrong.\n"
            f"- **{len(collapsed)} conditions lose the format** (boxed rate < 0.2), and these are "
            f"concentrated in the larger models. There the raw drop mostly measures that the "
            f"model stopped emitting `\\boxed{{}}` at all, and attributing it entirely to "
            f"forgetting would overstate the result.\n\n"
            f"Report both columns. The defensible claim is that SFT removes the contamination "
            f"advantage; the mechanism differs by scale, and `sft_score_given_boxed` is the "
            f"column that isolates capability from formatting. Note it has its own selection "
            f"effect — it conditions on a subset the model chose — so it is a diagnostic, not a "
            f"drop-in replacement for the headline number.\n\n"
        )
    f.write("## Per-condition\n\n")
    display = merged[
        [
            "Parameters",
            "Num. Replicas",
            "pretrained_score",
            "sft_score",
            "retained_fraction",
            "sft_boxed_rate",
        ]
    ].copy()
    for column in ("pretrained_score", "sft_score"):
        display[column] = (100 * display[column]).round(2)
    display["retained_fraction"] = display["retained_fraction"].round(3)
    display["sft_boxed_rate"] = display["sft_boxed_rate"].round(3)
    f.write(display.to_markdown(index=False))
    f.write(
        "\n\n`sft_boxed_rate` is the fraction of responses containing a `\\boxed{}` at all. "
        "If it collapses, the accuracy drop is partly a formatting artifact rather than lost "
        "capability — check it before attributing the drop entirely to forgetting.\n"
    )
print(f"Wrote {report_path}")
if not informative.empty:
    print(
        f"pretrained {100 * informative['pretrained_score'].mean():.2f}% -> "
        f"SFT {100 * informative['sft_score'].mean():.2f}%, "
        f"median retained {informative['retained_fraction'].median():.3f}"
    )

print("Finished 19_math_qwen3_sft_math_verify_zeroshot.py")
