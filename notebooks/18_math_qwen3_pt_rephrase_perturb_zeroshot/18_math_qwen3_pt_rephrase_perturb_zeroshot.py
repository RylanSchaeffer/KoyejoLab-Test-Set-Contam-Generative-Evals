"""Finding #2 under the protocol where it can actually be tested: 0-shot rephrase / perturb.

Table 1 claims contamination-driven gains vanish when MATH problems are rephrased (same numbers,
new wording) or perturbed (same wording, new numbers). Demonstrating that requires measuring
both conditions in a protocol where the gains exist. `notebooks/15_*` uses 4-shot, where even
the *original* test set scores ~0.005 at 344M — there is nothing to collapse from, and indeed
its Original, Rephrased and Perturbed curves are indistinguishable. At 0-shot the same
checkpoints reach ~1.0 on the original set, so the comparison becomes meaningful.

This also gives Table 1 citable provenance. Its printed values predate the 4-shot sweeps by two
months and correspond to no run in this W&B account (see `reviews/2026_neurips/PROTOCOL_CONFOUND.md`),
which is a problem because reviewer 8RFz asks directly how Table 1 is computed.

The original-condition baseline is reused from the 0-shot sweeps rather than re-run, since those
are the same checkpoints under the same protocol and scoring.
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

# See notebook 17: caching by md5-of-sweep-list is what let notebook 11 serve stale data for
# months. Re-downloading is cheap here.
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

CONDITION_GROUPS = {
    "Rephrased": "table1_rerun_zeroshot_rephrased",
    "Perturbed": "table1_rerun_zeroshot_perturbed",
}
PARAM_ORDER = ["34M", "62M", "93M", "153M", "344M"]

# 0-shot original-condition scores. MUST be the *rescored* file: the Rephrased/Perturbed runs
# below are scored with boxed-required scoring, and the originally logged 0-shot scores were not
# (they predate db75c5f and used lenient math_verify.parse()). Mixing them inflates the Original
# column by up to the lenient scorer's ~1.4% false-positive rate, which at R=0 is the entire
# value. See scripts/rescore_zeroshot_with_boxed_required.py.
# Problems whose perturbation left the ground-truth answer unchanged; see
# scripts/check_perturbed_answer_overlap.py.
PERTURBED_MASK_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reviews",
    "2026_neurips",
    "data",
    "perturbed_answer_unchanged_mask.csv",
)

ORIGINAL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "11_math_qwen3_pt_math_verify",
    "results",
    "protocol_sensitivity_rescored.csv",
)


def parse_model_config(model_config):
    return model_config if isinstance(model_config, dict) else ast.literal_eval(model_config)


def load_condition(condition: str, group: str) -> pd.DataFrame:
    """Return per-checkpoint mean Math Verify for one modified-dataset condition."""
    configs = src.analyze.download_wandb_project_runs_configs_by_group(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        groups=[group],
        refresh=refresh,
        wandb_username="rylan",
        finished_only=True,
    )
    histories = src.analyze.download_wandb_project_runs_histories_by_group(
        wandb_project_path="memorization-scoring-vs-sampling-eval",
        data_dir=data_dir,
        groups=[group],
        refresh=refresh,
        wandb_username="rylan",
        filetype="parquet",
        cols_to_keep=["problem_idx", "math_verify_score"],
    )

    configs["Model"] = configs["model_config"].apply(
        lambda c: parse_model_config(c)["model"]
    )
    configs["Parameters"] = configs["Model"].apply(
        lambda m: re.search(r"Qwen3-([\d.]+[MB])", m).group(1)
    )
    configs["Num. Replicas"] = configs["Model"].apply(
        lambda m: int(re.search(r"rep_(\d+)_sbst", m).group(1))
    )

    if condition == "Perturbed" and os.path.isfile(PERTURBED_MASK_CSV):
        # 11.64% of perturbed problems keep the original ground-truth answer, so a model that
        # merely regurgitates the memorized solution is scored correct on them by construction.
        # Reporting the Perturbed column without excluding these overstates residual capability.
        mask = pd.read_csv(PERTURBED_MASK_CSV)
        unchanged = set(mask.loc[mask["answer_unchanged"], "problem_idx"])
        before = len(histories)
        histories = histories[~histories["problem_idx"].isin(unchanged)]
        print(
            f"  Perturbed: dropped {before - len(histories)} rows on "
            f"{len(unchanged)} answer-unchanged problems"
        )

    scores = (
        histories.groupby("run_id")["math_verify_score"]
        .mean()
        .reset_index()
        .merge(configs[["run_id", "Parameters", "Num. Replicas"]], on="run_id", how="inner")
    )
    scores["Condition"] = condition
    return scores[["Condition", "Parameters", "Num. Replicas", "math_verify_score"]]


frames = []
for condition, group in CONDITION_GROUPS.items():
    try:
        frame = load_condition(condition, group)
        print(f"{condition}: {len(frame)} checkpoints")
        frames.append(frame)
    except Exception as e:
        # Perturbed runs after rephrased, so this notebook is useful before the grid is whole.
        print(f"{condition}: not available yet ({type(e).__name__}: {e})")

if not frames:
    raise SystemExit("No modified-dataset results available yet.")

original = pd.read_csv(ORIGINAL_CSV)
original = original[original["protocol"] == "0-shot"][
    ["Parameters", "Num. Replicas", "strict_score"]
].rename(columns={"strict_score": "math_verify_score"}).copy()

# 344M had no 0-shot Original run at R=0 or R=316 in the superseded sweeps, leaving two blank
# cells. Those two were run separately into their own group; merge them in.
try:
    gap = load_condition("Original", "zeroshot_original_gap_344m")
    gap = gap[["Parameters", "Num. Replicas", "math_verify_score"]]
    have = set(zip(original["Parameters"], original["Num. Replicas"]))
    gap = gap[
        [
            (p, r) not in have
            for p, r in zip(gap["Parameters"], gap["Num. Replicas"])
        ]
    ]
    if not gap.empty:
        original = pd.concat([original, gap], ignore_index=True)
        print(f"Merged {len(gap)} gap-filled Original run(s): "
              f"{list(zip(gap['Parameters'], gap['Num. Replicas']))}")
except Exception as e:
    print(f"Gap-fill group unavailable ({type(e).__name__}: {e}); two cells stay blank.")

original["Condition"] = "Original"
frames.append(original[["Condition", "Parameters", "Num. Replicas", "math_verify_score"]])

combined = pd.concat(frames, ignore_index=True)
combined["Num. Parameters"] = combined["Parameters"].map(
    src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
)
combined.to_csv(os.path.join(results_dir, "zeroshot_rephrase_perturb.csv"), index=False)
print(f"\n{len(combined)} condition-checkpoint results")

# --- Figure: one panel per condition, accuracy vs contamination ----------------------------
condition_order = [c for c in ["Original", "Rephrased", "Perturbed"] if c in set(combined["Condition"])]
num_parameters_log_norm = LogNorm(
    vmin=combined["Num. Parameters"].min(), vmax=combined["Num. Parameters"].max()
)

plt.close()
g = sns.relplot(
    data=combined,
    kind="line",
    x="Num. Replicas",
    y="math_verify_score",
    hue="Num. Parameters",
    hue_norm=num_parameters_log_norm,
    palette="flare",
    col="Condition",
    col_order=condition_order,
    marker="o",
)
g.set(
    xscale="symlog",
    xlim=(-0.1, 3500),
    xlabel="Num. MATH Test Set Replicas",
    ylabel="Math Verify Score",
    ylim=(-0.02, 1.05),
)
src.plot.format_g_legend_to_millions_and_billions(g=g)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=math_verify_x=num_replicas_hue=params_col=condition_zeroshot",
)
plt.close()

# --- Table 1 replacement -------------------------------------------------------------------
table = combined.pivot_table(
    index=["Parameters", "Num. Replicas"], columns="Condition", values="math_verify_score"
)
table = table[[c for c in condition_order if c in table.columns]].reset_index()
table["Parameters"] = pd.Categorical(table["Parameters"], PARAM_ORDER, ordered=True)
table = table.sort_values(["Parameters", "Num. Replicas"])

percent = table.copy()
for condition in condition_order:
    percent[condition] = (100 * percent[condition]).round(2)

report_path = os.path.join(results_dir, "TABLE1_ZEROSHOT.md")
with open(report_path, "w") as f:
    f.write("# Finding #2 at 0-Shot: Table 1, With a Baseline\n\n")
    f.write(
        "Greedy decoding, 0-shot — the protocol under which contamination actually produces "
        "gains, and the one behind the manuscript's Finding #1 figure.\n\n"
        "The manuscript's Table 1 omits an Original column, so the reader supplies the baseline "
        "from Fig. 1. Including it here makes the collapse legible in one table, and gives the "
        "numbers provenance that the printed table currently lacks.\n\n"
    )
    f.write("## Math Verify %, by condition\n\n")
    f.write(percent.to_markdown(index=False))
    f.write("\n")

    # The uncontaminated floor is the reference the collapse should be judged against — not
    # zero. Quoting "collapses to baseline" without it hides whatever residual survives.
    floor = table[table["Num. Replicas"] == 0]["Original"].dropna()
    floor_mean = float(floor.mean()) if len(floor) else float("nan")

    contaminated = table[table["Num. Replicas"] >= 100]
    f.write("\n## How complete is the collapse?\n\n")
    f.write(
        f"Uncontaminated floor (R = 0, Original): **{100 * floor_mean:.2f}%** "
        f"(n = {len(floor)} model sizes).\n\n"
    )
    for condition in condition_order:
        if condition == "Original":
            continue
        paired = contaminated[["Original", condition]].dropna()
        if paired.empty:
            continue
        original_mean = float(paired["Original"].mean())
        condition_mean = float(paired[condition].mean())
        removed = (
            (original_mean - condition_mean) / (original_mean - floor_mean)
            if np.isfinite(floor_mean) and original_mean > floor_mean
            else float("nan")
        )
        # Under boxed-required scoring the uncontaminated floor is *exactly* 0.0000, so a
        # "Nx the floor" ratio is undefined. Report the residual in absolute points instead;
        # an earlier version divided by a floor of ~1% that was entirely lenient-scorer false
        # positives. See scripts/rescore_zeroshot_with_boxed_required.py.
        residual = (
            f"{condition_mean / floor_mean:.1f}x the uncontaminated floor"
            if np.isfinite(floor_mean) and floor_mean > 0
            else (
                f"**{100 * condition_mean:.2f} percentage points above an uncontaminated floor "
                f"of exactly {100 * floor_mean:.2f}%**"
            )
        )
        f.write(
            f"- **{condition}**, at R >= 100: Original **{100 * original_mean:.2f}%** -> "
            f"{condition} **{100 * condition_mean:.2f}%** (n = {len(paired)} checkpoints). "
            f"That removes **{100 * removed:.1f}%** of the contamination advantage over the "
            f"floor — but note the residual: {condition} sits at {residual}, not at it.\n"
        )
    f.write(
        "\nThe residual is small but consistent and should be stated rather than rounded away. "
        "Describing the collapse as reaching 'baseline' overstates it; 'removes the large "
        "majority of the contamination advantage, leaving a small residual' is what the data "
        "support.\n\n"
        "Note that with boxed-required scoring the uncontaminated floor is exactly 0.00% at "
        "every model size, so the residual cannot be expressed as a multiple of it. The earlier "
        "'2-3x the floor' phrasing came from a floor of ~1% that was entirely lenient-scorer "
        "false positives.\n"
    )

    missing = table[table["Original"].isna()]
    if not missing.empty:
        f.write(
            f"\n## Gaps\n\n{len(missing)} cell(s) lack a 0-shot Original score, so their "
            f"collapse cannot be quantified:\n\n"
        )
        for _, row in missing.iterrows():
            f.write(f"- {row['Parameters']}, R = {int(row['Num. Replicas'])}\n")
print(f"Wrote {report_path}")
print(percent.to_markdown(index=False))

print("Finished 18_math_qwen3_pt_rephrase_perturb_zeroshot.py")
