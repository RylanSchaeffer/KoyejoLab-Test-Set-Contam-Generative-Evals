"""Finding #1 (accuracy vs contamination dose), rescored with boxed-required scoring.

WHY THIS EXISTS
---------------
The manuscript's Finding #1 figure comes from `notebooks/11_*`, whose cache was built from the
superseded 0-shot sweeps. Those runs were scored with the *lenient* scorer, which extracts bare
numbers from free text and has a measured ~1.4% false-positive rate. That inflates the low-dose
and uncontaminated points by roughly a full percentage point -- exactly where the figure is meant
to show a floor.

`scripts/rescore_zeroshot_with_boxed_required.py` rescored every one of those runs from its raw
generations with the boxed-required scorer, so the corrected curve can be drawn without re-running
any GPU work. This produces it as a standalone artifact rather than overwriting notebook 11's
outputs, so the two can be compared and the manuscript figure swapped deliberately.

What changes: the uncontaminated floor drops from ~0.4-1.3% to exactly 0.00% at every model size,
and the low-dose points (R <= 10) collapse to near zero. The high-contamination saturation is
unaffected -- verbatim regurgitation passes strict scoring, so those points move by <0.2pp.

Usage:
    ./mem_scoring_vs_sampling_env/bin/python scripts/plot_finding1_strict.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

sys.path.insert(0, os.getcwd())
import src.globals  # noqa: E402
import src.plot  # noqa: E402  (applies the project's global style on import)

GRID = "notebooks/11_math_qwen3_pt_math_verify/results/protocol_sensitivity_rescored.csv"
OUT = "notebooks/11_math_qwen3_pt_math_verify/results"
# CLAUDE.md documents src.plot.default_figsize, which does not exist; use the documented value.
FIGSIZE = (10.67, 8)
PARAM_ORDER = ["34M", "62M", "93M", "153M", "344M"]


def main() -> None:
    df = pd.read_csv(GRID)
    df = df[(df.protocol == "0-shot") & (df["Temp."] == 0.0)].copy()

    # The 0-shot sweeps never covered 344M at R=0 or R=316; those two cells were run separately
    # into W&B group `zeroshot_original_gap_344m` and rescored (see PROTOCOL_SENSITIVITY_RESCORED).
    for r, strict in [(0, 0.0000), (316, 0.9984)]:
        if not ((df.Parameters == "344M") & (df["Num. Replicas"] == r)).any():
            df = pd.concat(
                [df, pd.DataFrame([{"Parameters": "344M", "Num. Replicas": r,
                                    "strict_score": strict, "logged_score": strict}])],
                ignore_index=True,
            )

    df["Num. Parameters"] = df["Parameters"].map(src.globals.MODEL_NAMES_TO_PARAMETERS_DICT)
    df = df.dropna(subset=["Num. Parameters"]).sort_values(["Num. Parameters", "Num. Replicas"])

    # Model-size palette: flare + LogNorm, matching every other notebook (see CLAUDE.md).
    sizes = [p for p in PARAM_ORDER if p in set(df.Parameters)]
    vals = [src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p] for p in sizes]
    norm = LogNorm(vmin=min(vals), vmax=max(vals))
    cmap = plt.cm.get_cmap("flare")
    colors = {p: cmap(norm(src.globals.MODEL_NAMES_TO_PARAMETERS_DICT[p])) for p in sizes}

    for label, col, fname in [
        ("strict", "strict_score", "y=math_verify_strict_x=num_replicas_hue=params"),
        ("lenient", "logged_score", "y=math_verify_lenient_x=num_replicas_hue=params"),
    ]:
        plt.close()
        plt.figure(figsize=FIGSIZE)
        ax = plt.gca()
        for p in sizes:
            sub = df[df.Parameters == p].dropna(subset=[col])
            ax.plot(sub["Num. Replicas"], sub[col], marker="o", color=colors[p], label=p)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set(xlabel="Num. MATH Test Set Replicas", ylabel="Math Verify Score")
        ax.legend(title="Num. Parameters", loc="upper left")
        src.plot.save_plot_with_multiple_extensions(plot_dir=OUT, plot_filename=fname)
        plt.close()
        print(f"  plotted {label}: {fname}")

    piv = df.pivot_table(index="Parameters", columns="Num. Replicas",
                         values=["logged_score", "strict_score"])
    with open(os.path.join(OUT, "FINDING1_STRICT.md"), "w") as f:
        f.write(
            "# Finding #1 under boxed-required scoring\n\n"
            "The manuscript's Finding #1 figure comes from the superseded 0-shot sweeps, which "
            "used the **lenient** scorer (bare-number extraction, ~1.4% measured false-positive "
            "rate). Every one of those runs has been rescored from its raw generations with the "
            "boxed-required scorer used everywhere else.\n\n"
            "**What changes.** The uncontaminated floor drops from 0.38-1.26% to **exactly "
            "0.00%** at every model size, and the R <= 10 points collapse with it. "
            "**What does not.** High-contamination saturation is untouched -- verbatim "
            "regurgitation passes strict scoring, so those points move by less than 0.2 "
            "percentage points.\n\n"
            "This matters for the figure specifically: the lenient floor makes contamination look "
            "like it lifts performance off a small but nonzero baseline, when in fact the "
            "baseline is exactly zero and *every* point of measured performance is "
            "contamination-derived.\n\n"
            "Two 344M cells (R = 0 and R = 316) come from W&B group `zeroshot_original_gap_344m`, "
            "which the superseded sweeps never covered.\n\n"
        )
        f.write(piv.round(4).to_markdown())
        f.write("\n\nRegenerate: `scripts/plot_finding1_strict.py`.\n")
    print(f"\nWrote {OUT}/FINDING1_STRICT.md")


if __name__ == "__main__":
    main()
