"""Regenerate Figure 1's Math Verify panels under boxed-required (strict) scoring.

The published Figure 1 was drawn from the scores the 0-shot sweeps *logged*, which used the
lenient scorer (`math_verify.parse()` straight off the response, ~1.38% false positives). That
lifts every near-zero point off the axis: uncontaminated models read 0.38-1.26% rather than
0.00%. Under the boxed-required scorer used everywhere else in the revision the floor is exactly
zero, which states the paper's claim more cleanly -- the baseline is not a small positive number
that contamination lifts, it is zero.

Layout matches the published figure (two panels: score vs replicas coloured by size, score vs
size coloured by dose) so the "Left"/"Right" references in Sec. 3 still hold. The shaded bands
are exact binomial 95% intervals rather than seaborn's bootstrap over per-problem scores, since
the strict rescoring stores per-run aggregates; for a proportion the two agree, and the exact
interval is well defined at zero counts, which eight conditions have.

Usage:
    python scripts/plot_figure1_strict.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.stats import beta

sys.path.insert(0, os.getcwd())

import src.plot  # noqa: F401  (applies the project's global matplotlib style)

RESCORED = (
    "notebooks/11_math_qwen3_pt_math_verify/results/protocol_sensitivity_rescored.csv"
)
OUT_DIR = "notebooks/11_math_qwen3_pt_math_verify/results"

# Two conditions absent from the rescoring sweeps, recovered elsewhere and verified:
#   344M R=0   -- the ten runs of 2025-09-25 failed; sweeps woygzpil/oj6o8idv finished and are
#                 pre-4-shot. Strict 0.0000 (see reviews/2026_neurips/data/LENIENT_SCORER_AUDIT.md).
#   344M R=316 -- present in the 0-shot overtraining grid at ot=1
#                 (notebooks/17_*/results/OVERTRAINING_MATH_VERIFY.md).
GAP_FILLS = [
    {
        "Parameters": "344M",
        "Num. Replicas": 0,
        "strict_score": 0.0000,
        "n_problems": 5001,
    },
    {
        "Parameters": "344M",
        "Num. Replicas": 316,
        "strict_score": 0.9984,
        "n_problems": 5000,
    },
]

RIGHT_PANEL_REPLICAS = [0, 10, 100, 1000, 3162]

PARAM_COUNTS = {"34M": 34e6, "62M": 62e6, "93M": 93e6, "153M": 153e6, "344M": 344e6}


def binomial_ci(score, n, alpha=0.05):
    """Exact (Clopper-Pearson) interval; well defined when k = 0 or k = n."""
    k = int(round(score * n))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def load() -> pd.DataFrame:
    d = pd.read_csv(RESCORED)
    d = d[(d["protocol"] == "0-shot") & (d["Temp."] == 0.0)]
    d = d[["Parameters", "Num. Replicas", "strict_score", "n_problems"]].dropna(
        subset=["strict_score"]
    )
    d = pd.concat([d, pd.DataFrame(GAP_FILLS)], ignore_index=True)
    d["Num. Parameters"] = d["Parameters"].map(PARAM_COUNTS)
    lo_hi = [binomial_ci(r.strict_score, int(r.n_problems)) for r in d.itertuples()]
    d["ci_lo"] = [x[0] for x in lo_hi]
    d["ci_hi"] = [x[1] for x in lo_hi]
    return d.sort_values(["Num. Parameters", "Num. Replicas"])


def main() -> None:
    d = load()

    param_norm = LogNorm(
        vmin=d["Num. Parameters"].min(), vmax=d["Num. Parameters"].max()
    )
    rep_norm = SymLogNorm(
        linthresh=1.0, vmin=d["Num. Replicas"].min(), vmax=d["Num. Replicas"].max()
    )
    flare = plt.get_cmap("flare")
    viridis = plt.get_cmap("viridis")

    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Left: score vs contamination dose, coloured by model size.
    ax = axes[0]
    for p, sub in d.groupby("Num. Parameters"):
        sub = sub.sort_values("Num. Replicas")
        c = flare(param_norm(p))
        ax.plot(
            sub["Num. Replicas"],
            sub["strict_score"],
            marker="o",
            color=c,
            label=(
                src.plot.format_number_to_millions_and_billions(p)
                if hasattr(src.plot, "format_number_to_millions_and_billions")
                else f"{p/1e6:.0f}M"
            ),
        )
        ax.fill_between(
            sub["Num. Replicas"],
            sub["ci_lo"],
            sub["ci_hi"],
            color=c,
            alpha=0.25,
            linewidth=0,
        )
    ax.set_xscale("symlog")
    ax.set_xlim(-0.1, 3500)
    ax.set_xlabel("Num. MATH Test Set Replicas")
    ax.set_ylabel("Math Verify Score")
    ax.legend(loc="upper left", title="Num. Parameters")

    # Right: score vs model size, coloured by contamination dose. Restricted to the same five
    # dose levels the published figure shows, so the two are directly comparable and the legend
    # stays legible; a level present at only one size is drawn as a marker.
    ax = axes[1]
    for r in RIGHT_PANEL_REPLICAS:
        sub = d[d["Num. Replicas"] == r].sort_values("Num. Parameters")
        if sub.empty:
            continue
        c = viridis(rep_norm(r))
        ax.plot(
            sub["Num. Parameters"],
            sub["strict_score"],
            marker="o",
            color=c,
            label=f"{int(r)}",
        )
        if len(sub) > 1:
            ax.fill_between(
                sub["Num. Parameters"],
                sub["ci_lo"],
                sub["ci_hi"],
                color=c,
                alpha=0.25,
                linewidth=0,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Num. Parameters")
    ax.set_ylabel("Math Verify Score")
    # The R=0/R=10 curves run flat along y=0 and R>=1000 along y=1, so the free space is the
    # mid-left band between them.
    ax.legend(loc="center left", title="Num. Replicas")

    plt.tight_layout()
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=OUT_DIR,
        plot_filename="y=math_verify_strict_by_num_parameters_by_num_replicas",
    )
    plt.close()

    floor = d[d["Num. Replicas"] == 0]["strict_score"]
    print(
        f"Wrote {OUT_DIR}/y=math_verify_strict_by_num_parameters_by_num_replicas.[pdf|png]"
    )
    print(f"Uncontaminated floor across sizes: {floor.min():.4f}-{floor.max():.4f}")
    print(f"Conditions plotted: {len(d)}")


if __name__ == "__main__":
    main()
