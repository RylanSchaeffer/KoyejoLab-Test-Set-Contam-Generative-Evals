"""Bootstrap confidence intervals for Math Verify scores, resampling the 5,000 test problems.

Reviewer aPBL (W3) and the AC's third bullet ask for uncertainty quantification. Per-problem
`math_verify_score` values are already in W&B run history, so interval estimates need no GPU.

**This is not multi-seed.** It quantifies sampling error over the test set — how much the score
would move if we had drawn a different 5,000 MATH problems — and says nothing about variance
across pretraining seeds or decoding seeds. The rebuttal must say so plainly; presenting these
as if they covered seed variance would be worse than reporting no intervals at all. What they
do establish is that the effects being claimed are enormous relative to test-set noise: a jump
from ~1% to ~100% is not a sampling artifact under any reading.

Runs the 0-shot pretrained sweeps, which are the protocol the manuscript's Finding #1 figure
actually uses (see `reviews/2026_neurips/PROTOCOL_CONFOUND.md`).

Usage:
    python scripts/bootstrap_math_verify_cis.py --num-bootstrap 10000


SUPERSEDED 2026-07-30. This computes percentile-bootstrap intervals from the *leniently* scored logs, while every number the rebuttal reports is strict-scored, so its point estimates contradict the measured 0.00% floor; a percentile bootstrap is also degenerate at the eight zero-scoring conditions. Use the exact binomial intervals in `notebooks/11_*/results/strict_score_binomial_cis.csv` instead (median half-width 0.123 pp). Kept because it records how the superseded 0.33 pp figure arose.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import wandb

import src.globals

WANDB_ENTITY = "rylan"
WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"

# Superseded-in-name-only 0-shot sweeps: these are what notebook 11's figures come from.
ZERO_SHOT_SWEEPS = [
    "6y9dy2ow", "lnrpy3ed",   # 34M
    "5oo55o9s", "10q465ij",   # 62M
    "q5uoy1eu", "f5djvfth",   # 93M
    "vnz1h147", "xkzfmbhk",   # 153M
    "39rugx2e",               # 344M
]


def bootstrap_ci(scores: np.ndarray, num_bootstrap: int, seed: int, alpha: float):
    """Percentile bootstrap over problems.

    Scores are 0/1 per problem, so resampling problems with replacement is exactly the
    relevant sampling model: the test set is one draw from a population of problems.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    # Binomial shortcut: resampling n Bernoulli values and averaging is equivalent in
    # distribution to Binomial(n, p_hat) / n, which avoids materializing a huge index matrix.
    p_hat = scores.mean()
    draws = rng.binomial(n, p_hat, size=num_bootstrap) / n
    lower = float(np.quantile(draws, alpha / 2))
    upper = float(np.quantile(draws, 1 - alpha / 2))
    return p_hat, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", default="notebooks/11_math_qwen3_pt_math_verify/results"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    api = wandb.Api(timeout=600)
    records = []

    for sweep_id in ZERO_SHOT_SWEEPS:
        try:
            sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        except Exception as e:
            print(f"  sweep {sweep_id}: {e}")
            continue
        for run in sweep.runs:
            if run.state != "finished":
                continue
            try:
                model_name = run.config["model_config"]["model"]
                temperature = float(run.config["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(temperature - args.temperature) > 1e-6 or model_name.endswith("_sft"):
                continue

            scores = np.array(
                [
                    1 if row.get("math_verify_score") else 0
                    for row in run.scan_history(keys=["math_verify_score"])
                ],
                dtype=float,
            )
            if scores.size == 0:
                continue

            mean, lower, upper = bootstrap_ci(
                scores, args.num_bootstrap, args.seed, args.alpha
            )
            parameters = re.search(r"Qwen3-([\d.]+[MB])", model_name)
            replicas = re.search(r"rep_(\d+)_sbst", model_name)
            records.append(
                {
                    "Parameters": parameters.group(1) if parameters else None,
                    "Num. Replicas": int(replicas.group(1)) if replicas else None,
                    "n_problems": int(scores.size),
                    "math_verify_score": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_halfwidth": (upper - lower) / 2,
                    "run_id": run.id,
                }
            )
            print(
                f"  {records[-1]['Parameters']:>5} R={records[-1]['Num. Replicas']:<5} "
                f"{mean:.4f}  [{lower:.4f}, {upper:.4f}]"
            )

    if not records:
        raise SystemExit("No runs summarized.")

    df = pd.DataFrame(records).sort_values(["Parameters", "Num. Replicas"])
    csv_path = os.path.join(args.output_dir, "math_verify_bootstrap_cis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    df["formatted"] = df.apply(
        lambda r: f"{100 * r['math_verify_score']:.2f} "
        f"[{100 * r['ci_lower']:.2f}, {100 * r['ci_upper']:.2f}]",
        axis=1,
    )
    table = df.pivot_table(
        index="Num. Replicas",
        columns="Parameters",
        values="formatted",
        aggfunc="first",
    )
    # Pivot sorts columns as strings, which puts 153M and 344M before 34M. Order by size.
    table = table[
        sorted(
            table.columns,
            key=lambda p: src.globals.MODEL_NAMES_TO_PARAMETERS_DICT.get(p, float("inf")),
        )
    ]

    lines = [
        "# Bootstrap Confidence Intervals on Math Verify",
        "",
        f"{int(100 * (1 - args.alpha))}% percentile bootstrap over the "
        f"{int(df['n_problems'].median())} MATH test problems, "
        f"{args.num_bootstrap:,} resamples, greedy decoding, **0-shot** "
        "(the protocol behind the manuscript's Finding #1 figure).",
        "",
        "## What this does and does not cover",
        "",
        "These intervals quantify **sampling error over the test set** — how much the score would",
        "move given a different draw of 5,000 problems. They are **not** multi-seed error bars:",
        "they say nothing about variance across pretraining seeds or decoding seeds. State that",
        "explicitly in the rebuttal and commit to seeds for camera-ready rather than letting these",
        "be read as covering that concern.",
        "",
        "What they do establish: the intervals are on the order of a percentage point, while the",
        "effects claimed span roughly 1% to 100%. The contamination effect is orders of magnitude",
        "larger than test-set sampling noise.",
        "",
        "## Math Verify %, [95% CI]",
        "",
        table.to_markdown(),
        "",
        f"Median CI half-width across all conditions: "
        f"**{100 * df['ci_halfwidth'].median():.2f} percentage points**.",
        "",
    ]
    report_path = os.path.join(args.output_dir, "BOOTSTRAP_CIS.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")
    print(table.to_markdown())


if __name__ == "__main__":
    main()
