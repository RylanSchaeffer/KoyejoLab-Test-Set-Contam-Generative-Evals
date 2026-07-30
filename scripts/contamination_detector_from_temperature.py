"""Can the temperature response detect contamination without corpus or baseline access?

Every contamination detector in the paper so far requires something a third party auditing a
released model does not have: the training corpus, or a known-clean reference model of the same
size to compare accuracy against. Raw benchmark accuracy is only diagnostic if you already know
what accuracy an uncontaminated model of that scale should get.

The temperature response needs neither. It compares a model **to itself** at two decoding
temperatures, so it is self-normalizing: contaminated models lose most of their advantage
between tau = 0 and tau = 1, while uncontaminated models are flat because they had nothing to
lose. That makes it a candidate black-box detector.

This scores three features by ROC AUC:

  `greedy_only`      score(tau=0)                  — the baseline a naive auditor would use,
                                                     and which secretly requires knowing the
                                                     clean reference level to threshold.
  `absolute_drop`    score(tau=0) - score(tau=1)   — self-normalizing.
  `relative_drop`    1 - score(tau=1)/score(tau=0) — self-normalizing, scale-free.

**Read the caveats before quoting any number.** There are only a few dozen checkpoints here, all
from one architecture family, one benchmark, and one contamination mechanism (verbatim
replicas). AUC on a few dozen points is noisy, so a permutation test is reported rather than an
asymptotic interval. This is a proof of concept that the signal exists, not a validated detector.

Usage:
    python scripts/contamination_detector_from_temperature.py
"""

import argparse
import os

import numpy as np
import pandas as pd

RAW_CACHE = "notebooks/11_math_qwen3_pt_math_verify/results/temperature_response_raw.csv"


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic, with ties credited a half.

    Written out rather than pulled from sklearn to keep the tie handling explicit: several
    uncontaminated conditions produce identical feature values, and how ties are scored
    materially changes AUC at this sample size.
    """
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()
    return float(wins / (positive.size * negative.size))


def permutation_p_value(
    labels: np.ndarray, scores: np.ndarray, observed: float, num_permutations: int, seed: int
) -> float:
    """One-sided p-value: how often does shuffling labels reach this AUC?"""
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    count = 0
    for _ in range(num_permutations):
        rng.shuffle(shuffled)
        if roc_auc(shuffled, scores) >= observed:
            count += 1
    return (count + 1) / (num_permutations + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reviews/2026_neurips/data")
    parser.add_argument(
        "--contaminated-threshold",
        type=int,
        default=100,
        help="Replica count at or above which a checkpoint counts as contaminated.",
    )
    parser.add_argument(
        "--clean-threshold",
        type=int,
        default=10,
        help="Replica count at or below which a checkpoint counts as uncontaminated.",
    )
    parser.add_argument("--num-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(RAW_CACHE):
        raise SystemExit(
            f"{RAW_CACHE} not found — run scripts/analyze_temperature_response.py first."
        )
    raw = pd.read_csv(RAW_CACHE)

    wide = raw.pivot_table(
        index=["Parameters", "Num. Replicas"], columns="Temp.", values="math_verify_score"
    )
    for required in (0.0, 1.0):
        if required not in wide.columns:
            raise SystemExit(f"Temperature {required} missing from {RAW_CACHE}")

    df = wide.reset_index()[["Parameters", "Num. Replicas", 0.0, 1.0]]
    df.columns = ["Parameters", "Num. Replicas", "score_greedy", "score_tau1"]
    df = df.dropna()

    df["greedy_only"] = df["score_greedy"]
    df["absolute_drop"] = df["score_greedy"] - df["score_tau1"]
    df["relative_drop"] = 1.0 - (
        df["score_tau1"] / df["score_greedy"].where(df["score_greedy"] > 0, np.nan)
    )
    # A model at the floor at both temperatures has no measurable response; scoring it as a
    # large "drop" would be an artifact of dividing noise by noise.
    df["relative_drop"] = df["relative_drop"].fillna(0.0)

    labelled = df[
        (df["Num. Replicas"] >= args.contaminated_threshold)
        | (df["Num. Replicas"] <= args.clean_threshold)
    ].copy()
    labelled["label"] = (
        labelled["Num. Replicas"] >= args.contaminated_threshold
    ).astype(int)

    n_pos = int(labelled["label"].sum())
    n_neg = int((labelled["label"] == 0).sum())
    print(
        f"{len(labelled)} checkpoints: {n_pos} contaminated "
        f"(R >= {args.contaminated_threshold}), {n_neg} clean "
        f"(R <= {args.clean_threshold})"
    )

    results = []
    for feature in ("greedy_only", "absolute_drop", "relative_drop"):
        scores = labelled[feature].to_numpy(dtype=float)
        labels = labelled["label"].to_numpy()
        auc = roc_auc(labels, scores)
        p_value = permutation_p_value(
            labels, scores, auc, args.num_permutations, args.seed
        )
        results.append({"feature": feature, "auc": auc, "permutation_p": p_value})
        print(f"  {feature:<15} AUC={auc:.3f}  permutation p={p_value:.4g}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(args.output_dir, "contamination_detector_auc.csv"), index=False
    )
    labelled.to_csv(
        os.path.join(args.output_dir, "contamination_detector_features.csv"), index=False
    )

    lines = [
        "# Temperature Response as a Black-Box Contamination Detector",
        "",
        "Not requested by any reviewer, which is why it is worth including: it adds a",
        "contribution rather than only patching holes.",
        "",
        "## The idea",
        "",
        "Detecting contamination in a released model normally needs the training corpus, or a",
        "known-clean reference model to compare accuracy against. The temperature response needs",
        "neither — it compares a model **to itself** at two decoding temperatures. Contaminated",
        "models lose most of their advantage between tau = 0 and tau = 1 because verbatim",
        "regurgitation is a narrow high-probability path; uncontaminated models are flat because",
        "they had nothing to lose.",
        "",
        "## Separability",
        "",
        f"{len(labelled)} pretrained checkpoints, {n_pos} contaminated "
        f"(R >= {args.contaminated_threshold}) vs {n_neg} clean "
        f"(R <= {args.clean_threshold}), 0-shot greedy vs tau = 1.",
        "",
        results_df.round(4).to_markdown(index=False),
        "",
        "`greedy_only` is the naive baseline. Its advantage is that it is simple; its problem is",
        "that thresholding raw accuracy requires already knowing what accuracy a clean model of",
        "that size achieves, which is exactly what an auditor lacks. The drop features need no",
        "such reference.",
        "",
        "## The comparison above cannot establish which feature is better",
        "",
        "This is the most important caveat and it should not be buried. In this grid **every",
        "uncontaminated checkpoint scores near zero**, because these are tiny models trained from",
        "scratch with no mathematical capability (see the pass@k result: 0 correct out of",
        "5,000,000 samples). When all negatives sit at the floor, *any* feature that keys on high",
        "greedy accuracy separates the classes almost perfectly — which is why `greedy_only`",
        "scores 0.996 here.",
        "",
        "So these AUCs establish that **the temperature signal exists and is strong**. They do",
        "**not** establish that the drop features beat raw accuracy, because the regime where the",
        "drop features should win — a genuinely capable clean model that scores well at tau = 0",
        "and stays there at tau = 1 — contains no checkpoints in this study. Claiming superiority",
        "from this table would be an overclaim of exactly the kind the paper is already being",
        "criticized for.",
        "",
        "The honest framing: the mechanism is demonstrated, the deployment advantage is argued",
        "from first principles (no reference model required), and validating it needs capable",
        "clean models as negatives. That is a concrete, checkable camera-ready commitment.",
        "",
        "## What this does not show",
        "",
        "- A few dozen checkpoints, one architecture family, one benchmark, one contamination",
        "  mechanism (verbatim replicas). AUC on this sample size is noisy; the permutation test",
        "  is reported instead of an asymptotic interval for that reason.",
        "- These are small models trained from scratch. A large pretrained model with genuine",
        "  competence may degrade differently under sampling, and that is the case that matters",
        "  for real audits.",
        "- Untested against paraphrased or partial contamination, which is where a detector would",
        "  most plausibly fail.",
        "",
        "Present it as a proof of concept with a clear path to validation, not as a working",
        "detector. Overclaiming here would repeat the framing error the paper is already being",
        "criticized for.",
        "",
    ]
    report_path = os.path.join(args.output_dir, "CONTAMINATION_DETECTOR.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
