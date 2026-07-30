"""Do perturbed MATH problems ever keep the original answer?

Finding #2 at 0-shot gives Perturbed a *larger* residual than Rephrased (4.78% vs 2.74% at
R >= 100), which is backwards on the face of it. Rephrasing preserves the answer, so a model
regurgitating a memorized solution should still be scored correct whenever it recovers the
original wording. Perturbing changes the numbers *and the answer*, so regurgitation should be
scored wrong essentially always — Perturbed ought to sit at or below Rephrased.

The most likely benign explanation is that some perturbations leave the ground-truth answer
unchanged (the perturbed numbers happen to yield the same result, or the perturbation did not
alter the quantity the answer depends on). Every such problem hands a free point to a purely
memorizing model and inflates the Perturbed column.

This measures that overlap directly by comparing gold answers across the datasets. If it is
material, the Perturbed residual should be reported net of it, or those problems excluded.

Usage:
    python scripts/check_perturbed_answer_overlap.py
"""

import argparse
import os

import pandas as pd

import src.data
from src.scoring import extract_boxed_answer


def gold_answers(dataset) -> list:
    """Extract each problem's boxed ground-truth answer, or None when absent."""
    return [extract_boxed_answer(solution) for solution in dataset["solution"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reviews/2026_neurips/data")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    original = src.data.load_dataset_hendrycks_math()["test"]
    perturbed = src.data.load_dataset_math_perturbed()["test"]
    rephrased = src.data.load_dataset_math_rephrased()["test"]

    print(
        f"sizes — original {len(original)}, perturbed {len(perturbed)}, "
        f"rephrased {len(rephrased)}"
    )

    original_answers = gold_answers(original)
    perturbed_answers = gold_answers(perturbed)
    rephrased_answers = gold_answers(rephrased)

    rows = []
    for label, answers in [
        ("Perturbed", perturbed_answers),
        ("Rephrased", rephrased_answers),
    ]:
        n = min(len(original_answers), len(answers))
        # Aligned by index; both datasets derive from the same test set in the same order.
        comparable = [
            (a, b)
            for a, b in zip(original_answers[:n], answers[:n])
            if a is not None and b is not None
        ]
        identical = sum(1 for a, b in comparable if a.strip() == b.strip())
        rows.append(
            {
                "condition": label,
                "n_compared": len(comparable),
                "n_identical_answer": identical,
                "fraction_identical": identical / max(len(comparable), 1),
            }
        )
        print(
            f"{label}: {identical}/{len(comparable)} "
            f"({100 * identical / max(len(comparable), 1):.2f}%) keep the original answer"
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(args.output_dir, "perturbed_answer_overlap.csv"), index=False
    )

    # Per-problem mask so downstream analysis can report the Perturbed column net of the
    # problems where memorization scores correct by construction.
    mask_rows = []
    for problem_idx, (original_answer, perturbed_answer) in enumerate(
        zip(original_answers, perturbed_answers)
    ):
        unchanged = (
            original_answer is not None
            and perturbed_answer is not None
            and original_answer.strip() == perturbed_answer.strip()
        )
        mask_rows.append(
            {
                "problem_idx": problem_idx,
                "answer_unchanged": bool(unchanged),
                "original_answer": original_answer,
                "perturbed_answer": perturbed_answer,
            }
        )
    mask_path = os.path.join(args.output_dir, "perturbed_answer_unchanged_mask.csv")
    pd.DataFrame(mask_rows).to_csv(mask_path, index=False)
    print(f"Wrote {mask_path}")

    perturbed_fraction = float(
        df.loc[df["condition"] == "Perturbed", "fraction_identical"].iloc[0]
    )

    lines = [
        "# Do Modified Problems Keep the Original Answer?",
        "",
        "At 0-shot, the Perturbed column shows a *larger* residual than Rephrased "
        "(4.78% vs 2.74% at R >= 100). That ordering is backwards: rephrasing preserves the "
        "answer, so regurgitation can still score correct, whereas perturbing changes the "
        "answer and regurgitation should score wrong.",
        "",
        "A problem whose perturbation leaves the ground-truth answer unchanged hands a free "
        "point to a purely memorizing model.",
        "",
        df.round(4).to_markdown(index=False),
        "",
    ]
    if perturbed_fraction >= 0.02:
        lines += [
            f"**Material.** {100 * perturbed_fraction:.2f}% of perturbed problems keep the "
            "original answer, which is on the same order as the entire Perturbed residual. The "
            "residual should be reported net of these problems, or they should be excluded from "
            "the Perturbed column, before the number goes in the paper.",
        ]
    else:
        lines += [
            f"**Not the explanation.** Only {100 * perturbed_fraction:.2f}% of perturbed "
            "problems keep the original answer — too few to account for the Perturbed residual. "
            "Something else is producing it; candidates are partial credit from answers that "
            "are close but not identical, or genuine (weak) transfer. Investigate before "
            "characterizing the residual in the paper.",
        ]
    lines.append("")

    report_path = os.path.join(args.output_dir, "PERTURBED_ANSWER_OVERLAP.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
