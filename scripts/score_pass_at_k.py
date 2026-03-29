"""Score generated samples and compute pass@k for MATH benchmark evaluation.

This CPU-only script reads generated samples from a JSONL file, scores each
response with math-verify (the same scoring used in eval_language_model.py),
computes the unbiased pass@k estimator, and outputs results stratified by
MATH difficulty level and subject type.

Scoring is cached to a scores.jsonl file alongside the samples file. On
re-runs, only new (unscored) samples are evaluated, making it efficient
to incrementally add samples and re-score.

Usage:
    python scripts/score_pass_at_k.py \
        --samples_path results/pass_at_k/model_name/temp=1.0/samples.jsonl \
        --k_values 1 10 100 1000

    # With custom output directory
    python scripts/score_pass_at_k.py \
        --samples_path results/pass_at_k/model_name/temp=1.0/samples.jsonl \
        --k_values 1 10 100 \
        --output_dir results/pass_at_k/model_name/temp=1.0/
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from math_verify import parse, verify

import src.data


def extract_boxed_answer(text: str) -> str | None:
    """Extract content of the last \\boxed{...} in text using brace-depth matching.

    Returns the content inside the braces, or None if no \\boxed{} is found.
    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}} -> \\frac{1}{2}).
    """
    # Find the start of the last \boxed{
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    # Move past \boxed{
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    # i now points one past the closing brace
    return text[start : i - 1]


def score_response(gold_parsed, response_text: str) -> bool:
    """Score a response by extracting \\boxed{} answer and verifying with math-verify.

    Only counts a response as correct if it contains a \\boxed{} expression.
    This eliminates false positives from parse() extracting bare numbers from
    garbage text (see SCORING_INVESTIGATION.md).
    """
    boxed_content = extract_boxed_answer(response_text)
    if boxed_content is None:
        return False
    try:
        target_parsed = parse(f"\\boxed{{{boxed_content}}}")
        return bool(verify(gold=gold_parsed, target=target_parsed))
    except Exception:
        return False


def pass_at_k(n: int, c: int, k: int):
    """Unbiased estimator of pass@k.

    Computes the probability that at least one of k randomly chosen samples
    from n total samples is correct, given c correct samples total. Uses the
    combinatorial formula: pass@k = 1 - comb(n-c, k) / comb(n, k).

    Reference: Chen et al., "Evaluating Large Language Models Trained on Code"
    (https://arxiv.org/abs/2107.03374), Appendix A.

    Args:
        n: Total number of samples for this problem.
        c: Number of correct samples.
        k: Number of samples to draw.

    Returns:
        float if computable, None if k > n (insufficient samples).
    """
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score generated samples with math-verify and compute pass@k."
    )
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to samples.jsonl file produced by generate_pass_at_k_samples.py.",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 10, 100],
        help="k values for pass@k computation (default: 1 10 100).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output files. Defaults to the same directory as samples_path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    samples_path = Path(args.samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    output_dir = Path(args.output_dir) if args.output_dir else samples_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MATH ground-truth solutions for scoring.
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    ground_truth_solutions = test_dataset["solution"]

    # 2. Read all samples from JSONL, grouped by problem_idx.
    samples_by_problem = defaultdict(list)  # problem_idx -> [response_text, ...]
    metadata_by_problem = {}  # problem_idx -> {"level": ..., "type": ...}

    with open(samples_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pid = record["problem_idx"]
            samples_by_problem[pid].append(record["response_text"])
            if pid not in metadata_by_problem:
                metadata_by_problem[pid] = {
                    "level": record["level"],
                    "type": record["type"],
                }

    n_problems_with_samples = len(samples_by_problem)
    total_samples = sum(len(v) for v in samples_by_problem.values())
    print(
        f"Loaded {total_samples} samples across {n_problems_with_samples} problems."
    )

    # 3. Load cached scores from scores.jsonl (if it exists).
    scores_path = samples_path.with_name("scores.jsonl")
    cached_scores = {}  # (problem_idx, sample_idx) -> bool
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cached_scores[
                    (record["problem_idx"], record["sample_idx"])
                ] = record["correct"]
        print(f"Loaded {len(cached_scores)} cached scores from {scores_path}")

    # 4. Score uncached samples and append new scores to cache.
    n_new_scores = 0
    scores_by_problem = defaultdict(list)  # problem_idx -> [bool, ...]

    with open(scores_path, "a") as f_scores:
        for pid in sorted(samples_by_problem.keys()):
            gold_parsed = parse(ground_truth_solutions[pid])
            for sample_idx, response_text in enumerate(samples_by_problem[pid]):
                cache_key = (pid, sample_idx)
                if cache_key in cached_scores:
                    correct = cached_scores[cache_key]
                else:
                    correct = score_response(gold_parsed, response_text)
                    f_scores.write(
                        json.dumps(
                            {
                                "problem_idx": pid,
                                "sample_idx": sample_idx,
                                "correct": correct,
                            }
                        )
                        + "\n"
                    )
                    n_new_scores += 1
                scores_by_problem[pid].append(correct)

            # Progress reporting every 100 problems.
            if (pid + 1) % 100 == 0:
                print(
                    f"  Scored {pid + 1}/{n_problems_with_samples} problems..."
                )

    print(
        f"Scored {n_new_scores} new samples. "
        f"Total scored: {n_new_scores + len(cached_scores)}"
    )

    # 5. Compute pass@k per problem.
    rows = []
    for pid in sorted(samples_by_problem.keys()):
        n_total = len(scores_by_problem[pid])
        n_correct = sum(scores_by_problem[pid])
        row = {
            "problem_idx": pid,
            "level": metadata_by_problem[pid]["level"],
            "type": metadata_by_problem[pid]["type"],
            "n_total": n_total,
            "n_correct": n_correct,
        }
        for k in args.k_values:
            row[f"pass@{k}"] = pass_at_k(n_total, n_correct, k)
        rows.append(row)

    df = pd.DataFrame(rows)

    # 6. Save per-problem results to CSV.
    results_csv_path = output_dir / "results.csv"
    df.to_csv(results_csv_path, index=False)
    print(f"\nPer-problem results saved to {results_csv_path}")

    # 7. Build summary tables stratified by level and type.
    k_cols = [f"pass@{k}" for k in args.k_values if f"pass@{k}" in df.columns]

    # Filter to only rows where pass@k is not None (sufficient samples).
    # For aggregation, we need numeric values.
    df_for_agg = df.copy()
    for col in k_cols:
        df_for_agg[col] = pd.to_numeric(df_for_agg[col], errors="coerce")

    # --- Summary by difficulty level ---
    level_agg = {
        "n_problems": ("problem_idx", "count"),
        "n_total_samples": ("n_total", "sum"),
        "n_total_correct": ("n_correct", "sum"),
    }
    for col in k_cols:
        level_agg[col] = (col, "mean")

    summary_level = df_for_agg.groupby("level").agg(**level_agg).reset_index()

    # Add overall row.
    overall_row = {
        "level": "Overall",
        "n_problems": len(df_for_agg),
        "n_total_samples": df_for_agg["n_total"].sum(),
        "n_total_correct": df_for_agg["n_correct"].sum(),
    }
    for col in k_cols:
        overall_row[col] = df_for_agg[col].mean()
    summary_level = pd.concat(
        [summary_level, pd.DataFrame([overall_row])], ignore_index=True
    )

    print("\n=== Results by Difficulty Level ===")
    print(summary_level.to_markdown(index=False))

    # --- Summary by subject type ---
    type_agg = {
        "n_problems": ("problem_idx", "count"),
        "n_total_samples": ("n_total", "sum"),
        "n_total_correct": ("n_correct", "sum"),
    }
    for col in k_cols:
        type_agg[col] = (col, "mean")

    summary_type = df_for_agg.groupby("type").agg(**type_agg).reset_index()

    print("\n=== Results by Subject Type ===")
    print(summary_type.to_markdown(index=False))

    # 8. Save markdown summary.
    summary_md_path = output_dir / "summary.md"
    with open(summary_md_path, "w") as f:
        f.write("# pass@k Results\n\n")
        f.write(f"**Samples file:** `{args.samples_path}`\n\n")
        f.write(
            f"**k values:** {', '.join(str(k) for k in args.k_values)}\n\n"
        )
        f.write("## By Difficulty Level\n\n")
        f.write(summary_level.to_markdown(index=False))
        f.write("\n\n## By Subject Type\n\n")
        f.write(summary_type.to_markdown(index=False))
        f.write("\n")

    print(f"\nSummary saved to {summary_md_path}")
    print(f"All results written to {output_dir}")


if __name__ == "__main__":
    main()
