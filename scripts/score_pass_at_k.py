"""Score generated samples and compute pass@k for MATH benchmark evaluation.

This CPU-only script reads generated samples from one or more JSONL files, scores each
response with math-verify (the same scoring used in eval_language_model.py), computes the
unbiased pass@k estimator, and outputs results stratified by MATH difficulty level and
subject type.

Two properties make this tractable on the full 5,000 x 1,000 grid:

*Streaming.* Sample files run to several GB apiece; responses are never all held in memory.
Only per-problem counters and the (tiny) set of `\\boxed{}`-bearing responses are retained.

*Boxed prefilter.* `src.scoring.score_response` scores a response as incorrect unless it
contains a `\\boxed{...}`, so math-verify only ever needs to run on responses that contain
that substring. On the uncontaminated 344M checkpoint that is ~0.3% of samples, which turns
a multi-day job into a few minutes. The prefilter is exact, not an approximation: it is the
same condition `score_response` itself short-circuits on.

Scoring of the boxed candidates is cached to `scores.jsonl` alongside the first samples
file. Non-boxed samples are not cached — recomputing them is free. On re-runs only new
candidates are evaluated.

Usage:
    python scripts/score_pass_at_k.py \\
        --samples_path results/pass_at_k/model_name/temp=1.0/samples_shard_*.jsonl \\
        --k_values 1 10 100 1000

    # With custom output directory
    python scripts/score_pass_at_k.py \\
        --samples_path results/pass_at_k/model_name/temp=1.0/samples.jsonl \\
        --k_values 1 10 100 \\
        --output_dir results/pass_at_k/model_name/temp=1.0/
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from math_verify import parse

import src.data
from src.scoring import extract_boxed_answer, score_response

# math-verify logs a "Timeout during comparison" line for every hard parse; on this data
# that is thousands of lines of noise around a handful of real results.
logging.getLogger("math_verify").setLevel(logging.ERROR)

# The exact substring `score_response` short-circuits on. Anything without it is scored
# incorrect without calling math-verify.
BOXED_MARKER = "\\boxed{"


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
        nargs="+",
        required=True,
        help="One or more samples.jsonl files produced by generate_pass_at_k_samples.py. "
        "Shards of a single model/temperature are pooled into one result.",
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
        help="Directory for output files. Defaults to the directory of the first samples file.",
    )
    return parser.parse_args()


def scan_samples(
    samples_paths,
    boxed_candidates,
    counts_by_problem,
    metadata_by_problem,
    gold_answer_by_problem,
    lenient_counts_by_problem,
):
    """Stream every sample file once, filling the accumulators in place.

    `counts_by_problem[pid]` counts every sample seen. `boxed_candidates` collects only the
    responses that could possibly score correct under the strict criterion, keyed by
    (problem_idx, sample_idx).

    `lenient_counts_by_problem[pid]` counts responses in which the ground-truth answer
    string appears *anywhere*, ignoring formatting entirely. That is a deliberately
    over-generous upper bound on capability: it counts a response correct if the answer
    shows up by coincidence in unparseable text. It exists so that a strict score of zero
    cannot be dismissed as an artifact of the `\\boxed{}` requirement.
    """
    for samples_path in samples_paths:
        n_lines = 0
        with open(samples_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pid = record["problem_idx"]
                counts_by_problem[pid] += 1
                n_lines += 1
                if pid not in metadata_by_problem:
                    metadata_by_problem[pid] = {
                        "level": record["level"],
                        "type": record["type"],
                    }
                response_text = record["response_text"]
                if BOXED_MARKER in response_text:
                    boxed_candidates[(pid, record["sample_idx"])] = response_text
                gold_answer = gold_answer_by_problem.get(pid)
                if gold_answer and gold_answer in response_text:
                    lenient_counts_by_problem[pid] += 1
        print(f"  {samples_path}: {n_lines} samples")


def main():
    args = parse_args()

    samples_paths = [Path(p) for p in args.samples_path]
    for samples_path in samples_paths:
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")

    output_dir = Path(args.output_dir) if args.output_dir else samples_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MATH ground-truth solutions for scoring.
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    ground_truth_solutions = test_dataset["solution"]

    # 2. Stream every samples file, counting all samples but retaining only the responses
    #    that contain a \boxed{} (the only ones that can score correct).
    counts_by_problem = defaultdict(int)  # problem_idx -> n samples seen
    metadata_by_problem = {}  # problem_idx -> {"level": ..., "type": ...}
    boxed_candidates = {}  # (problem_idx, sample_idx) -> response_text
    lenient_counts_by_problem = defaultdict(int)  # problem_idx -> n containing gold answer

    # Ground-truth answers for the lenient upper bound, taken from the gold solution's own
    # \boxed{}. Problems whose reference solution has no \boxed{} are skipped rather than
    # matched against something weaker.
    gold_answer_by_problem = {}
    for pid, solution in enumerate(ground_truth_solutions):
        gold_answer = extract_boxed_answer(solution)
        if gold_answer:
            gold_answer_by_problem[pid] = gold_answer

    print(f"Streaming {len(samples_paths)} samples file(s)...")
    scan_samples(
        samples_paths,
        boxed_candidates,
        counts_by_problem,
        metadata_by_problem,
        gold_answer_by_problem,
        lenient_counts_by_problem,
    )

    total_samples = sum(counts_by_problem.values())
    print(
        f"Loaded {total_samples} samples across {len(counts_by_problem)} problems.\n"
        f"{len(boxed_candidates)} samples "
        f"({100.0 * len(boxed_candidates) / max(total_samples, 1):.3f}%) contain a "
        f"\\boxed{{}} and require math-verify; the rest are incorrect by construction.\n"
        f"{sum(lenient_counts_by_problem.values())} samples contain the ground-truth "
        f"answer string anywhere at all (lenient upper bound)."
    )

    # 3. Load cached verdicts for boxed candidates (if any).
    scores_path = samples_paths[0].with_name("scores.jsonl")
    cached_scores = {}  # (problem_idx, sample_idx) -> bool
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cached_scores[(record["problem_idx"], record["sample_idx"])] = record[
                    "correct"
                ]
        print(f"Loaded {len(cached_scores)} cached scores from {scores_path}")

    # 4. Score the boxed candidates, grouping by problem so each gold solution is parsed once.
    candidates_by_problem = defaultdict(list)  # problem_idx -> [(sample_idx, text), ...]
    for (pid, sample_idx), response_text in boxed_candidates.items():
        candidates_by_problem[pid].append((sample_idx, response_text))

    n_correct_by_problem = defaultdict(int)
    n_new_scores = 0
    with open(scores_path, "a") as f_scores:
        for problem_number, pid in enumerate(sorted(candidates_by_problem.keys())):
            gold_parsed = parse(ground_truth_solutions[pid])
            for sample_idx, response_text in candidates_by_problem[pid]:
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
                if correct:
                    n_correct_by_problem[pid] += 1

            if (problem_number + 1) % 100 == 0:
                print(
                    f"  Scored candidates for {problem_number + 1}/"
                    f"{len(candidates_by_problem)} problems..."
                )

    print(
        f"Scored {n_new_scores} new candidates. "
        f"Total candidates: {len(boxed_candidates)}. "
        f"Correct: {sum(n_correct_by_problem.values())}."
    )

    # 5. Compute pass@k per problem.
    rows = []
    for pid in sorted(counts_by_problem.keys()):
        n_total = counts_by_problem[pid]
        n_correct = n_correct_by_problem[pid]
        row = {
            "problem_idx": pid,
            "level": metadata_by_problem[pid]["level"],
            "type": metadata_by_problem[pid]["type"],
            "n_total": n_total,
            "n_correct": n_correct,
            "n_lenient_correct": lenient_counts_by_problem[pid],
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
        "n_lenient_correct": ("n_lenient_correct", "sum"),
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
        "n_lenient_correct": df_for_agg["n_lenient_correct"].sum(),
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
        "n_lenient_correct": ("n_lenient_correct", "sum"),
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
        f.write(
            "**Samples files:** "
            + ", ".join(f"`{p}`" for p in args.samples_path)
            + "\n\n"
        )
        f.write(f"**k values:** {', '.join(str(k) for k in args.k_values)}\n\n")
        f.write(
            f"**Samples scored:** {total_samples} across {len(counts_by_problem)} "
            f"problems; {len(boxed_candidates)} contained a `\\boxed{{}}`.\n\n"
        )
        f.write(
            "**`n_lenient_correct`** counts samples containing the ground-truth answer "
            "string anywhere in the response, ignoring formatting. It is a deliberately "
            "over-generous upper bound on capability, reported so that a strict score of "
            "zero cannot be attributed to the `\\boxed{}` requirement.\n\n"
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
