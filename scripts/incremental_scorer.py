"""Incremental scorer that runs alongside pass@k generation.

Periodically reads sample shard files (read-only), scores any new samples
with math-verify, appends scores to a cache file, and prints a pass@k
summary. Designed to run safely alongside the generation script.

Safety:
    - ONLY READS sample shard files (never writes to them)
    - Scores are written to a separate scores.jsonl file
    - Handles partial/incomplete lines from in-progress generation
    - Sole writer to scores.jsonl (do not run multiple instances)

Usage:
    nohup python scripts/incremental_scorer.py \
        --results_dir results/pass_at_k/MODEL/temp=1.0/ \
        --poll_interval 300 \
        > logs/pass_at_k/scorer.log 2>&1 &
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from math_verify import parse

import src.data
from src.scoring import extract_boxed_answer, score_response


def pass_at_k(n: int, c: int, k: int):
    """Unbiased estimator of pass@k (Chen et al., 2021)."""
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_cached_scores(scores_path: Path) -> dict:
    """Load existing scores from cache. Returns {(pid, sid): bool}."""
    cached = {}
    if scores_path.exists():
        with open(scores_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cached[(record["problem_idx"], record["sample_idx"])] = record[
                        "correct"
                    ]
                except (json.JSONDecodeError, KeyError):
                    continue
    return cached


def read_all_samples(results_dir: Path) -> dict:
    """Read all samples from shard files. Returns {pid: [(sid, text, level, type), ...]}."""
    samples = defaultdict(list)
    for shard_file in sorted(results_dir.glob("samples_shard_*.jsonl")):
        with open(shard_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    samples[record["problem_idx"]].append(
                        (
                            record["sample_idx"],
                            record["response_text"],
                            record["level"],
                            record["type"],
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    # Skip partial/corrupt lines from in-progress writes
                    continue
    return samples


def score_new_samples(samples, cached_scores, scores_path, ground_truth):
    """Score any samples not in cache. Appends new scores to scores_path."""
    n_new = 0
    n_skipped = 0

    with open(scores_path, "a") as f_scores:
        for pid in sorted(samples.keys()):
            gold_parsed = parse(ground_truth[pid])
            for sample_idx, response_text, _, _ in samples[pid]:
                cache_key = (pid, sample_idx)
                if cache_key in cached_scores:
                    n_skipped += 1
                    continue
                correct = score_response(gold_parsed, response_text)
                cached_scores[cache_key] = correct
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
                n_new += 1

            # Flush after each problem for safety
            f_scores.flush()

    return n_new, n_skipped


def print_summary(samples, cached_scores, k_values):
    """Compute and print pass@k summary from cached scores."""
    rows_by_level = defaultdict(lambda: {"n_problems": 0, "pass_at_k": defaultdict(list)})
    total_samples = 0
    total_correct = 0
    total_scored = 0

    for pid in sorted(samples.keys()):
        level = None
        n_total = 0
        n_correct = 0
        for sample_idx, _, lvl, _ in samples[pid]:
            level = lvl
            cache_key = (pid, sample_idx)
            if cache_key in cached_scores:
                n_total += 1
                total_scored += 1
                if cached_scores[cache_key]:
                    n_correct += 1

        if n_total == 0 or level is None:
            continue

        total_samples += n_total
        total_correct += n_correct
        rows_by_level[level]["n_problems"] += 1
        for k in k_values:
            pk = pass_at_k(n_total, n_correct, k)
            if pk is not None:
                rows_by_level[level]["pass_at_k"][k].append(pk)

    print(f"\n{'Level':<10} {'Problems':>8} {'pass@' + ' pass@'.join(str(k) for k in k_values)}")
    print("-" * 60)
    for level in sorted(rows_by_level.keys()):
        info = rows_by_level[level]
        pk_strs = []
        for k in k_values:
            vals = info["pass_at_k"][k]
            if vals:
                pk_strs.append(f"{sum(vals)/len(vals):.6f}")
            else:
                pk_strs.append("N/A")
        print(f"{level:<10} {info['n_problems']:>8}   {'   '.join(pk_strs)}")

    print(f"\nTotal scored: {total_scored}, Total correct: {total_correct} ({100*total_correct/max(total_scored,1):.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Incremental scorer for pass@k generation."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing samples_shard_*.jsonl files.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=300,
        help="Seconds between scoring passes (default: 300).",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 10, 100, 1000],
        help="k values for pass@k (default: 1 10 100 1000).",
    )
    parser.add_argument(
        "--one_shot",
        action="store_true",
        help="Score once and exit (don't poll).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    scores_path = results_dir / "scores.jsonl"

    # Load ground truth
    raw_datasets = src.data.load_dataset_hendrycks_math()
    ground_truth = raw_datasets["test"]["solution"]

    # Load existing score cache
    cached_scores = load_cached_scores(scores_path)
    print(f"{datetime.now()}: Loaded {len(cached_scores)} cached scores.")

    while True:
        # Read current samples from all shards
        samples = read_all_samples(results_dir)
        total_samples = sum(len(v) for v in samples.values())
        print(f"\n{datetime.now()}: Found {total_samples} samples across {len(samples)} problems.")

        # Score new samples
        n_new, n_skipped = score_new_samples(
            samples, cached_scores, scores_path, ground_truth
        )
        print(f"  Scored {n_new} new samples ({n_skipped} already cached).")

        # Print summary
        print_summary(samples, cached_scores, args.k_values)

        if args.one_shot:
            break

        print(f"\n  Sleeping {args.poll_interval}s before next pass...")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
