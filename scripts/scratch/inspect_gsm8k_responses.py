"""Print sample model responses from a Phase 0 GSM8K run to diagnose a zero score.

A 0.00 accuracy has two very different causes, and the manuscript wording depends
on which one it is:
  (a) the model produces coherent attempts but gets the arithmetic wrong, or
  (b) the model cannot produce a parseable answer at all (no "####", no \\boxed{},
      possibly not even on-topic text).

Only (a) supports any statement about capability. This script shows the raw text
so the distinction is made by looking rather than by assuming.

Usage:
    uv run python scripts/scratch/inspect_gsm8k_responses.py --group phase0-gsm8k-smoke
"""

import argparse
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import wandb

import src.scoring

WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=700)
    args = parser.parse_args()

    api = wandb.Api(timeout=600)
    runs = list(
        api.runs(
            f"{api.default_entity}/{WANDB_PROJECT}",
            filters={"group": args.group, "state": "finished"},
            per_page=200,
        )
    )
    if not runs:
        print(f"No finished runs in group {args.group!r}")
        return

    for run in runs[:1]:
        model = run.config["model_config"]["model"]
        print(f"=== {model}  (tau={run.config.get('temperature')}) ===\n")
        rows = list(
            run.scan_history(
                keys=["problem_idx", "solution", "response", "math_verify_score"],
                page_size=200,
            )
        )
        print(f"{len(rows)} logged problems\n")

        marker_counts = {"hash": 0, "boxed": 0, "neither": 0, "empty": 0}
        for row in rows:
            response = row.get("response") or ""
            if not response.strip():
                marker_counts["empty"] += 1
            elif "####" in response:
                marker_counts["hash"] += 1
            elif src.scoring.extract_boxed_answer(response) is not None:
                marker_counts["boxed"] += 1
            else:
                marker_counts["neither"] += 1
        total = max(len(rows), 1)
        print("Answer-format census across all responses:")
        for key, count in marker_counts.items():
            print(f"  {key:>8}: {count:>5} ({count / total:.1%})")
        print()

        for row in rows[: args.num_examples]:
            response = (row.get("response") or "")[: args.max_chars]
            solution = (row.get("solution") or "")[:300]
            gold = src.scoring.extract_gsm8k_gold_answer(row.get("solution") or "")
            predicted = src.scoring.extract_gsm8k_predicted_answer(
                row.get("response") or ""
            )
            print(f"--- problem {row.get('problem_idx')} ---")
            print(f"GOLD ANSWER   : {gold!r}")
            print(f"PARSED PRED   : {predicted!r}")
            print(f"SCORE         : {row.get('math_verify_score')}")
            print(f"GOLD SOLUTION : {solution!r}")
            print(f"RESPONSE      : {response!r}")
            print()


if __name__ == "__main__":
    main()
