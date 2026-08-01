"""Show every response that survives the CURRENT scorer in a Phase 0 run.

inspect_gsm8k_responses.py --only-correct filters on the *logged* score, which was
produced by whichever scorer version ran at eval time. After the 2026-08-01
tightening, the interesting question is different: which responses does the current,
stricter scorer still credit? Those are the only candidates for genuine capability,
and each one has to be read before it is believed.

Usage:
    uv run python scripts/scratch/inspect_surviving_correct.py \
        --group phase0-gsm8k-4shot --model-contains ot_8.000
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
    parser.add_argument("--model-contains", default=None)
    parser.add_argument("--max-chars", type=int, default=600)
    args = parser.parse_args()

    api = wandb.Api(timeout=600)
    runs = list(
        api.runs(
            f"{api.default_entity}/{WANDB_PROJECT}",
            filters={"group": args.group, "state": "finished"},
            per_page=200,
        )
    )
    if args.model_contains:
        runs = [
            r
            for r in runs
            if args.model_contains in r.config.get("model_config", {}).get("model", "")
        ]

    total_surviving = 0
    for run in runs:
        model = run.config["model_config"]["model"]
        rows = list(
            run.scan_history(
                keys=[
                    "problem_idx",
                    "solution",
                    "response",
                    "math_verify_score",
                    "token_per_response",
                ],
                page_size=2000,
            )
        )
        surviving = []
        for row in rows:
            gold = src.scoring.extract_gsm8k_gold_answer(row.get("solution") or "")
            if src.scoring.score_gsm8k_response(gold, row.get("response") or ""):
                surviving.append((row, gold))
        if not surviving:
            continue
        total_surviving += len(surviving)
        print(f"=== {model} ===")
        print(f"{len(surviving)} of {len(rows)} responses survive the current scorer\n")
        for row, gold in surviving:
            print(f"--- problem {row.get('problem_idx')} ---")
            print(f"GOLD ANSWER : {gold!r}")
            print(
                f"PARSED PRED : "
                f"{src.scoring.extract_gsm8k_predicted_answer(row.get('response') or '')!r}"
            )
            response = row.get("response") or ""
            print(f"TOKENS      : {row.get('token_per_response')}")
            tail_after_marker = response.rsplit("####", 1)[-1]
            print(f"AFTER LAST '####': {tail_after_marker[:80]!r}")
            print(f"RESPONSE TAIL: {response[-160:]!r}")
            print(f"RESPONSE    : {response[: args.max_chars]!r}")
            print()

    print(f"TOTAL surviving across inspected runs: {total_surviving}")


if __name__ == "__main__":
    main()
