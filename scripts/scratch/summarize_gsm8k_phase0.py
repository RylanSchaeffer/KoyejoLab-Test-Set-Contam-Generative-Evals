"""Summarize Phase 0 GSM8K evaluation runs from W&B into a per-checkpoint table.

Phase 0 of docs/ICLR_2027_CHECKLIST.md measures the clean (uncontaminated) GSM8K
capability floor. The eval scripts log per-problem history only and never an
aggregate, so accuracy has to be computed here by averaging `math_verify_score`
over each run's history.

Also reports `has_boxed` and the fraction of responses carrying GSM8K's "####"
marker, because a 0.00 accuracy has two very different causes: the model cannot
solve the problems, or the model cannot produce a parseable answer at all. Those
call for different write-ups.

Usage:
    uv run python scripts/scratch/summarize_gsm8k_phase0.py --group phase0-gsm8k
"""

import argparse
import os
import re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import wandb

import src.scoring

WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"

NAME_RE = re.compile(
    r"mem_Qwen3-(?P<size>[\d.]+[MB])_.*?_rep_(?P<rep>[\d.]+)_"
    r"sbst_(?P<sbst>[\d.]+)_epch_[\d.]+_ot_(?P<ot>[\d.]+)(?P<sft>_sft)?$"
)

SIZE_ORDER = {
    "34M": 34e6,
    "48M": 48e6,
    "62M": 62e6,
    "63M": 63e6,
    "93M": 93e6,
    "153M": 153e6,
    "165M": 165e6,
    "262M": 262e6,
    "344M": 344e6,
    "499M": 499e6,
    "660M": 660e6,
    "934M": 934e6,
    "1.44B": 1.44e9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    api = wandb.Api(timeout=600)
    runs = list(
        api.runs(
            f"{api.default_entity}/{WANDB_PROJECT}",
            filters={"group": args.group, "state": "finished"},
            per_page=200,
        )
    )
    print(
        f"entity={api.default_entity} group={args.group}: {len(runs)} finished runs\n"
    )

    records = []
    for run in runs:
        try:
            model = run.config["model_config"]["model"]
            temperature = float(run.config["temperature"])
            dataset = run.config["data_config"]["dataset"]
            num_fewshot = run.config.get("num_fewshot")
        except (KeyError, TypeError, ValueError):
            print(f"  skipping {run.id}: unreadable config")
            continue

        # scan_history, not history: run.history() SAMPLES to ~500 points, which
        # silently turns an exact count over 1,209 problems into an estimate.
        #
        # Pull the raw response and solution text too, and RESCORE with the current
        # scorer rather than trusting the logged `math_verify_score`. The logged value
        # reflects whichever scorer version happened to run; the scorer was tightened
        # on 2026-08-01 after a degenerate model looping on a few-shot demonstration
        # scored a coincidental match. Rescoring needs no GPU -- this is the same
        # pattern as scripts/rescore_zeroshot_with_boxed_required.py.
        rows = list(
            run.scan_history(
                keys=[
                    "problem_idx",
                    "math_verify_score",
                    "has_boxed",
                    "response",
                    "solution",
                ],
                page_size=2000,
            )
        )
        history = pd.DataFrame(rows)
        if history.empty or "math_verify_score" not in history:
            print(f"  skipping {run.id} ({model}): empty history")
            continue
        # W&B pagination can duplicate rows (see CLAUDE.md on the 5,001-row footnote).
        history = history.drop_duplicates(subset=["problem_idx"])

        if {"response", "solution"}.issubset(history.columns):
            history["rescored"] = [
                int(
                    src.scoring.score_gsm8k_response(
                        gold_answer=src.scoring.extract_gsm8k_gold_answer(
                            solution or ""
                        ),
                        response_text=response or "",
                    )
                )
                for solution, response in zip(history["solution"], history["response"])
            ]
            history["has_hash"] = [
                int("####" in (response or "")) for response in history["response"]
            ]
        else:
            history["rescored"] = history["math_verify_score"]
            history["has_hash"] = float("nan")

        match = NAME_RE.search(model.split("/", 1)[1])
        fields = match.groupdict() if match else {}
        records.append(
            {
                "model": model,
                "size": fields.get("size", "?"),
                "ot": float(fields["ot"]) if fields.get("ot") else float("nan"),
                "sft": bool(fields.get("sft")),
                "replicas": float(fields["rep"]) if fields.get("rep") else float("nan"),
                "dataset": dataset,
                "num_fewshot": num_fewshot,
                "temperature": temperature,
                "n_problems": len(history),
                "logged_acc": history["math_verify_score"].mean(),
                "accuracy": history["rescored"].mean(),
                "n_correct": int(history["rescored"].sum()),
                "boxed_rate": (
                    history["has_boxed"].mean()
                    if "has_boxed" in history
                    else float("nan")
                ),
                "hash_rate": history["has_hash"].mean(),
            }
        )

    if not records:
        print("No runs with usable history.")
        return

    df = pd.DataFrame(records)
    df["size_num"] = df["size"].map(SIZE_ORDER).fillna(0.0)
    df = df.sort_values(["sft", "size_num", "ot", "temperature"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 72)
    print(
        df[
            [
                "model",
                "size",
                "ot",
                "sft",
                "temperature",
                "n_problems",
                "n_correct",
                "accuracy",
                "logged_acc",
                "hash_rate",
                "boxed_rate",
            ]
        ].to_string(index=False)
    )

    print("\n=== Headline (accuracy = rescored with the current scorer) ===")
    print(f"checkpoints evaluated : {df['model'].nunique()}")
    print(f"total problems scored : {df['n_problems'].sum():,}")
    print(f"total correct         : {df['n_correct'].sum():,}")
    print(f"max accuracy any cell : {df['accuracy'].max():.4f}")
    print(f"mean accuracy         : {df['accuracy'].mean():.4f}")
    # The format rate is the other half of the story: it separates "cannot answer"
    # from "cannot answer in a parseable form". On MATH, 4-shot lifted the boxed rate
    # to 0.43-0.89 while accuracy stayed at exactly 0.0000.
    print(f"mean '####' rate      : {df['hash_rate'].mean():.4f}")
    print(f"max  '####' rate      : {df['hash_rate'].max():.4f}")
    disagree = df[df["logged_acc"] != df["accuracy"]]
    if not disagree.empty:
        print(
            f"\n{len(disagree)} cells changed under rescoring "
            f"(logged scorer was looser):"
        )
        print(disagree[["model", "logged_acc", "accuracy"]].to_string(index=False))
    nonzero = df[df["accuracy"] > 0]
    print(f"cells with accuracy>0 : {len(nonzero)} of {len(df)}")
    if not nonzero.empty:
        print("\nNon-zero cells:")
        print(
            nonzero[
                ["model", "size", "ot", "sft", "temperature", "accuracy"]
            ].to_string(index=False)
        )

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
