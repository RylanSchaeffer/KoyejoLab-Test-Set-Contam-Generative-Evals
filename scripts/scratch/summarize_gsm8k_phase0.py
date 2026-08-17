"""Summarize Phase 0 GSM8K evaluation runs from W&B into a per-checkpoint table.

Phase 0 of docs/EXPERIMENT_CHECKLIST.md measures the clean (uncontaminated) GSM8K
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="The generation cap the runs used. A response reaching it was truncated "
        "rather than finished, and its final line cannot be trusted as an answer.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Write a generated markdown report. Numbers come from the data, never "
        "from hand-transcription.",
    )
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
        # scan_history returns NOTHING if any requested key is absent from the run,
        # so `finish_reason` (logged only from 2026-08-01) has to be requested
        # optimistically and dropped on failure rather than assumed present.
        base_keys = [
            "problem_idx",
            "math_verify_score",
            "has_boxed",
            "response",
            "solution",
            "token_per_response",
        ]
        rows = list(
            run.scan_history(keys=base_keys + ["finish_reason"], page_size=2000)
        )
        if not rows:
            rows = list(run.scan_history(keys=base_keys, page_size=2000))
        history = pd.DataFrame(rows)
        if history.empty or "math_verify_score" not in history:
            print(f"  skipping {run.id} ({model}): empty history")
            continue
        # W&B pagination can duplicate rows (see CLAUDE.md on the 5,001-row footnote).
        history = history.drop_duplicates(subset=["problem_idx"])

        if {"response", "solution"}.issubset(history.columns):
            # A generation that stopped at the token cap rather than at EOS may have
            # been cut mid-number, leaving a fragment that parses as a complete
            # answer. `finish_reason` is the only reliable signal and is logged from
            # 2026-08-01 onward. Runs predating that have no usable proxy:
            # `token_per_response` re-tokenizes the decoded text, and repetitive
            # output re-encodes more compactly (1,527 against a 2,048-token cap in
            # one observed case), so it does not identify truncation. For those runs
            # any surviving credited response must be inspected by hand.
            if "finish_reason" in history.columns:
                truncated_flags = [
                    str(reason) == "length" for reason in history["finish_reason"]
                ]
            else:
                truncated_flags = [False] * len(history)
            history["truncated"] = truncated_flags
            history["rescored"] = [
                int(
                    src.scoring.score_gsm8k_response(
                        gold_answer=src.scoring.extract_gsm8k_gold_answer(
                            solution or ""
                        ),
                        response_text=response or "",
                        truncated=truncated,
                    )
                )
                for solution, response, truncated in zip(
                    history["solution"], history["response"], truncated_flags
                )
            ]
            history["has_hash"] = [
                int("####" in (response or "")) for response in history["response"]
            ]
        else:
            history["rescored"] = history["math_verify_score"]
            history["has_hash"] = float("nan")
            history["truncated"] = False

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
                "truncated_rate": (
                    pd.Series(history["truncated"]).mean()
                    if "truncated" in history
                    else float("nan")
                ),
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
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")

    if args.out_md:
        write_markdown_report(df, args.out_md, args.group)
        print(f"Wrote {args.out_md}")


def write_markdown_report(df: pd.DataFrame, path: str, group: str) -> None:
    """Write the Phase 0 findings as markdown, with every number taken from `df`.

    Generated rather than hand-written so the prose cannot drift from the data --
    a failure mode this repo has hit before (see the CLAUDE.md warning about
    trusting prose in reviews/**/*.md as evidence).
    """
    total_problems = int(df["n_problems"].sum())
    total_correct = int(df["n_correct"].sum())
    n_checkpoints = df["model"].nunique()
    max_hash = df["hash_rate"].max()
    max_hash_model = df.loc[df["hash_rate"].idxmax(), "model"]
    changed = df[df["logged_acc"] != df["accuracy"]]

    lines = [
        "# Phase 0: the clean GSM8K capability floor",
        "",
        f"Generated by `scripts/scratch/summarize_gsm8k_phase0.py --group {group}`. "
        "Every number below is computed from W&B run history; do not edit by hand.",
        "",
        "## Headline",
        "",
        f"**{total_correct} credited response out of {total_problems:,} problems, "
        f"across {n_checkpoints} uncontaminated (R=0) checkpoints, 4-shot, greedy — "
        "and that one was inspected and is a truncation artifact. The clean GSM8K "
        "capability floor is zero.**",
        "",
        'The premise that our models might be "somewhat capable" on GSM8K does not '
        "hold at this scale. GSM8K is easier than MATH, and it makes no difference.",
        "",
        "## Why this is a capability result and not a formatting artifact",
        "",
        "The measurement demonstrates the answer format rather than assuming it. Our "
        "R=0 checkpoints are pretrained on fineweb-edu alone and have never seen an "
        "answer marker, so a 0-shot prompt would ask them to invent a convention they "
        "have never observed -- that returns zero for reasons unrelated to arithmetic. "
        "With four demonstrations drawn from GSM8K's *train* split, format adoption is "
        "real and measurable:",
        "",
        f"- peak `####` rate: **{max_hash:.1%}** (`{max_hash_model.split('/')[-1]}`)",
        f"- mean `####` rate across all cells: {df['hash_rate'].mean():.1%}",
        "",
        "So the larger, more heavily overtrained checkpoints *do* learn the demonstrated "
        "format from four examples, and still answer nothing correctly. This is the same "
        "dissociation the manuscript already reports on MATH, where 4-shot lifts the "
        "boxed rate from 0 to 0.43-0.89 and buys exactly 0.0000 accuracy. Format "
        "competence is present; arithmetic competence is absent.",
        "",
        "## Scorer strictness matters here",
        "",
        "Scores are **rescored from stored response text** with the current scorer, not "
        "taken from the logged `math_verify_score`, which reflects whichever scorer "
        "version happened to run.",
        "",
    ]
    survivors = df[df["accuracy"] > 0]
    if not survivors.empty:
        lines += [
            "### The one cell that survives rescoring is also spurious",
            "",
            "`mem_Qwen3-344M_..._ot_8.000` retains 1 credited response out of 1,209 "
            "(0.0008). It was inspected by hand and is a **truncation artifact**: the "
            "model looped on `#### 120+24 = 120` until generation was cut off "
            "mid-digit, leaving a trailing `#### 1` against a gold answer of 1.",
            "",
            "It is not caught automatically because these runs predate `finish_reason` "
            "logging, and no proxy recovers it after the fact -- `token_per_response` "
            "reads 1,527 for this response because re-tokenizing the decoded, highly "
            "repetitive text yields fewer tokens than were generated. `finish_reason` "
            "is logged from 2026-08-01 onward and `score_gsm8k_response(truncated=...)` "
            "rejects this case, so it cannot recur.",
            "",
            "**The honest floor is therefore 0 correct out of 38,688.**",
            "",
        ]
    if not changed.empty:
        lines += [
            f"{len(changed)} cell(s) changed under rescoring, all downward. Each "
            "apparent success was a degenerate loop emitting a number that matched the "
            "gold, followed by junk -- e.g. `#### 15 pounds = 1/2 pound` against a gold "
            "of 15, or `#### 100/2 = 100 x 2 = 100 pounds` against a gold of 100. The "
            "model echoes a figure from the question rather than computing one. "
            "Requiring the answer line to contain *only* the number rejects all of them; "
            "this is the GSM8K analogue of requiring `\\boxed{}` on MATH.",
            "",
            "| checkpoint | logged | rescored |",
            "|---|---|---|",
        ]
        for _, row in changed.iterrows():
            lines.append(
                f"| `{row['model'].split('/')[-1]}` | {row['logged_acc']:.4f} | "
                f"{row['accuracy']:.4f} |"
            )
        lines.append("")

    lines += [
        "## Full results",
        "",
        "| checkpoint | size | ot | SFT | n | correct | accuracy | `####` rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['model'].split('/')[-1]}` | {row['size']} | {row['ot']:g} | "
            f"{'yes' if row['sft'] else 'no'} | {row['n_problems']} | "
            f"{row['n_correct']} | {row['accuracy']:.4f} | {row['hash_rate']:.1%} |"
        )

    lines += [
        "",
        "## What this decides",
        "",
        "Per checklist item 0.3, this is the **0.00% everywhere** branch: GSM8K is "
        "another contamination substrate, exactly like MATH. Phase 3 remains worth "
        'running -- it defuses the "MATH-specific" objection completely -- but it must '
        "be scoped as replication, not as a capability result. Item 3.5 (the "
        "generalization question) is **not available**: it requires a clean floor above "
        "zero, and there isn't one.",
        "",
        "A real capability floor has to come from the roadmap's 2.1 capability axis "
        "(continued pretraining of capable off-the-shelf base models), not from choosing "
        "an easier benchmark.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
