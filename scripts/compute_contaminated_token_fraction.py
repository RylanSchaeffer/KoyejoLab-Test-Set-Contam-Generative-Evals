"""Report what fraction of each model's pretraining tokens are MATH test set replicas.

Reviewer aPBL calls the contamination setup contrived. The honest answer is arithmetic: state
what fraction of the training budget the injected replicas actually occupy, at each replica
count and model size, and let the number speak. It also bounds where the setup stops
resembling realistic leakage and starts resembling training on the test set.

Chinchilla-style budgets make this size-dependent in a way that is easy to miss: tokens seen
is `20 * overtrain_multiplier * num_parameters`, so the *same* replica count is a far larger
share of a 34M model's budget than of a 344M model's.

`create_dataset_for_pretraining` holds total tokens per epoch **fixed** and lets replicas
displace corpus tokens, raising `ValueError` if the replicated benchmark would exceed the
budget. So the contaminated fraction is bounded by 100% by construction, and that bound is
what produces the ragged replica ladders in `docs/EXPERIMENT_INVENTORY.md`: a 34M model at
`ot = 1` stops at R = 316 because R = 1000 simply would not fit, while the same model at
`ot = 8` reaches R = 3162. To avoid reporting configurations that cannot exist, this script
enumerates the checkpoints actually on the Hub and computes fractions only for those.

The MATH test set's token count is measured with the actual tokenizer rather than taken from
the 1.5e6 constant hardcoded in `src/analyze.py`.

Usage:
    python scripts/compute_contaminated_token_fraction.py
"""

import argparse
import os
import re

import pandas as pd
from huggingface_hub import HfApi
from transformers import AutoTokenizer

import src.data

# Sizes that appear in the manuscript.
MODEL_PARAMETERS = {
    "34M": 34e6,
    "62M": 62e6,
    "93M": 93e6,
    "153M": 153e6,
    "344M": 344e6,
}

CHECKPOINT_RE = re.compile(
    r"mem_Qwen3-(?P<size>[\d.]+M)_minerva_math_rep_(?P<rep>\d+)_sbst_"
    r"(?P<sbst>[\d.]+)_epch_(?P<epch>\d+)_ot_(?P<ot>[\d.]+)$"
)


def enumerate_checkpoints() -> pd.DataFrame:
    """List the contamination checkpoints on the Hub as (size, replicas, subset, overtrain).

    Only fully-pretrained, non-SFT checkpoints at `sbst = 1.0` are returned, since those are
    the ones the replica ladders in the manuscript refer to.
    """
    api = HfApi()
    rows = []
    for model in api.list_models(author="RylanSchaeffer", search="mem_Qwen3"):
        name = model.id.split("/")[-1]
        match = CHECKPOINT_RE.match(name)
        if match is None:
            continue  # SFT checkpoints and other conventions
        if abs(float(match.group("sbst")) - 1.0) > 1e-9:
            continue
        size = match.group("size")
        if size not in MODEL_PARAMETERS:
            continue
        rows.append(
            {
                "Parameters": size,
                "Num. Replicas": int(match.group("rep")),
                "Num. Epochs": int(match.group("epch")),
                "Overtrain Multiplier": float(match.group("ot")),
            }
        )
    return pd.DataFrame(rows).drop_duplicates()

TOKENIZER = "RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"


def measure_benchmark_tokens() -> int:
    """Token count of one full copy of the MATH test set, as injected during pretraining.

    Documents are injected in the same `Problem:\\n{problem}\\n\\nSolution: {solution}` form the
    evaluation uses, so the prompt scaffolding counts toward the contaminated budget.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    test_dataset = src.data.load_dataset_hendrycks_math()["test"]
    documents = [
        src.data.MINERVA_MATH_DOC_TO_TEXT.format(problem=problem, solution=solution)
        for problem, solution in zip(
            test_dataset["problem"], test_dataset["solution"]
        )
    ]
    return sum(len(ids) for ids in tokenizer(documents).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reviews/2026_neurips/data")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_tokens = measure_benchmark_tokens()
    print(
        f"One copy of the MATH test set = {benchmark_tokens:,} tokens "
        f"(src/analyze.py assumes 1,500,000)."
    )

    df = enumerate_checkpoints()
    print(f"{len(df)} pretrained checkpoints (sbst=1.0) enumerated from the Hub.")

    df["training_tokens"] = (
        20.0
        * df["Overtrain Multiplier"]
        * df["Parameters"].map(MODEL_PARAMETERS)
    )
    df["contaminated_tokens"] = (
        df["Num. Replicas"] * df["Num. Epochs"] * benchmark_tokens
    )
    df["contaminated_fraction"] = df["contaminated_tokens"] / df["training_tokens"]
    df = df.sort_values(["Parameters", "Overtrain Multiplier", "Num. Replicas"])

    over_budget = df[df["contaminated_fraction"] > 1.0]
    if not over_budget.empty:
        print(
            f"\nWARNING: {len(over_budget)} checkpoints compute to >100% contaminated, "
            f"which create_dataset_for_pretraining should have rejected. Investigate:"
        )
        print(over_budget.to_string(index=False))
    csv_path = os.path.join(args.output_dir, "contaminated_token_fraction.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    compute_optimal = df[df["Overtrain Multiplier"] == 1]
    pivot = compute_optimal.pivot_table(
        index="Num. Replicas", columns="Parameters", values="contaminated_fraction"
    )
    pivot = pivot[[s for s in MODEL_PARAMETERS if s in pivot.columns]]

    max_by_size = (
        df.groupby(["Parameters", "Overtrain Multiplier"])["contaminated_fraction"]
        .max()
        .reset_index()
        .pivot_table(
            index="Overtrain Multiplier", columns="Parameters", values="contaminated_fraction"
        )
    )
    max_by_size = max_by_size[[s for s in MODEL_PARAMETERS if s in max_by_size.columns]]

    print("\n=== Contaminated fraction of the training budget, compute-optimal (ot=1) ===")
    print((100 * pivot).round(2).to_markdown(floatfmt=".2f"))
    print("\n=== Largest contaminated fraction reached at each (size, overtrain) ===")
    print((100 * max_by_size).round(2).to_markdown(floatfmt=".2f"))

    lines = [
        "# Contaminated Fraction of the Pretraining Budget",
        "",
        f"One copy of the MATH test set, tokenized as injected, is **{benchmark_tokens:,} tokens**.",
        "Training budget is `20 x overtrain_multiplier x num_parameters` (Chinchilla-optimal at",
        "`ot = 1`), so the same replica count occupies a very different share of a 34M budget",
        "than of a 344M budget.",
        "",
        "Only configurations that exist as checkpoints on the Hub are shown. Combinations that",
        "would exceed the token budget were rejected at dataset-construction time and were never",
        "trained, which is why the replica ladders are ragged and why a given model reaches higher",
        "replica counts only at higher overtraining multipliers.",
        "",
        "## Percent of training tokens that are MATH test set, at `ot = 1`",
        "",
        (100 * pivot).round(2).to_markdown(floatfmt=".2f"),
        "",
        "## Largest contaminated fraction reached at each (size, overtrain multiplier)",
        "",
        (100 * max_by_size).round(2).to_markdown(floatfmt=".2f"),
        "",
        "## How to use this in the rebuttal",
        "",
        "- **Low replica counts bracket realistic leakage from below.** At `R = 1` the test set is",
        "  a fraction of a percent of the budget for every model size — comparable to or below",
        "  published estimates of real-world benchmark leakage, and the paper measures effects there.",
        "- **High replica counts are deliberately extreme, and should be described that way.** At",
        "  the top of the ladder the injected replicas are a large share of the smaller models'",
        "  budgets. That is a feature — it upper-bounds the effect — but it is not 'realistic",
        "  leakage' and claiming so invites exactly aPBL's objection.",
        "- The honest framing is a **dose-response curve spanning from below-realistic to",
        "  saturating**, with the interesting science in where the transition happens.",
        "",
        "## Caveat on `Num. Tokens` for overtrained checkpoints",
        "",
        "`src/analyze.py:75` computes pretraining `Num. Tokens` as",
        "`20 * overtrain_multiplier * num_parameters`, which is right. But the **eval-side**",
        "computation in `notebooks/11_*` (`Num. Tokens = 20 * Num. Parameters`) omits the",
        "overtrain multiplier. That is harmless for the `ot = 1` runs it was written for, and",
        "**wrong for the overtrained checkpoints** — it would understate their compute by up to",
        "16x and misplace every point on a FLOP axis. Fix before plotting the overtraining results.",
        "",
    ]
    report_path = os.path.join(args.output_dir, "CONTAMINATED_TOKEN_FRACTION.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
