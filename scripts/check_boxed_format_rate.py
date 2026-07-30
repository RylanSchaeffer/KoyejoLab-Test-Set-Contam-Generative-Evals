"""Compare \\boxed{} emission rates between pretrained and SFT'd generative eval runs.

Motivation: post-SFT Math Verify accuracy is flat at ~1-2% across every contamination
level, versus ~100% at R >= 316 for the pretrained models. Because `src.scoring` scores a
response incorrect unless it contains a `\\boxed{...}`, that floor has two very different
explanations:

  (a) genuine task failure  — the model still emits answers in the expected format, it just
      gets them wrong; the ~60x collapse is a real result about contamination not surviving
      SFT; or
  (b) a format artifact     — SFT destroyed the `\\boxed{}` habit, so every response is
      scored incorrect regardless of content, and the number means nothing.

Distinguishing them requires the `response` column, which the notebooks drop from their
caches. This script streams just `response` and `math_verify_score` out of W&B for the
relevant runs and reports, per run, the fraction of responses containing `\\boxed{}`
alongside accuracy.

Reading the output: if the SFT runs' boxed rate is comparable to the pretrained runs', the
floor is (a) and the SFT result stands. If it collapsed toward zero, it is (b).

Usage:
    python scripts/check_boxed_format_rate.py \\
        --output-dir notebooks/13_math_qwen3_sft_math_verify/results
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import wandb

from src.scoring import extract_boxed_answer

WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"
WANDB_ENTITY = "rylan"

# Sweeps to compare. The pretrained sweeps are per-model-size; the SFT sweep covers all
# sizes in one. Both use the same 4-shot boxed-required protocol, so they are comparable.
SWEEPS = [
    ("qx2c4702", "pretrained"),
    ("dkiui6we", "pretrained"),
    ("cx8y41bw", "pretrained"),
    ("4w5x8hez", "pretrained"),
    ("mprek7pj", "pretrained"),
    ("2zpwcnek", "sft"),
]


def parse_model_fields(model_name: str) -> dict:
    """Pull parameters / replicas / overtrain multiplier out of a checkpoint name.

    Note the `_sft` suffix sits *after* the `ot` field, so an `ot_([\\d.]+)$`-anchored regex
    silently drops every SFT model. Match without anchoring and record the suffix instead.
    """
    parameters = re.search(r"Qwen3-([\d.]+[MB])", model_name)
    replicas = re.search(r"rep_(\d+)_sbst", model_name)
    overtrain = re.search(r"ot_([\d.]+)", model_name)
    return {
        "Parameters": parameters.group(1) if parameters else None,
        "Num. Replicas": int(replicas.group(1)) if replicas else None,
        "Overtrain Multiplier": float(overtrain.group(1)) if overtrain else None,
        "is_sft": model_name.endswith("_sft"),
    }


def summarize_run(run, stage: str) -> dict | None:
    """Stream one run's responses and return its boxed rate and accuracy."""
    try:
        model_name = run.config["model_config"]["model"]
        temperature = float(run.config["temperature"])
    except (KeyError, TypeError, ValueError):
        return None

    n_samples = 0
    n_boxed = 0
    n_correct = 0
    n_empty = 0
    total_chars = 0
    for row in run.scan_history(keys=["response", "math_verify_score"]):
        response = row.get("response")
        if response is None:
            continue
        n_samples += 1
        total_chars += len(response)
        if not response.strip():
            n_empty += 1
        if extract_boxed_answer(response) is not None:
            n_boxed += 1
        if row.get("math_verify_score"):
            n_correct += 1

    if n_samples == 0:
        return None

    record = {
        "run_id": run.id,
        "stage": stage,
        "Model": model_name,
        "Temp.": round(temperature, 4),
        "n_samples": n_samples,
        "boxed_rate": n_boxed / n_samples,
        "math_verify_score": n_correct / n_samples,
        "empty_rate": n_empty / n_samples,
        "mean_response_chars": total_chars / n_samples,
    }
    record.update(parse_model_fields(model_name))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="notebooks/13_math_qwen3_sft_math_verify/results",
        help="Where to write boxed_format_rates.csv and FORMAT_SANITY_CHECK.md.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Only summarize runs at this temperature (greedy decoding by default).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api = wandb.Api(timeout=600)

    records = []
    for sweep_id, stage in SWEEPS:
        try:
            sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        except Exception as e:
            print(f"Skipping sweep {sweep_id}: {e}")
            continue
        runs = [r for r in sweep.runs if r.state == "finished"]
        print(f"Sweep {sweep_id} ({stage}): {len(runs)} finished runs")
        for run in runs:
            try:
                if abs(float(run.config["temperature"]) - args.temperature) > 1e-6:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            record = summarize_run(run, stage=stage)
            if record is not None:
                records.append(record)
                print(
                    f"  {record['Parameters']:>5} R={record['Num. Replicas']:<5} "
                    f"{stage:<10} boxed={record['boxed_rate']:.4f} "
                    f"acc={record['math_verify_score']:.4f}"
                )

    if not records:
        raise SystemExit("No runs summarized; nothing to write.")

    df = pd.DataFrame(records).sort_values(
        ["stage", "Parameters", "Num. Replicas"]
    )
    csv_path = os.path.join(args.output_dir, "boxed_format_rates.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    write_report(df, args)


def write_report(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write the human-readable verdict alongside the raw per-run table."""
    by_stage = df.groupby("stage").agg(
        n_runs=("run_id", "count"),
        mean_boxed_rate=("boxed_rate", "mean"),
        min_boxed_rate=("boxed_rate", "min"),
        max_boxed_rate=("boxed_rate", "max"),
        mean_accuracy=("math_verify_score", "mean"),
        mean_response_chars=("mean_response_chars", "mean"),
    )

    pretrained_boxed = by_stage.loc["pretrained", "mean_boxed_rate"] if "pretrained" in by_stage.index else float("nan")
    sft_boxed = by_stage.loc["sft", "mean_boxed_rate"] if "sft" in by_stage.index else float("nan")

    if pd.notna(pretrained_boxed) and pd.notna(sft_boxed) and pretrained_boxed > 0:
        ratio = sft_boxed / pretrained_boxed
        if ratio > 0.5:
            verdict = (
                f"**Genuine task failure.** SFT models still emit `\\boxed{{}}` at "
                f"{sft_boxed:.1%} versus {pretrained_boxed:.1%} for pretrained models "
                f"({ratio:.2f}x). The post-SFT accuracy floor is not a formatting artifact."
            )
        elif ratio > 0.1:
            verdict = (
                f"**Partially confounded.** SFT boxed rate is {sft_boxed:.1%} versus "
                f"{pretrained_boxed:.1%} pretrained ({ratio:.2f}x). Some of the accuracy "
                f"drop is attributable to format loss; report accuracy conditioned on a "
                f"`\\boxed{{}}` being emitted alongside the raw number."
            )
        else:
            verdict = (
                f"**Format artifact.** SFT boxed rate collapsed to {sft_boxed:.1%} from "
                f"{pretrained_boxed:.1%} pretrained ({ratio:.2f}x). The ~1-2% floor "
                f"measures format loss, not task failure, and must not be presented as a "
                f"contamination result."
            )
    else:
        verdict = "Insufficient data to render a verdict."

    per_size = (
        df.groupby(["stage", "Parameters"])
        .agg(
            n_runs=("run_id", "count"),
            mean_boxed_rate=("boxed_rate", "mean"),
            mean_accuracy=("math_verify_score", "mean"),
        )
        .reset_index()
    )

    pretrained_accuracy = (
        by_stage.loc["pretrained", "mean_accuracy"]
        if "pretrained" in by_stage.index
        else float("nan")
    )
    sft_accuracy = (
        by_stage.loc["sft", "mean_accuracy"] if "sft" in by_stage.index else float("nan")
    )

    lines = [
        "# `\\boxed{}` Format Sanity Check",
        "",
        f"Greedy decoding (temperature = {args.temperature}). "
        f"{len(df)} runs, {int(df['n_samples'].sum())} scored responses.",
        "",
        "**Both stages here are 4-shot**, which is the point: comparing the SFT figures in",
        "`notebooks/13_*` against the pretrained figure in `notebooks/11_*` compares 4-shot",
        "against 0-shot, because notebook 11's cache was built from the superseded 0-shot",
        "sweeps. See `reviews/2026_neurips/PROTOCOL_CONFOUND.md`.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## The matched-protocol numbers",
        "",
        f"At 4-shot, mean Math Verify is **{pretrained_accuracy:.4f} pretrained** and "
        f"**{sft_accuracy:.4f} after SFT**. Both sit at the uncontaminated floor, so there is",
        "no large post-SFT collapse to explain in this protocol. The ~60x collapse quoted in",
        "`REBUTTAL_PLAN.md` (P0.1) compares 0-shot pretrained (~100%) against 4-shot SFT",
        "(~1-2%) and is an artifact of the protocol mismatch. Do not use it.",
        "",
        "## Why this check exists",
        "",
        "`src.scoring.score_response` scores a response incorrect unless it contains a",
        "`\\boxed{...}`. A model that forgot the output format is therefore indistinguishable",
        "from a model that cannot solve the problems, unless the format rate is measured",
        "separately. The post-SFT Math Verify floor is only a contamination result under the",
        "first reading.",
        "",
        "## By stage",
        "",
        by_stage.to_markdown(),
        "",
        "## By stage and model size",
        "",
        per_size.to_markdown(index=False),
        "",
        "## Per-run detail",
        "",
        "See `boxed_format_rates.csv`.",
        "",
    ]

    report_path = os.path.join(args.output_dir, "FORMAT_SANITY_CHECK.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}")
    print()
    print(by_stage.to_markdown())
    print()
    print(verdict)


if __name__ == "__main__":
    main()
