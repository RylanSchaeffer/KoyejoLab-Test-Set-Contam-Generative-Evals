"""Quantify how much of the measured contamination effect depends on the eval prompt format.

The same pretrained checkpoint can score Math Verify 1.0000 or 0.0052 depending only on
whether the prompt carries a 4-shot prefix. Under 0-shot the prompt reproduces the opening
of the memorized training document, so a contaminated model regurgitates the stored solution
verbatim; prepending four unrelated worked examples moves the prompt off that memorized
context and the regurgitation does not fire.

That makes the protocol a first-class experimental variable rather than an implementation
detail, and it has to be reported as one: `notebooks/11_*` (which feeds the manuscript's
Finding #1 figure) reads a cache built from the 0-shot sweeps, while `notebooks/13_*` (SFT)
reads 4-shot sweeps. Comparing across those two notebooks compares protocols, not stages.

This script pulls both protocols' greedy-decoding runs, writes a side-by-side table, and
plots accuracy against contamination level with one line per protocol.

Usage:
    python scripts/compare_zeroshot_vs_fewshot_protocol.py \\
        --output-dir notebooks/11_math_qwen3_pt_math_verify/results
"""

import argparse
import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.colors import LogNorm

import src.analyze
import src.globals
import src.plot

WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"
WANDB_ENTITY = "rylan"

# Superseded 0-shot sweeps. These are what notebook 11's cache was actually built from, and
# therefore what the manuscript's Finding #1 figure shows.
ZERO_SHOT_SWEEPS = [
    "6y9dy2ow",  # 34M
    "lnrpy3ed",  # 34M, more temperatures
    "5oo55o9s",  # 62M
    "10q465ij",  # 62M, more temperatures
    "q5uoy1eu",  # 93M
    "f5djvfth",  # 93M, more temperatures
    "vnz1h147",  # 153M
    "xkzfmbhk",  # 153M
    "39rugx2e",  # 344M, more temperatures
]

# 4-shot boxed-required sweeps, the protocol the notebooks currently declare.
FEW_SHOT_SWEEPS = [
    "qx2c4702",  # 34M
    "dkiui6we",  # 62M
    "cx8y41bw",  # 93M
    "4w5x8hez",  # 153M
    "mprek7pj",  # 344M
]


def summarize_sweeps(sweep_ids, protocol_label: str, temperature: float) -> pd.DataFrame:
    """Return one row per run: model size, replicas, and mean Math Verify score.

    Uses each run's summary/history mean rather than re-downloading full per-problem
    histories, which for these sweeps runs to tens of GB.
    """
    api = wandb.Api(timeout=600)
    records = []
    for sweep_id in sweep_ids:
        try:
            sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        except Exception as e:
            print(f"  sweep {sweep_id}: {e}")
            continue
        for run in sweep.runs:
            if run.state != "finished":
                continue
            try:
                model_name = run.config["model_config"]["model"]
                run_temperature = float(run.config["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(run_temperature - temperature) > 1e-6:
                continue
            if model_name.endswith("_sft"):
                continue

            n_rows = 0
            n_correct = 0
            n_boxed_like = 0
            for row in run.scan_history(keys=["math_verify_score"]):
                n_rows += 1
                if row.get("math_verify_score"):
                    n_correct += 1
            if n_rows == 0:
                continue

            parameters = re.search(r"Qwen3-([\d.]+[MB])", model_name)
            replicas = re.search(r"rep_(\d+)_sbst", model_name)
            overtrain = re.search(r"ot_([\d.]+)", model_name)
            records.append(
                {
                    "protocol": protocol_label,
                    "sweep": sweep_id,
                    "run_id": run.id,
                    "Model": model_name,
                    "Parameters": parameters.group(1) if parameters else None,
                    "Num. Replicas": int(replicas.group(1)) if replicas else None,
                    "Overtrain Multiplier": float(overtrain.group(1)) if overtrain else None,
                    "Temp.": round(run_temperature, 4),
                    "n_problems": n_rows,
                    "math_verify_score": n_correct / n_rows,
                }
            )
            print(
                f"  [{protocol_label}] {records[-1]['Parameters']:>5} "
                f"R={records[-1]['Num. Replicas']:<5} "
                f"acc={records[-1]['math_verify_score']:.4f}"
            )
    return pd.DataFrame(records)


def plot_comparison(df: pd.DataFrame, results_dir: str) -> None:
    """One panel per protocol, accuracy vs contamination, coloured by model size."""
    df = df.dropna(subset=["Parameters", "Num. Replicas"]).copy()
    df["Num. Parameters"] = df["Parameters"].map(
        src.globals.MODEL_NAMES_TO_PARAMETERS_DICT
    )
    df = df.dropna(subset=["Num. Parameters"])

    num_parameters_log_norm = LogNorm(
        vmin=df["Num. Parameters"].min(), vmax=df["Num. Parameters"].max()
    )

    protocols = ["0-shot", "4-shot"]
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)
    for ax, protocol in zip(axes, protocols):
        subset = df[df["protocol"] == protocol]
        g = sns.lineplot(
            data=subset,
            x="Num. Replicas",
            y="math_verify_score",
            hue="Num. Parameters",
            hue_norm=num_parameters_log_norm,
            palette="flare",
            marker="o",
            legend="full" if protocol == protocols[0] else False,
            ax=ax,
        )
        g.set(
            xscale="symlog",
            xlim=(-0.1, 3500),
            xlabel="Num. MATH Test Set Replicas",
            ylabel="Math Verify Score" if protocol == protocols[0] else "",
            ylim=(-0.02, 1.05),
        )
        ax.set_title(protocol)
        if protocol == protocols[0]:
            ax.legend(loc="upper left", title="Num. Parameters")
            src.plot.format_g_legend_to_millions_and_billions(g=ax)

    plt.tight_layout()
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename="y=math_verify_x=num_replicas_hue=params_col=protocol",
    )
    plt.close()


def write_report(df: pd.DataFrame, results_dir: str, temperature: float) -> None:
    pivot = df.pivot_table(
        index=["Parameters", "Num. Replicas"],
        columns="protocol",
        values="math_verify_score",
    ).reset_index()

    for protocol in ("0-shot", "4-shot"):
        if protocol not in pivot.columns:
            pivot[protocol] = np.nan

    pivot["ratio (0-shot / 4-shot)"] = pivot["0-shot"] / pivot["4-shot"].replace(0, np.nan)
    pivot = pivot.sort_values(["Parameters", "Num. Replicas"])

    peak = df.groupby("protocol")["math_verify_score"].max()
    lines = [
        "# Evaluation Protocol Determines the Measured Contamination Effect",
        "",
        f"Greedy decoding (temperature = {temperature}), pretrained (`ot=1`) checkpoints only.",
        "",
        "## Headline",
        "",
        "| Protocol | Peak Math Verify across the whole grid |",
        "|---|---|",
    ]
    for protocol, value in peak.items():
        lines.append(f"| {protocol} | {value:.4f} |")
    lines += [
        "",
        "Under 0-shot prompting, heavily contaminated checkpoints reproduce the memorized",
        "solution verbatim and saturate. Prepending four worked examples moves the prompt off",
        "the memorized context, and the same checkpoints score near the uncontaminated floor.",
        "",
        "## Why this matters for the manuscript",
        "",
        "- `notebooks/11_*` declares the 4-shot sweep IDs but its cached data was built from",
        "  the 0-shot list (confirmed by reproducing the cache's md5 filename). The figure the",
        "  manuscript uses for Finding #1 is therefore 0-shot.",
        "- `notebooks/13_*` (SFT) reads 4-shot sweeps. Comparing the pretrained figure against",
        "  the SFT figure compares protocols as well as training stages, so the apparent",
        "  'SFT collapses accuracy' effect is confounded with the protocol change.",
        "- Any claim about contamination magnitude must state the protocol it was measured under.",
        "",
        "## Per-condition comparison",
        "",
        pivot.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]

    report_path = os.path.join(results_dir, "PROTOCOL_SENSITIVITY.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {report_path}")
    print(pivot.to_markdown(index=False, floatfmt=".4f"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="notebooks/11_math_qwen3_pt_math_verify/results"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Summarizing 0-shot sweeps...")
    zero_shot = summarize_sweeps(ZERO_SHOT_SWEEPS, "0-shot", args.temperature)
    print("Summarizing 4-shot sweeps...")
    few_shot = summarize_sweeps(FEW_SHOT_SWEEPS, "4-shot", args.temperature)

    df = pd.concat([zero_shot, few_shot], ignore_index=True)
    if df.empty:
        raise SystemExit("No runs summarized.")

    csv_path = os.path.join(args.output_dir, "protocol_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    write_report(df, args.output_dir, args.temperature)
    plot_comparison(df, args.output_dir)


if __name__ == "__main__":
    main()
