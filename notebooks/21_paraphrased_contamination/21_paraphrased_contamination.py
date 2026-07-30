"""Does contamination with *paraphrased* benchmark items transfer to the original test set?

Reviewers 1wx9 (W1/Q1), aPBL (Q1) and the AC (bullet 1) all press the same point: exact-replica
contamination is a clean causal testbed but not the dominant realistic leakage mode. This
measures the paraphrased case directly.

DESIGN
------
Three 34M models pretrained with `RylanSchaeffer/math_rephrased` injected at R = 32, 100, 316,
while `eval_after/eval_benchmark_loss` is measured on the ORIGINAL `EleutherAI/minerva_math` test
set. So the loss answers exactly the question asked: does contamination with paraphrase i lower
loss on original i?

The control arm is not retrained. `scripts/pretrain_language_model_v1.py` reproduces the
published (pre-934546a) optimizer configuration, and every other config value was read out of the
published runs' recorded configs, so the exact-replica runs at the same three doses ARE the
control. Their losses come from the notebook-11 cache.

READ THE CAVEAT in the generated report before quoting the transfer fraction: the paraphrased
corpus is still MATH-domain text, so part of any loss reduction is domain adaptation rather than
item-level leakage. The R=0 baseline saw no math at all and therefore does not separate these.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      ./mem_scoring_vs_sampling_env/bin/python \
      notebooks/21_paraphrased_contamination/21_paraphrased_contamination.py
"""

import ast
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.plot  # noqa: E402  (applies the project's global style on import)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(
    REPO,
    "notebooks/11_math_qwen3_pt_math_verify/data",
    "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv",
)
PROJECT = "rylan/memorization-scoring-vs-sampling-pt-paraphrased"
LOSS = "eval_after/eval_benchmark_loss"
os.makedirs(RESULTS, exist_ok=True)


def published_exact_replica() -> pd.DataFrame:
    """34M, ot=1, subset=1.0 exact-replica benchmark losses, from the surviving cache."""
    df = pd.read_csv(CACHE, low_memory=False)
    rows = []
    for _, r in df.iterrows():
        try:
            mc = ast.literal_eval(r["model_config"])
            dc = ast.literal_eval(r["data_config"])
            tc = ast.literal_eval(r["trainer_config"])
        except Exception:
            continue
        if (
            mc.get("model_name") == "Qwen3/Qwen3-34M"
            and tc.get("overtrain_multiplier") in (1, 1.0)
            and dc.get("benchmark_subset_fraction") == 1
            and pd.notna(r.get(LOSS))
        ):
            rows.append({"R": dc["num_benchmark_replicas_per_epoch"], "exact": r[LOSS]})
    return pd.DataFrame(rows).groupby("R", as_index=False)["exact"].mean()


def paraphrased_runs() -> pd.DataFrame:
    api = wandb.Api(timeout=90)
    rows = []
    for run in api.runs(PROJECT):
        if run.state != "finished":
            print(f"  skipping {run.id} (state={run.state})")
            continue
        dc = run.config.get("data_config", {})
        loss = run.summary.get(LOSS)
        if loss is None:
            print(f"  skipping {run.id} (no {LOSS} in summary)")
            continue
        rows.append(
            {
                "run_id": run.id,
                "R": dc.get("num_benchmark_replicas_per_epoch"),
                "contaminant": dc.get("contaminant"),
                "paraphrased": loss,
                "grad_accum": run.config.get("trainer_config", {}).get(
                    "gradient_accumulation_steps"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    exact = published_exact_replica()
    para = paraphrased_runs()
    if para.empty:
        raise SystemExit("No finished paraphrased runs with a benchmark loss yet.")

    # Sanity: the v1 config must have taken effect, or the comparison is invalid.
    bad = para[para["grad_accum"] != 9]
    if len(bad):
        print(f"!! {len(bad)} run(s) have gradient_accumulation_steps != 9 (published value); "
              f"these are NOT comparable to Fig. 3:\n{bad}")

    floor = float(exact.loc[exact.R == 0, "exact"].iloc[0])
    df = para.merge(exact, on="R", how="left").sort_values("R")
    df["uncontaminated"] = floor
    # Fraction of the exact-replica loss reduction that paraphrased contamination achieves.
    df["transfer_fraction"] = (floor - df["paraphrased"]) / (floor - df["exact"])
    df.to_csv(os.path.join(RESULTS, "paraphrased_vs_exact.csv"), index=False)

    plt.close()
    plt.figure(figsize=src.plot.default_figsize)
    ax = plt.gca()
    ax.axhline(floor, color="0.4", linestyle=":", label="Uncontaminated ($R=0$)")
    ax.plot(df["R"], df["exact"], marker="o", label="Exact replicas (published)")
    ax.plot(df["R"], df["paraphrased"], marker="s", label="Paraphrased replicas")
    ax.set_xscale("log")
    ax.set(xlabel="Num. MATH Test Set Replicas", ylabel="Benchmark Cross-Entropy (original test set)")
    ax.legend(loc="lower left")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=RESULTS, plot_filename="y=benchmark_loss_x=num_replicas_hue=contaminant"
    )
    plt.close()

    with open(os.path.join(RESULTS, "PARAPHRASED_CONTAMINATION.md"), "w") as f:
        f.write(
            "# Paraphrased contamination during pretraining\n\n"
            "Qwen3-34M, 1xOT, `RylanSchaeffer/math_rephrased` injected into the corpus; "
            "cross-entropy measured on the **original** `EleutherAI/minerva_math` test set. "
            "The exact-replica column is the published run at the same dose, reproduced under "
            "the same (pre-`934546a`) optimizer configuration, so the contaminant is the only "
            "variable.\n\n"
            f"Uncontaminated baseline (R=0): **{floor:.4f}**\n\n"
            "| R | Exact replicas | Paraphrased | Transfer fraction |\n|---|---|---|---|\n"
        )
        for _, r in df.iterrows():
            tf = "—" if pd.isna(r["transfer_fraction"]) else f"{r['transfer_fraction']:.3f}"
            f.write(
                f"| {int(r['R'])} | {r['exact']:.4f} | {r['paraphrased']:.4f} | {tf} |\n"
            )
        f.write(
            "\n`Transfer fraction` = (L(R=0) - L_paraphrased) / (L(R=0) - L_exact): the share of "
            "the exact-replica loss reduction that paraphrased contamination achieves. 1.0 would "
            "mean paraphrase is as good as verbatim leakage; 0.0 would mean it buys nothing.\n\n"
            "## Caveat to state in the paper\n\n"
            "The paraphrased corpus is still MATH-domain text, so part of any reduction is "
            "domain adaptation rather than item-level leakage. The R=0 baseline saw no "
            "mathematics at all and so does not separate the two. A clean separation needs a "
            "third arm contaminated with *disjoint* math problems; that is the natural "
            "follow-up and is not claimed here.\n"
        )
    print(df.to_string(index=False))
    print(f"\nWrote {RESULTS}/PARAPHRASED_CONTAMINATION.md")


if __name__ == "__main__":
    main()
