"""Which part of a leaked document carries the contamination effect?

Reviewers 1wx9 (W1/Q1), aPBL (Q1) and the AC (bullet 1) all press the same point: exact-replica
contamination is a clean causal testbed but not the dominant realistic leakage mode. This
measures the paraphrased case directly.

DESIGN -- three arms, see reviews/2026_neurips/CONTAMINANT_ABLATION.md
------------------------------------------------------------------
34M models pretrained at R = 32, 100, 316 with different contaminants injected, while
`eval_after/eval_benchmark_loss` is always measured on the ORIGINAL `EleutherAI/minerva_math`
test set.

    arm         problem    solution   sweep       isolates
    exact       same       same       published   full verbatim leakage
    rephrased   differs    SAME       mxamktp0    solution-only leakage
    perturbed   differs    differs    vrxwx4dz    no verbatim leakage

The rephrased arm is NOT a paraphrase condition: `math_rephrased` keeps the original solution in
99.8% of rows (4991/5000), and the loss is measured on solution text. Treating it as
"paraphrased contamination" would badly overstate paraphrase transfer. `math_perturbed` differs
on both sides (4/5000 identical solutions) and is the realistic-leakage arm.

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
ARM_BY_CONTAMINANT = {
    "RylanSchaeffer/math_rephrased": "rephrased",
    "RylanSchaeffer/math_perturbed": "perturbed",
}
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
                "arm": ARM_BY_CONTAMINANT.get(dc.get("contaminant"), "unknown"),
                "loss": loss,
                "grad_accum": run.config.get("trainer_config", {}).get(
                    "gradient_accumulation_steps"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    exact = published_exact_replica()
    runs = paraphrased_runs()
    if runs.empty:
        raise SystemExit("No finished contaminant runs with a benchmark loss yet.")

    # The v1 optimizer config must have taken effect or nothing here is comparable to Fig. 3.
    bad = runs[runs["grad_accum"] != 9]
    if len(bad):
        print(
            f"!! {len(bad)} run(s) have gradient_accumulation_steps != 9 (the published value); "
            f"these are NOT comparable to the published exact-replica losses:\n{bad}\n"
        )

    floor = float(exact.loc[exact.R == 0, "exact"].iloc[0])
    df = runs.merge(exact, on="R", how="left").sort_values(["arm", "R"])
    df["uncontaminated"] = floor
    # Share of the exact-replica loss reduction that this contaminant achieves.
    df["transfer_fraction"] = (floor - df["loss"]) / (floor - df["exact"])
    df.to_csv(os.path.join(RESULTS, "contaminant_ablation.csv"), index=False)

    wide = df.pivot_table(index="R", columns="arm", values="loss")
    for col in ("rephrased", "perturbed"):
        if col not in wide:
            wide[col] = float("nan")
    wide = wide.join(exact.set_index("R")["exact"]).sort_index()

    plt.close()
    plt.figure(figsize=src.plot.default_figsize)
    ax = plt.gca()
    ax.axhline(floor, color="0.4", linestyle=":", label="Uncontaminated ($R=0$)")
    ax.plot(wide.index, wide["exact"], marker="o", label="Exact replicas (problem $=$, solution $=$)")
    ax.plot(wide.index, wide["rephrased"], marker="s",
            label=r"Rephrased (problem $\neq$, solution $=$)")
    ax.plot(wide.index, wide["perturbed"], marker="^",
            label=r"Perturbed (problem $\neq$, solution $\neq$)")
    ax.set_xscale("log")
    ax.set(xlabel="Num. MATH Test Set Replicas",
           ylabel="Benchmark Cross-Entropy (original test set)")
    ax.legend(loc="upper right")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=RESULTS, plot_filename="y=benchmark_loss_x=num_replicas_hue=contaminant"
    )
    plt.close()

    with open(os.path.join(RESULTS, "CONTAMINANT_ABLATION.md"), "w") as f:
        f.write(
            "# Which part of a leaked document carries the effect?\n\n"
            "Qwen3-34M, 1xOT. Cross-entropy is always measured on the **original** "
            "`EleutherAI/minerva_math` test set; only the injected contaminant changes. Run from "
            "`scripts/pretrain_language_model_v1.py`, which reproduces the published "
            "(pre-`934546a`) optimizer configuration, so the published exact-replica runs serve "
            "as the control without retraining.\n\n"
            "| Arm | Problem | Solution | Isolates |\n|---|---|---|---|\n"
            "| Exact | same | same | full verbatim leakage |\n"
            "| Rephrased | differs | **same** (99.8% identical) | solution-only leakage |\n"
            "| Perturbed | differs | differs (0.1% identical) | no verbatim leakage |\n\n"
            "⚠️ The rephrased arm is **not** a paraphrase condition — `math_rephrased` keeps the "
            "original solution, and the loss is measured on solution text. See "
            "`reviews/2026_neurips/CONTAMINANT_ABLATION.md`.\n\n"
            f"Uncontaminated baseline (R = 0): **{floor:.4f}**\n\n"
            "| R | Exact | Rephrased | Perturbed | Transfer: rephrased | Transfer: perturbed |\n"
            "|---|---|---|---|---|---|\n"
        )
        for R in wide.index:
            ex = wide.loc[R, "exact"]
            row = [f"| {int(R)} ", f"| {ex:.4f} " if pd.notna(ex) else "| — "]
            tf = {}
            for arm in ("rephrased", "perturbed"):
                v = wide.loc[R, arm]
                row.append(f"| {v:.4f} " if pd.notna(v) else "| — ")
                tf[arm] = (
                    (floor - v) / (floor - ex)
                    if pd.notna(v) and pd.notna(ex) and (floor - ex) != 0
                    else float("nan")
                )
            for arm in ("rephrased", "perturbed"):
                row.append(f"| {tf[arm]:.3f} " if pd.notna(tf[arm]) else "| — ")
            f.write("".join(row) + "|\n")
        f.write(
            "\n`Transfer` = (L(R=0) - L_arm) / (L(R=0) - L_exact): the share of the "
            "exact-replica loss reduction the arm achieves. 1.0 means as damaging as verbatim "
            "leakage; 0.0 means it buys nothing.\n\n"
            "## Caveat to state in the paper\n\n"
            "Both modified corpora are still MATH-domain text with MATH-style solutions, so part "
            "of any reduction is **domain adaptation** rather than item-level leakage. The R = 0 "
            "baseline saw no mathematics at all and so does not separate the two. A clean "
            "separation needs a fourth arm contaminated with *disjoint* math problems; that is "
            "not run here. Treat the perturbed number as an upper bound on realistic-leakage "
            "transfer.\n"
        )
    print(wide.to_string())
    print(f"\nWrote {RESULTS}/CONTAMINANT_ABLATION.md")


if __name__ == "__main__":
    main()
