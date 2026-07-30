"""Print the measured Original / Rephrased / Perturbed numbers behind Table 1, at 4-shot.

Table 1 reports rephrased and perturbed scores but no original column, so the reader supplies
the baseline from Fig. 1 — which is 0-shot and saturates near 1.0. The runs in Table 1 are
4-shot, where the original set scores far lower, so the implied collapse is much larger than
the one actually measured. This prints all three conditions from the same protocol so the
real effect size can be quoted instead of inferred.
"""

import re

import pandas as pd
import wandb

WANDB_ENTITY = "rylan"
WANDB_PROJECT = "memorization-scoring-vs-sampling-eval"

SWEEPS = {
    "Original": "mprek7pj",
    "Perturbed": "w8j3qnru",
    "Rephrased": "25xeednq",
}


def main() -> None:
    api = wandb.Api(timeout=600)
    records = []

    for condition, sweep_id in SWEEPS.items():
        sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")
        for run in sweep.runs:
            if run.state != "finished":
                continue
            try:
                model_name = run.config["model_config"]["model"]
                temperature = float(run.config["temperature"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(temperature) > 1e-6:
                continue

            n_rows = 0
            n_correct = 0
            for row in run.scan_history(keys=["math_verify_score"]):
                n_rows += 1
                if row.get("math_verify_score"):
                    n_correct += 1
            if n_rows == 0:
                continue

            replicas = re.search(r"rep_(\d+)_sbst", model_name)
            records.append(
                {
                    "Condition": condition,
                    "Num. Replicas": int(replicas.group(1)) if replicas else None,
                    "math_verify_score": n_correct / n_rows,
                    "n_problems": n_rows,
                }
            )
            print(
                f"  {condition:<10} R={records[-1]['Num. Replicas']:<5} "
                f"acc={records[-1]['math_verify_score']:.4f}"
            )

    df = pd.DataFrame(records)
    pivot = (
        df.pivot_table(
            index="Num. Replicas", columns="Condition", values="math_verify_score"
        )
        .reindex(columns=["Original", "Rephrased", "Perturbed"])
        .reset_index()
    )
    print("\n=== 344M, greedy, 4-shot: measured percentages ===")
    percent = pivot.copy()
    for column in ("Original", "Rephrased", "Perturbed"):
        if column in percent:
            percent[column] = (100 * percent[column]).round(3)
    print(percent.to_markdown(index=False))

    out_path = "table1_measured_4shot.csv"
    pivot.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
