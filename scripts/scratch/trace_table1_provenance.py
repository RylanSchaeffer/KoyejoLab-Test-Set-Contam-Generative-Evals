"""Find every W&B eval run against the rephrased/perturbed MATH sets, at any model size.

Table 1 of `manuscript_neurips_2026/03_pretraining.tex` reports 34M, 93M and 344M columns,
but `docs/EXPERIMENT_INVENTORY.md` records rephrased/perturbed coverage as 344M-only.
Reviewer 8RFz asks directly how Table 1 is computed, so the question has to be settled
against W&B rather than against prose.

This scans the whole eval project (not just the three sweeps notebook 15 reads) for runs
whose eval dataset is a rephrased or perturbed variant, and reports what model sizes and
replica levels actually exist, in any run state.
"""

import ast
import re
from collections import defaultdict

import pandas as pd
import wandb

WANDB_ENTITY = "rylan"
PROJECTS = [
    "memorization-scoring-vs-sampling-eval",
    "memorization-scoring-vs-sampling-eval-teacher-forcing",
]
MODIFIED_DATASETS = {
    "RylanSchaeffer/math_perturbed",
    "RylanSchaeffer/math_rephrased",
    "stellaathena/math_perturbed",
    "stellaathena/math_rephrased",
}


def as_dict(value):
    """W&B returns nested config either as a dict or as its repr; accept both."""
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}


def main() -> None:
    api = wandb.Api(timeout=600)
    records = []

    for project in PROJECTS:
        try:
            runs = api.runs(f"{WANDB_ENTITY}/{project}", per_page=200)
        except Exception as e:
            print(f"Skipping {project}: {e}")
            continue

        n_scanned = 0
        for run in runs:
            n_scanned += 1
            data_config = as_dict(run.config.get("data_config"))
            dataset = data_config.get("dataset")
            if dataset not in MODIFIED_DATASETS:
                continue
            model_config = as_dict(run.config.get("model_config"))
            model_name = model_config.get("model", "")
            parameters = re.search(r"Qwen3-([\d.]+[MB])", model_name)
            replicas = re.search(r"rep_(\d+)_sbst", model_name)
            records.append(
                {
                    "project": project,
                    "sweep": run.sweep.id if run.sweep is not None else None,
                    "run_id": run.id,
                    "state": run.state,
                    "created": str(run.created_at),
                    "dataset": dataset,
                    "Parameters": parameters.group(1) if parameters else None,
                    "Num. Replicas": int(replicas.group(1)) if replicas else None,
                    "temperature": run.config.get("temperature"),
                    "model": model_name,
                }
            )
        print(f"{project}: scanned {n_scanned} runs")

    if not records:
        print("No runs found against any rephrased/perturbed dataset.")
        return

    df = pd.DataFrame(records)
    out_path = "table1_provenance_runs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} matching runs)\n")

    print("=== Runs by dataset x model size x state ===")
    print(
        df.groupby(["dataset", "Parameters", "state"])
        .size()
        .reset_index(name="n_runs")
        .to_markdown(index=False)
    )

    print("\n=== Sweeps involved ===")
    print(
        df.groupby(["sweep", "dataset", "Parameters"])
        .size()
        .reset_index(name="n_runs")
        .to_markdown(index=False)
    )

    print("\n=== Replica coverage per (dataset, size) ===")
    coverage = defaultdict(set)
    for _, row in df.iterrows():
        coverage[(row["dataset"], row["Parameters"])].add(row["Num. Replicas"])
    for key in sorted(coverage, key=lambda k: (str(k[0]), str(k[1]))):
        values = sorted(v for v in coverage[key] if v is not None)
        print(f"  {key[0]:<34} {str(key[1]):>6}: {values}")


if __name__ == "__main__":
    main()
