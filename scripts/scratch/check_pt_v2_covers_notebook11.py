"""Can `-pt-v2` supply the pretraining losses notebook 11 needs, under different sweep IDs?

Notebook 11's 15 pretraining sweep IDs are gone from every visible W&B project, so the sweeps
themselves are unrecoverable. But notebook 11 does not actually need those sweep IDs — it needs
`eval_after/eval_benchmark_loss` for each (model size, replica count) configuration, in order to
merge cross-entropy against Math Verify. If `memorization-scoring-vs-sampling-pt-v2` covers the
same grid, the analysis is recoverable and the local cache stops being a single point of failure.

Compares the configurations present in -pt-v2 against those in notebook 11's cached configs.
"""

import ast

import pandas as pd
import wandb

CACHE = (
    "notebooks/11_math_qwen3_pt_math_verify/data/"
    "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv"
)
PT_V2 = "rylan/memorization-scoring-vs-sampling-pt-v2"
LOSS_KEY = "eval_after/eval_benchmark_loss"


def as_dict(value):
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}


def describe(configs: pd.DataFrame, label: str) -> set:
    rows = set()
    for _, row in configs.iterrows():
        data_config = as_dict(row.get("data_config"))
        model_config = as_dict(row.get("model_config"))
        replicas = data_config.get("num_benchmark_replicas_per_epoch")
        subset = data_config.get("benchmark_subset_fraction")
        model = model_config.get("model_name") or model_config.get("model")
        if replicas is None or model is None:
            continue
        rows.add((str(model), int(replicas), float(subset) if subset is not None else None))
    print(f"{label}: {len(rows)} distinct (model, replicas, subset) configurations")
    return rows


def main() -> None:
    cached = pd.read_csv(CACHE, low_memory=False)
    print(f"Cached notebook-11 pretraining configs: {len(cached)} runs")
    has_loss = LOSS_KEY in cached.columns
    print(f"  cache has {LOSS_KEY}: {has_loss}")
    if has_loss:
        print(f"  non-null loss values: {cached[LOSS_KEY].notna().sum()}")
    cached_configs = describe(cached, "cache")

    api = wandb.Api(timeout=600)
    records = []
    for run in api.runs(PT_V2, per_page=200):
        records.append(
            {
                "data_config": run.config.get("data_config"),
                "model_config": run.config.get("model_config"),
                "state": run.state,
                LOSS_KEY: run.summary._json_dict.get(LOSS_KEY),
                "sweep": run.sweep.id if run.sweep is not None else None,
            }
        )
    v2 = pd.DataFrame(records)
    print(f"\n{PT_V2}: {len(v2)} runs")
    print(f"  finished: {(v2['state'] == 'finished').sum()}")
    print(f"  with {LOSS_KEY}: {v2[LOSS_KEY].notna().sum()}")
    print(f"  sweeps: {sorted(set(v2['sweep'].dropna()))}")
    v2_configs = describe(v2[v2["state"] == "finished"], "pt-v2")

    overlap = cached_configs & v2_configs
    print(f"\nConfigurations in both: {len(overlap)}")
    print(f"Only in cache (would be lost if cache is deleted): {len(cached_configs - v2_configs)}")
    print(f"Only in pt-v2: {len(v2_configs - cached_configs)}")

    missing = sorted(cached_configs - v2_configs)
    if missing:
        print("\nSample of configurations the cache holds that pt-v2 does not:")
        for row in missing[:15]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
