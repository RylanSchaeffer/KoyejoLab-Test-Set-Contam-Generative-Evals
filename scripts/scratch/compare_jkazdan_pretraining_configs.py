"""Why do none of jkazdan's 110 pretraining runs match the cached configurations?

`jkazdan/memorization-scoring-vs-sampling-pt` holds 236 runs, 110 of which parse as
contamination-pretraining configs — yet none matched a configuration in notebook 11's cache.
Either they are a different experiment, or they are the same experiment differing on some field
the match key includes (benchmark string, subset fraction, overtrain multiplier). If the latter,
the data may be recoverable after all.

Prints the raw distributions on both sides so the mismatch is attributable rather than guessed.
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
import wandb

CACHE = (
    "notebooks/11_math_qwen3_pt_math_verify/data/"
    "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv"
)
LOSS_COL = "eval_after/eval_benchmark_loss"
JKAZDAN = "jkazdan/memorization-scoring-vs-sampling-pt"


def as_dict(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def summarize(label, records):
    print(f"\n=== {label}  (n={len(records)})")
    for field in ("benchmark", "subset", "overtrain", "params_M", "replicas", "has_loss"):
        counts = Counter(r.get(field) for r in records)
        top = counts.most_common(10)
        print(f"  {field}: {top}")


def main() -> None:
    cache = pd.read_csv(CACHE, low_memory=False)
    cache_records = []
    for _, row in cache.iterrows():
        data_config = as_dict(row.get("data_config"))
        trainer_config = as_dict(row.get("trainer_config"))
        params = row.get("model/num_parameters")
        cache_records.append(
            {
                "benchmark": data_config.get("benchmark"),
                "subset": data_config.get("benchmark_subset_fraction"),
                "overtrain": trainer_config.get("overtrain_multiplier"),
                "params_M": (
                    int(round(float(params) / 1e6))
                    if params is not None and np.isfinite(pd.to_numeric(params, errors="coerce"))
                    else None
                ),
                "replicas": data_config.get("num_benchmark_replicas_per_epoch"),
                "has_loss": pd.notna(row.get(LOSS_COL)),
            }
        )
    summarize("notebook-11 cache", cache_records)

    api = wandb.Api(timeout=600)
    jk_records = []
    for run in api.runs(JKAZDAN, per_page=200):
        data_config = as_dict(run.config.get("data_config"))
        trainer_config = as_dict(run.config.get("trainer_config"))
        if not data_config:
            continue
        summary = run.summary._json_dict or {}
        params = summary.get("model/num_parameters") or run.config.get("model/num_parameters")
        params_m = None
        if params is not None:
            numeric = pd.to_numeric(params, errors="coerce")
            if np.isfinite(numeric):
                params_m = int(round(float(numeric) / 1e6))
        jk_records.append(
            {
                "benchmark": data_config.get("benchmark"),
                "subset": data_config.get("benchmark_subset_fraction"),
                "overtrain": trainer_config.get("overtrain_multiplier"),
                "params_M": params_m,
                "replicas": data_config.get("num_benchmark_replicas_per_epoch"),
                "has_loss": LOSS_COL in summary,
                "state": run.state,
            }
        )
    summarize("jkazdan/...-pt", jk_records)

    print("\n=== Overlap on (params_M, replicas) alone, ignoring subset/overtrain/benchmark")
    cache_pairs = {
        (r["params_M"], r["replicas"]) for r in cache_records if r["params_M"] is not None
    }
    jk_pairs = {
        (r["params_M"], r["replicas"]) for r in jk_records if r["params_M"] is not None
    }
    print(f"  cache pairs: {len(cache_pairs)}, jkazdan pairs: {len(jk_pairs)}")
    print(f"  intersection: {len(cache_pairs & jk_pairs)}")
    print(f"  jkazdan runs carrying {LOSS_COL}: {sum(1 for r in jk_records if r['has_loss'])}")

    only_cache = sorted(p for p in cache_pairs - jk_pairs if p[0] is not None)
    print(f"\n  (params_M, replicas) in cache but NOT in jkazdan: {len(only_cache)}")
    print(f"    {only_cache[:25]}")


if __name__ == "__main__":
    main()
