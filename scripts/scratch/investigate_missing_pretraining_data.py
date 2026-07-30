"""Establish exactly what pretraining data is missing, and what still exists where.

Notebook 11 and notebook 20 both merge pretraining cross-entropy against Math Verify. They
name 15 sweep IDs in a project (`memorization-scoring-vs-sampling-pt`) that does not exist under
`rylan`, and none of those 15 sweeps was found in any of 325 projects across every visible
entity. Their data survives only in a local cache file.

This produces the full inventory: which configurations the cache holds, which of those exist
anywhere in W&B today (under any project, matched by *configuration* rather than by sweep ID),
and which are therefore cache-only. Matching by configuration is the point — sweep IDs are gone,
but a run for the same (model, replicas, overtrain) is just as usable.
"""

import ast
import json
import os

import numpy as np
import pandas as pd
import wandb

CACHE = (
    "notebooks/11_math_qwen3_pt_math_verify/data/"
    "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv"
)
LOSS_COL = "eval_after/eval_benchmark_loss"
OUT_DIR = "reviews/2026_neurips/data"

EXTRA_ENTITIES = ["jkazdan", "stellaathena"]
# Any project that could plausibly hold contamination pretraining runs.
PROJECT_HINTS = ("mem", "contam", "scaling", "pretrain", "pt")


def config_key(data_config, trainer_config, num_parameters):
    try:
        data_config = ast.literal_eval(data_config) if isinstance(data_config, str) else data_config
        trainer_config = (
            ast.literal_eval(trainer_config) if isinstance(trainer_config, str) else trainer_config
        )
    except (ValueError, SyntaxError):
        return None
    if not isinstance(data_config, dict) or not isinstance(trainer_config, dict):
        return None
    replicas = data_config.get("num_benchmark_replicas_per_epoch")
    subset = data_config.get("benchmark_subset_fraction")
    benchmark = data_config.get("benchmark")
    overtrain = trainer_config.get("overtrain_multiplier")
    if replicas is None or num_parameters is None:
        return None
    # Crashed runs can log a NaN parameter count; they carry no usable loss either.
    try:
        num_parameters = float(num_parameters)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num_parameters):
        return None
    return (
        int(round(float(num_parameters) / 1e6)),
        int(replicas),
        float(subset) if subset is not None else None,
        float(overtrain) if overtrain is not None else None,
        str(benchmark),
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = pd.read_csv(CACHE, low_memory=False)
    print(f"Cache: {len(cache)} rows, {LOSS_COL} present: {LOSS_COL in cache.columns}")
    cache_keys = {}
    for _, row in cache.iterrows():
        key = config_key(
            row.get("data_config"), row.get("trainer_config"), row.get("model/num_parameters")
        )
        if key is None:
            continue
        has_loss = pd.notna(row.get(LOSS_COL))
        # Keep the row with a loss if any duplicate has one.
        cache_keys[key] = cache_keys.get(key, False) or bool(has_loss)
    print(f"Cache holds {len(cache_keys)} distinct configurations, "
          f"{sum(cache_keys.values())} with a benchmark loss")

    api = wandb.Api(timeout=600)
    entities = [api.default_entity]
    for team in getattr(api.viewer, "teams", []) or []:
        if team not in entities:
            entities.append(team)
    for entity in EXTRA_ENTITIES:
        if entity not in entities:
            entities.append(entity)

    live_keys = {}
    scanned_projects = []
    for entity in entities:
        try:
            projects = list(api.projects(entity))
        except Exception as e:
            print(f"[{entity}] cannot list: {type(e).__name__}")
            continue
        for project in projects:
            if not any(h in project.name.lower() for h in PROJECT_HINTS):
                continue
            path = f"{entity}/{project.name}"
            try:
                n = 0
                hits = 0
                for run in api.runs(path, per_page=200):
                    n += 1
                    key = config_key(
                        run.config.get("data_config"),
                        run.config.get("trainer_config"),
                        (run.summary._json_dict or {}).get("model/num_parameters")
                        or run.config.get("model/num_parameters"),
                    )
                    if key is None:
                        continue
                    loss = (run.summary._json_dict or {}).get(LOSS_COL)
                    if key not in live_keys or (loss is not None and not live_keys[key][1]):
                        live_keys[key] = (path, loss is not None)
                    hits += 1
                scanned_projects.append((path, n, hits))
                if hits:
                    print(f"  {path}: {n} runs, {hits} parse as pretraining configs")
            except Exception as e:
                print(f"  {path}: error {type(e).__name__}")

    rows = []
    for key, cache_has_loss in sorted(cache_keys.items()):
        params_m, replicas, subset, overtrain, benchmark = key
        live = live_keys.get(key)
        rows.append(
            {
                "params_M": params_m,
                "replicas": replicas,
                "subset_fraction": subset,
                "overtrain": overtrain,
                "benchmark": benchmark,
                "in_cache_with_loss": cache_has_loss,
                "found_in_wandb": live[0] if live else None,
                "wandb_has_loss": live[1] if live else False,
            }
        )
    inventory = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "missing_pretraining_data_inventory.csv")
    inventory.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    cache_only = inventory[inventory["found_in_wandb"].isna()]
    recoverable = inventory[inventory["found_in_wandb"].notna()]
    print(f"\nConfigurations in cache: {len(inventory)}")
    print(f"  also live in W&B somewhere: {len(recoverable)}")
    print(f"  CACHE-ONLY (unrecoverable if the file is lost): {len(cache_only)}")
    if not cache_only.empty:
        print("\nCache-only configurations by model size:")
        print(
            cache_only.groupby("params_M")["replicas"]
            .apply(lambda s: sorted(s))
            .to_string()
        )
    if not recoverable.empty:
        print("\nWhere the recoverable ones live:")
        print(recoverable["found_in_wandb"].value_counts().to_string())

    with open(os.path.join(OUT_DIR, "missing_pretraining_scan_projects.json"), "w") as f:
        json.dump(
            [{"project": p, "n_runs": n, "n_pretraining": h} for p, n, h in scanned_projects],
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
