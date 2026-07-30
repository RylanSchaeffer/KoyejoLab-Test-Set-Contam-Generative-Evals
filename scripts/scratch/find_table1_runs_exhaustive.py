"""Exhaustively hunt for the runs behind Table 1's 34M and 93M rephrased/perturbed columns.

An earlier pass scanned only two `rylan/*` projects and matched four exact dataset names,
and found nothing at 34M or 93M. That is weak evidence: collaborators log to their own
entities (`src/globals.py` still defaults to a `jkazdan/...` checkpoint), project names may
differ, dataset strings may be spelled differently, and runs may not be in a `finished`
state.

This widens the search on every one of those axes:
  * every project of every entity the API key can see (not a hardcoded list);
  * substring matching on "rephras"/"perturb" anywhere in the config, not an exact name set;
  * every run state, not just finished;
  * reports model sizes present, so 34M/93M coverage is answered directly.

It also greps the local filesystem for cached run tables that might hold the numbers even if
the W&B runs were deleted.
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd
import wandb

NEEDLES = ("rephras", "perturb")
EXTRA_ENTITIES = ["rylan", "jkazdan", "stellaathena"]


def config_mentions_modified_dataset(config: dict) -> str | None:
    """Return the matching dataset-ish string if any config value mentions a modified set."""
    try:
        blob = json.dumps(config, default=str).lower()
    except (TypeError, ValueError):
        blob = str(config).lower()
    for needle in NEEDLES:
        if needle in blob:
            return needle
    return None


def extract_dataset(config: dict) -> str:
    data_config = config.get("data_config")
    if isinstance(data_config, dict):
        return str(data_config.get("dataset"))
    return str(data_config)


def main() -> None:
    api = wandb.Api(timeout=600)

    entities = []
    try:
        entities.append(api.default_entity)
    except Exception:
        pass
    for entity in EXTRA_ENTITIES:
        if entity not in entities:
            entities.append(entity)

    records = []
    for entity in entities:
        try:
            projects = list(api.projects(entity))
        except Exception as e:
            print(f"[entity {entity}] cannot list projects: {e}")
            continue
        print(f"[entity {entity}] {len(projects)} projects")

        for project in projects:
            try:
                runs = api.runs(f"{entity}/{project.name}", per_page=200)
                n_runs = 0
                n_hits = 0
                for run in runs:
                    n_runs += 1
                    needle = config_mentions_modified_dataset(run.config)
                    if needle is None:
                        continue
                    n_hits += 1
                    model_config = run.config.get("model_config")
                    model_name = (
                        model_config.get("model", "")
                        if isinstance(model_config, dict)
                        else str(model_config)
                    )
                    parameters = re.search(r"Qwen3-([\d.]+[MB])", str(model_name))
                    replicas = re.search(r"rep_(\d+)_sbst", str(model_name))
                    records.append(
                        {
                            "entity": entity,
                            "project": project.name,
                            "sweep": run.sweep.id if run.sweep is not None else None,
                            "run_id": run.id,
                            "state": run.state,
                            "created": str(run.created_at),
                            "dataset": extract_dataset(run.config),
                            "Parameters": parameters.group(1) if parameters else None,
                            "Num. Replicas": int(replicas.group(1)) if replicas else None,
                            "temperature": run.config.get("temperature"),
                            "model": model_name,
                        }
                    )
                if n_runs:
                    print(f"    {project.name}: {n_runs} runs, {n_hits} mention a modified set")
            except Exception as e:
                print(f"    {project.name}: error {e}")

    if records:
        df = pd.DataFrame(records)
        out_path = "table1_provenance_exhaustive.csv"
        df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(df)} runs)\n")
        print("=== Coverage by dataset x model size x state ===")
        print(
            df.groupby(["dataset", "Parameters", "state"], dropna=False)
            .size()
            .reset_index(name="n_runs")
            .to_markdown(index=False)
        )
        print("\n=== Model sizes seen anywhere ===")
        print(sorted({str(p) for p in df["Parameters"]}))
        small = df[df["Parameters"].isin(["34M", "93M", "62M", "63M", "48M"])]
        print(f"\n=== Runs at 34M/48M/62M/63M/93M: {len(small)} ===")
        if not small.empty:
            print(small.to_string(index=False))
    else:
        print("\nNo W&B runs mention a rephrased/perturbed dataset anywhere.")

    # Local caches may retain numbers whose runs were deleted from W&B.
    print("\n=== Local cached run tables mentioning rephrase/perturb ===")
    hits = defaultdict(list)
    for root, _dirs, files in os.walk("notebooks"):
        for name in files:
            if not name.endswith((".csv", ".feather", ".parquet")):
                continue
            path = os.path.join(root, name)
            if not name.endswith(".csv"):
                hits["non-csv cache (not scanned)"].append(path)
                continue
            try:
                with open(path, "r", errors="ignore") as f:
                    head = f.read(2_000_000).lower()
            except OSError:
                continue
            if any(needle in head for needle in NEEDLES):
                sizes = sorted(set(re.findall(r"Qwen3-([\d.]+[MB])", head, re.I)))
                hits["mentions modified dataset"].append(f"{path}  sizes={sizes}")
    for key, paths in hits.items():
        print(f"\n  {key}: {len(paths)}")
        for path in paths[:20]:
            print(f"    {path}")


if __name__ == "__main__":
    main()
