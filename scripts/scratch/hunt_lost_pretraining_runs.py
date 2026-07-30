"""Exhaustively hunt the lost pretraining runs by exact W&B run ID.

The earlier investigation matched by *configuration*, which cannot distinguish "these runs are
gone" from "my matcher is wrong". The notebook-11 cache records the actual `run_id` and `Sweep`
for every row, so this searches by exact identifier instead.

If project `memorization-scoring-vs-sampling-pt` was RENAMED rather than deleted, the run IDs
survive under the new name and this will find them. If it was deleted, nothing will match and
that is then a real answer rather than an artifact of a bad matcher.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      ./mem_scoring_vs_sampling_env/bin/python scripts/scratch/hunt_lost_pretraining_runs.py
"""

import json
import sys

import pandas as pd
import wandb

CACHE = "notebooks/11_math_qwen3_pt_math_verify/data/c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv"
OUT = "reviews/2026_neurips/data/lost_run_hunt.json"


def main() -> None:
    api = wandb.Api(timeout=90)

    df = pd.read_csv(CACHE)
    run_ids = sorted(set(df["run_id"].dropna().astype(str)))
    sweeps = sorted(set(df["Sweep"].dropna().astype(str)))
    print(f"Hunting {len(run_ids)} run ids from {len(sweeps)} sweeps.\n", flush=True)

    viewer = api.viewer
    entities = sorted({viewer.entity, *(viewer.teams or [])})
    print(f"Entities visible to this key: {entities}\n", flush=True)

    # Enumerate every project of every entity.
    projects = []
    for ent in entities:
        try:
            ps = list(api.projects(ent))
            print(f"  {ent:24s} {len(ps)} projects", flush=True)
            projects += [(ent, p.name) for p in ps]
        except Exception as exc:  # entity may be inaccessible
            print(f"  {ent:24s} ERROR {type(exc).__name__}: {str(exc)[:80]}", flush=True)
    print(f"\nTotal projects to scan: {len(projects)}\n", flush=True)

    wanted = set(run_ids)
    found: dict[str, list[str]] = {}
    errors = 0

    for i, (ent, proj) in enumerate(projects, 1):
        path = f"{ent}/{proj}"
        try:
            # Server-side filter on run id: one query per project, not one per run.
            rs = api.runs(path, filters={"name": {"$in": sorted(wanted)}}, per_page=200)
            hits = [r.id for r in rs]
        except Exception:
            errors += 1
            continue
        if hits:
            found[path] = hits
            print(f"  *** {len(hits):4d} MATCHES in {path}", flush=True)
        if i % 25 == 0:
            print(f"  ...scanned {i}/{len(projects)}", flush=True)

    total = sum(len(v) for v in found.values())
    print(f"\n=== RESULT ===")
    print(f"projects scanned : {len(projects)} ({errors} unreadable)")
    print(f"run ids sought   : {len(run_ids)}")
    print(f"run ids found    : {total}")
    for path, hits in sorted(found.items(), key=lambda kv: -len(kv[1])):
        print(f"  {path}: {len(hits)}")

    missing = sorted(wanted - {h for v in found.values() for h in v})
    print(f"still missing    : {len(missing)}")

    with open(OUT, "w") as f:
        json.dump(
            {
                "entities": entities,
                "projects_scanned": len(projects),
                "projects_unreadable": errors,
                "run_ids_sought": run_ids,
                "found_by_project": found,
                "still_missing": missing,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
