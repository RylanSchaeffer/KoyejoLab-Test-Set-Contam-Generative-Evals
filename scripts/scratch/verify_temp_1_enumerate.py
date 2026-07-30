"""Independently enumerate the 0-shot temperature runs. Writes a job manifest."""
import json
import os
import re
import sys

import wandb

ENTITY, PROJECT = "rylan", "memorization-scoring-vs-sampling-eval"
SWEEPS = [
    "6y9dy2ow", "lnrpy3ed", "5oo55o9s", "10q465ij",
    "q5uoy1eu", "f5djvfth", "vnz1h147", "xkzfmbhk", "39rugx2e",
]
OUT = sys.argv[1]

api = wandb.Api(timeout=120)
rows = []
for sid in SWEEPS:
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
    for run in sweep.runs:
        model = run.config.get("model_config", {}).get("model", "")
        m_p = re.search(r"Qwen3-([\d.]+[MB])", model)
        m_r = re.search(r"rep_(\d+)_sbst", model)
        rows.append({
            "sweep": sid,
            "run_id": run.id,
            "state": run.state,
            "model": model,
            "Parameters": m_p.group(1) if m_p else None,
            "R": int(m_r.group(1)) if m_r else None,
            "T": run.config.get("temperature"),
            "num_fewshot": run.config.get("num_fewshot",
                                          run.config.get("evaluation_config", {}).get("num_fewshot")),
        })
    print(f"{sid}: cumulative {len(rows)}", flush=True)

with open(OUT, "w") as f:
    json.dump(rows, f, indent=1)
print(f"wrote {len(rows)} rows to {OUT}")
