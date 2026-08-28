"""Check whether a W&B sweep ID resolves, and report its state and run count.

Usage: uv run python scripts/scratch/check_sweep_exists.py dj21lgk3 [more_ids...]
"""

import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import wandb

PROJECT = "memorization-scoring-vs-sampling-pt-v1-scale-ladder"

api = wandb.Api()
for sweep_id in sys.argv[1:]:
    try:
        sweep = api.sweep(f"{api.default_entity}/{PROJECT}/{sweep_id}")
        print(f"{sweep_id}: EXISTS state={sweep.state} runs={len(sweep.runs)}")
    except Exception as exc:
        print(f"{sweep_id}: NOT FOUND ({type(exc).__name__}: {exc})")
