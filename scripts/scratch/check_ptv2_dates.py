"""When were the -pt-v2 runs created, relative to the cache file?

If -pt-v2 postdates the cache, the likely history is that the original project was deleted and
a partial re-run started, rather than the project being renamed (a rename would have carried all
177 configurations across, and only 22 are there).
"""
import os, datetime
import wandb

CACHE = ("notebooks/11_math_qwen3_pt_math_verify/data/"
         "c39ba9b590fe96b52183328d3d4c7323_runs_configs.csv")
print(f"cache mtime: {datetime.datetime.fromtimestamp(os.path.getmtime(CACHE))}")

api = wandb.Api(timeout=600)
for path in ["rylan/memorization-scoring-vs-sampling-pt-v2",
             "jkazdan/memorization-scoring-vs-sampling-pt"]:
    dates, sweeps = [], set()
    for run in api.runs(path, per_page=200):
        dates.append(str(run.created_at))
        if run.sweep is not None:
            sweeps.add(run.sweep.id)
    if dates:
        print(f"\n{path}")
        print(f"  n={len(dates)} earliest={min(dates)} latest={max(dates)}")
        print(f"  sweeps={sorted(sweeps)}")
