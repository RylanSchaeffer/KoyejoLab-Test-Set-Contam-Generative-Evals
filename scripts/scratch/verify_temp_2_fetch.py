"""Stage 1: download raw per-problem (response, solution, logged score) for every
finished 0-shot temperature run. Pure network work -> threads are fine, no math_verify here.
Each run is written to its own gzipped jsonl so scoring can be redone offline."""
import gzip
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import wandb

ENTITY, PROJECT = "rylan", "memorization-scoring-vs-sampling-eval"
SP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SP, "raw")
os.makedirs(RAW, exist_ok=True)

jobs = [j for j in json.load(open(os.path.join(SP, "jobs.json")))
        if j["state"] == "finished" and j["Parameters"] and j["R"] is not None]
lock = threading.Lock()
done = [0]


def fetch(job):
    path = os.path.join(RAW, f"{job['run_id']}.jsonl.gz")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return job["run_id"], "cached"
    api = wandb.Api(timeout=180)
    run = api.run(f"{ENTITY}/{PROJECT}/{job['run_id']}")
    tmp = path + ".part"
    n = 0
    with gzip.open(tmp, "wt") as f:
        for h in run.scan_history(keys=["math_verify_score", "response", "solution"]):
            f.write(json.dumps({"s": h.get("math_verify_score"),
                                "r": h.get("response"),
                                "g": h.get("solution")}) + "\n")
            n += 1
    os.replace(tmp, path)
    with lock:
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] {job['run_id']} {job['Parameters']} R={job['R']} "
              f"T={job['T']} rows={n}", flush=True)
    return job["run_id"], n


with ThreadPoolExecutor(max_workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
    res = list(ex.map(fetch, jobs))
print("DONE", len(res))
