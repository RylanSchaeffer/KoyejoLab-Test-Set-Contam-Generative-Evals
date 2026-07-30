"""Stage 2: score the locally cached raw responses independently.

Each run is scored in its OWN process (math_verify installs signal-based timeouts, which
only work in a process's main thread).  The parent enforces a wall-clock budget per child
and, if a child is killed, restarts it with the offending problem index blacklisted, so a
single pathological expression cannot stall the whole job.  A per-run exception cap is
also enforced inside the child.
"""
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, "/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization")

SP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SP, "raw")
OUT = os.path.join(SP, "scored")
PROG = os.path.join(SP, "progress")
os.makedirs(OUT, exist_ok=True)
os.makedirs(PROG, exist_ok=True)

CHILD_BUDGET_S = 900          # wall-clock per run
MAX_EXC_PER_RUN = 50          # abort a run if the scorer throws this often


def child(run_id, blacklist):
    from math_verify import parse
    from src.scoring import extract_boxed_answer, score_response

    prog_path = os.path.join(PROG, run_id)
    n = n_logged = n_strict = n_boxed = n_exc = 0
    bl = set(blacklist)
    with gzip.open(os.path.join(RAW, f"{run_id}.jsonl.gz"), "rt") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            resp, gold = rec.get("r"), rec.get("g")
            if resp is None or gold is None:
                continue
            n += 1
            if rec.get("s"):
                n_logged += 1
            if i in bl:
                continue
            with open(prog_path, "w") as pf:
                pf.write(str(i))
            if extract_boxed_answer(resp) is None:
                continue
            n_boxed += 1
            try:
                if score_response(parse(gold), resp):
                    n_strict += 1
            except Exception:
                n_exc += 1
                if n_exc >= MAX_EXC_PER_RUN:
                    break
    with open(os.path.join(OUT, f"{run_id}.json"), "w") as f:
        json.dump({"run_id": run_id, "n": n, "n_logged": n_logged, "n_strict": n_strict,
                   "n_boxed": n_boxed, "n_exc": n_exc, "n_skipped": len(bl),
                   "skipped_idx": sorted(bl)}, f)
    if os.path.exists(prog_path):
        os.remove(prog_path)


def run_one(job):
    """Run a single job to completion, blacklisting any index that hangs the child."""
    rid = job["run_id"]
    out_path = os.path.join(OUT, f"{rid}.json")
    if os.path.exists(out_path):
        return rid, "cached"
    blacklist = []
    for attempt in range(6):
        p = mp.Process(target=child, args=(rid, list(blacklist)))
        p.start()
        p.join(CHILD_BUDGET_S)
        if p.is_alive():
            stuck = None
            pp = os.path.join(PROG, rid)
            if os.path.exists(pp):
                stuck = int(open(pp).read().strip())
            p.terminate()
            p.join(30)
            if p.is_alive():
                p.kill()
                p.join()
            print(f"  !! {rid} hung at index {stuck} (attempt {attempt}); blacklisting",
                  flush=True)
            if stuck is None:
                return rid, "HUNG_UNKNOWN"
            blacklist.append(stuck)
            continue
        if os.path.exists(out_path):
            return rid, "ok" + (f" (skipped {len(blacklist)})" if blacklist else "")
        return rid, f"CRASH exit={p.exitcode}"
    return rid, "GAVE_UP"


def main():
    jobs = [j for j in json.load(open(os.path.join(SP, "jobs.json")))
            if j["state"] == "finished" and j["Parameters"] and j["R"] is not None]
    jobs = [j for j in jobs if os.path.exists(os.path.join(RAW, f"{j['run_id']}.jsonl.gz"))]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f"scoring {len(jobs)} runs with {workers} concurrent children", flush=True)

    # Bounded concurrency over run_one, each of which forks its own scoring child.
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rid, status in ex.map(run_one, jobs):
            done += 1
            if status != "cached":
                print(f"[{done}/{len(jobs)}] {rid} {status} ({time.time()-t0:.0f}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
