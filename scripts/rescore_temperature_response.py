"""Recompute the temperature-response analysis under boxed-required scoring.

WHY
---
`scripts/analyze_temperature_response.py` answers 8RFz's W2/Q2 with the contamination advantage
at matched temperature, `score(R) - score(R=0)`, on the theory that any uniform degradation
cancels in the difference. The design is right, but it reads the scores those 0-shot runs
*logged*, which used the lenient scorer (~1.4% false positives).

Those false positives do NOT cancel, because the two arms are not equally affected. Measured on
62M at tau=1.0:

    R=0    logged 0.0124   strict 0.0000
    R=316  logged 0.0190   strict 0.0100
    advantage: lenient 0.0066  vs  strict 0.0100

The uncontaminated arm is almost entirely false positives while the contaminated arm is mostly
real, so subtracting a lenient R=0 over-subtracts and *understates* the advantage. The effect is
largest exactly where the reported numbers are smallest -- the high-temperature tail.

This rescores every temperature run from its raw responses and recomputes the table.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      ./mem_scoring_vs_sampling_env/bin/python scripts/rescore_temperature_response.py
"""

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import wandb
from math_verify import parse

sys.path.insert(0, os.getcwd())
from src.scoring import extract_boxed_answer, score_response  # noqa: E402

ENTITY, PROJECT = "rylan", "memorization-scoring-vs-sampling-eval"
# The 0-shot sweeps, i.e. the protocol behind the manuscript's temperature figure.
ZERO_SHOT_SWEEPS = [
    "6y9dy2ow", "lnrpy3ed", "5oo55o9s", "10q465ij",
    "q5uoy1eu", "f5djvfth", "vnz1h147", "xkzfmbhk", "39rugx2e",
]
OUT_DIR = "notebooks/11_math_qwen3_pt_math_verify/results"
OUT_CSV = os.path.join(OUT_DIR, "temperature_response_rescored.csv")
OUT_MD = os.path.join(OUT_DIR, "TEMPERATURE_RESPONSE_RESCORED.md")


def rescore(job: dict) -> dict:
    """Rescore one run. MUST run in a process, not a thread (math_verify uses signal timeouts)."""
    api = wandb.Api(timeout=120)
    run = api.run(f"{ENTITY}/{PROJECT}/{job['run_id']}")
    n = lg = st = exc = 0
    for h in run.scan_history(keys=["math_verify_score", "response", "solution"]):
        resp, sol = h.get("response"), h.get("solution")
        if resp is None or sol is None:
            continue
        n += 1
        lg += int(bool(h.get("math_verify_score")))
        if extract_boxed_answer(resp) is not None:
            try:
                st += int(bool(score_response(parse(sol), resp)))
            except Exception:
                exc += 1
    out = dict(job, n=n, n_exc=exc,
               logged=(lg / n if n else float("nan")),
               strict=(st / n if n else float("nan")))
    print(f"  {job['Parameters']:>5} R={job['R']:<5d} T={job['T']:<8} "
          f"logged={out['logged']:.4f} strict={out['strict']:.4f}"
          + (f"  !! {exc} exceptions" if exc else ""), flush=True)
    return out


def main() -> None:
    api = wandb.Api(timeout=90)
    jobs = []
    for sid in ZERO_SHOT_SWEEPS:
        try:
            sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        except Exception as e:
            print(f"  sweep {sid}: {e}")
            continue
        for run in sweep.runs:
            if run.state != "finished":
                continue
            model = run.config.get("model_config", {}).get("model", "")
            params = re.search(r"Qwen3-([\d.]+[MB])", model)
            reps = re.search(r"rep_(\d+)_sbst", model)
            if not params or not reps:
                continue
            jobs.append({"run_id": run.id, "Parameters": params.group(1),
                         "R": int(reps.group(1)), "T": run.config.get("temperature")})
    # Deduplicate (a config can appear in more than one sweep); keep the first.
    seen, uniq = set(), []
    for j in jobs:
        k = (j["Parameters"], j["R"], j["T"])
        if k not in seen:
            seen.add(k)
            uniq.append(j)
    print(f"Rescoring {len(uniq)} temperature runs...\n", flush=True)

    with ProcessPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(rescore, uniq))
    df = pd.DataFrame(recs)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    if int(df["n_exc"].sum()):
        print(f"\n!! {int(df['n_exc'].sum())} scoring exceptions -- results not trustworthy")

    # Contamination advantage at matched temperature, strict scoring.
    base = df[df.R == 0].set_index(["Parameters", "T"])["strict"]
    df["baseline"] = [base.get((p, t), float("nan")) for p, t in zip(df.Parameters, df["T"])]
    df["advantage"] = df["strict"] - df["baseline"]

    greedy = df[(df["T"] == 0.0)].set_index(["Parameters", "R"])["strict"]
    df["greedy_strict"] = [greedy.get((p, r), float("nan")) for p, r in zip(df.Parameters, df.R)]
    real = df[(df.R > 0) & (df.greedy_strict >= 0.05)]

    greedy_adv = real[real["T"] == 0.0].set_index(["Parameters", "R"])["advantage"]
    real = real.copy()
    real["greedy_advantage"] = [
        greedy_adv.get((p, r), float("nan")) for p, r in zip(real.Parameters, real.R)
    ]
    real["fraction_of_greedy_advantage"] = real["advantage"] / real["greedy_advantage"]
    table = (real.groupby("T")[["advantage", "fraction_of_greedy_advantage"]]
             .mean().reset_index().sort_values("T"))

    with open(OUT_MD, "w") as f:
        f.write(
            "# Temperature response, rescored with boxed-required scoring\n\n"
            "Supersedes the `fraction_of_greedy_advantage` column of `TEMPERATURE_RESPONSE.md`, "
            "which was computed from leniently scored runs.\n\n"
            "The matched-temperature *difference* was supposed to cancel scoring artifacts, but "
            "it does not: the uncontaminated arm's lenient score is almost entirely false "
            "positives while the contaminated arm's is mostly real, so subtracting a lenient "
            "R=0 over-subtracts. That **understates** the advantage, most severely in the "
            "high-temperature tail where the true values are smallest.\n\n"
            "Averaged over conditions with greedy (strict) score >= 5%.\n\n"
        )
        f.write(table.round(4).to_markdown(index=False))
        f.write("\n\nPer-run detail: `temperature_response_rescored.csv`.\n")
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
