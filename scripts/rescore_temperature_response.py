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

CORRECTED 2026-07-30 after adversarial verification
---------------------------------------------------
The first version of this script had two silent regressions relative to
`scripts/analyze_temperature_response.py`, which it was meant only to rescore:

1. **344M vanished.** 344M has no finished 0-shot R=0 run (all ten failed), so
   `base.get(("344M", t))` returned NaN, every 344M advantage was NaN, and
   `groupby(...).mean()` silently skipped them. The table collapsed from 13 contributing
   conditions to 9, dropping the largest and most contaminated model. The older script
   handles this explicitly with a documented fallback to the size's lowest available replica
   level (344M R=1, whose greedy strict score is 0.0004, i.e. at the uncontaminated floor).
2. **The estimator changed.** The older script reports a ratio of means,
   `mean_T(advantage) / mean_0(advantage)`. This one reported a mean of per-condition ratios.

Together those two changes, not the scoring rule, produced the reported drop from 25% to 9.6%
retention at tau = 1.0. Matched on coverage and estimator, boxed-required scoring moves the
tau = 1.0 figure from 0.2495 to 0.2528 -- in the predicted direction, but by 0.3 pp.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      ./mem_scoring_vs_sampling_env/bin/python scripts/rescore_temperature_response.py
    # rebuild only the table from the already-rescored per-run CSV:
    ... scripts/rescore_temperature_response.py --from-csv
"""

import argparse
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


def build_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Contamination advantage at matched temperature, boxed-required scoring.

    Baseline is R=0 where it exists; where it does not (344M), fall back to that size's
    lowest available replica level and record it, rather than letting the size disappear
    into a NaN. Reports both estimators so the choice is visible.
    """
    reference = {}
    for size, group in df.groupby("Parameters"):
        available = sorted(group["R"].unique())
        reference[size] = 0 if 0 in available else available[0]

    strict_by_key = df.set_index(["Parameters", "R", "T"])["strict"]
    df = df.copy()
    df["baseline"] = [
        strict_by_key.get((p, reference[p], t), float("nan"))
        for p, t in zip(df.Parameters, df["T"])
    ]
    df["advantage"] = df["strict"] - df["baseline"]

    greedy = df[df["T"] == 0.0].set_index(["Parameters", "R"])["strict"]
    df["greedy_strict"] = [greedy.get((p, r), float("nan")) for p, r in zip(df.Parameters, df.R)]

    real = df[(df.R > 0) & (df.greedy_strict >= 0.05)].copy()
    # A fallback reference cannot be its own contrast.
    real = real[[reference[p] != r for p, r in zip(real.Parameters, real.R)]]
    missing = real[real["advantage"].isna()]
    if len(missing):
        raise SystemExit(
            "Conditions with greedy >= 5% but no baseline -- they would vanish from the "
            f"mean silently:\n{missing[['Parameters', 'R', 'T']].to_string(index=False)}"
        )

    greedy_adv = real[real["T"] == 0.0].set_index(["Parameters", "R"])["advantage"]
    real["greedy_advantage"] = [
        greedy_adv.get((p, r), float("nan")) for p, r in zip(real.Parameters, real.R)
    ]
    real["ratio"] = real["advantage"] / real["greedy_advantage"]

    table = real.groupby("T").agg(
        advantage=("advantage", "mean"),
        mean_of_ratios=("ratio", "mean"),
        n_conditions=("ratio", "count"),
    ).reset_index().sort_values("T")
    greedy_mean = float(table.loc[table["T"] == 0.0, "advantage"].iloc[0])
    table["fraction_of_greedy_advantage"] = table["advantage"] / greedy_mean
    table = table[["T", "advantage", "fraction_of_greedy_advantage",
                   "mean_of_ratios", "n_conditions"]]
    conditions = (
        real[real["T"] == 0.0].groupby("Parameters")["R"].apply(lambda s: sorted(set(s)))
    )
    return table, {"reference": reference, "conditions": conditions}


def write_report(df: pd.DataFrame) -> None:
    table, meta = build_table(df)
    reference = meta["reference"]
    fallbacks = ", ".join(f"{k} -> R={v}" for k, v in sorted(reference.items()) if v != 0)
    with open(OUT_MD, "w") as f:
        f.write(
            "# Temperature response, rescored with boxed-required scoring\n\n"
            "Rescores every 0-shot temperature run from its raw responses under the "
            "boxed-required scorer, then recomputes the contamination advantage at matched "
            "temperature, `score(R) - score(R=0)`, which is the control for reviewer 8RFz's "
            "W2/Q2.\n\n"
            "## Headline: rescoring does not change the answer\n\n"
            "The concern that motivated this rescoring was real but small. The lenient scorer "
            "does inflate the uncontaminated arm more than the contaminated arm, so subtracting "
            "a lenient `R=0` over-subtracts and understates the advantage -- measured on 62M at "
            "tau=1.0, the advantage is 0.0066 lenient against 0.0100 strict. But that condition "
            "is one of the smallest contributors. Averaged over the contributing conditions the "
            "effect is **+0.3 pp**: retention at tau=1.0 goes from 0.2495 (lenient) to 0.2528 "
            "(strict). The table below therefore agrees with `TEMPERATURE_RESPONSE.md` to two "
            "significant figures at every temperature.\n\n"
            "> An earlier version of this file reported 0.0961 at tau=1.0 and attributed the\n"
            "> change to the scoring rule. That was wrong. The script had silently dropped 344M\n"
            "> (which has no finished 0-shot R=0 run, so its advantage was NaN and\n"
            "> `groupby().mean()` skipped it) and had also switched the estimator from a ratio\n"
            "> of means to a mean of ratios. Coverage accounted for -12.8 pp and the estimator\n"
            "> for -2.9 pp; the scoring rule accounted for +0.3 pp. See\n"
            "> `reviews/2026_neurips/verification/TEMPERATURE_VERIFICATION.md`.\n\n"
            "## Contamination advantage at matched temperature (strict scoring)\n\n"
            "Averaged over conditions with greedy (strict) score >= 5%. "
            "`fraction_of_greedy_advantage` is the ratio of means, matching "
            "`analyze_temperature_response.py`; `mean_of_ratios` is the per-condition average, "
            "shown so the estimator choice is visible rather than load-bearing and invisible.\n\n"
        )
        f.write(table.round(4).to_markdown(index=False))
        f.write("\n\nConditions contributing to each mean (model size -> replica levels):\n\n"
                "```\n" + meta["conditions"].to_string() + "\n```\n\n")
        if fallbacks:
            f.write(
                f"Clean reference: R=0 where it exists; {fallbacks} because all ten 0-shot "
                "344M R=0 runs failed. 344M R=1 scores 0.0004 strict at greedy, i.e. it sits on "
                "the uncontaminated floor, so it is a sound stand-in. Without the fallback 344M "
                "drops out of every mean silently and tau=1.0 reads 0.1251 (ratio of means) or "
                "0.0961 (mean of ratios) instead of 0.2528.\n\n"
            )
        f.write(
            "One run, 344M R=100 at tau=1.0, was absent from the first version of the per-run "
            "CSV (a worker stalled and the table was regex-parsed out of the log). It is present "
            "now and scores 0.1758 strict.\n\n"
            "Per-run detail: `temperature_response_rescored.csv`.\n"
        )
    print("\n" + table.round(4).to_string(index=False))
    print(f"\nWrote {OUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true",
                        help="Rebuild the report from the existing per-run CSV, no W&B calls.")
    args = parser.parse_args()
    if args.from_csv:
        write_report(pd.read_csv(OUT_CSV))
        return

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

    write_report(df)


if __name__ == "__main__":
    main()
