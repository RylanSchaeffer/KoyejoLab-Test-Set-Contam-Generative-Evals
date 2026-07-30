"""Rescore the 0-shot evaluation runs with boxed-required scoring, to de-confound the
0-shot-vs-4-shot comparison.

THE PROBLEM
-----------
`scripts/compare_zeroshot_vs_fewshot_protocol.py` reads the `math_verify_score` that each run
*logged at the time*. But the 0-shot and 4-shot sweeps were run on opposite sides of commit
`db75c5f` (2026-03-29), which changed TWO things at once:

  1. 0-shot -> 4-shot prompting, and
  2. lenient `math_verify.parse()` -> boxed-required `src.scoring.score_response`.

The lenient scorer extracts bare numbers from free text at priority 300 and was measured to have
a ~1.4% false-positive rate on garbage output. So the reported "0-shot vs 4-shot" contrast
confounds the prompt format with the scoring rule, and the uncontaminated R=0 baseline reading
0.0038-0.0126 at 0-shot but exactly 0.0000 at 4-shot is exactly what the scoring change alone
would produce. That sub-claim cannot be used to argue anything about prompt format.

THE FIX
-------
The raw `response` text is in W&B run history, so the 0-shot runs can be rescored with the strict
scorer with no GPU. That makes both columns apples-to-apples and isolates the prompt format.

This does NOT threaten the headline: 344M R=3162 scores 1.0000 at 0-shot, and a 1.4%
false-positive rate cannot manufacture that. Inspected 0-shot generations are verbatim copies of
the gold solution including a well-formed `\boxed{}`, so they pass strict scoring too. The
purpose here is to establish which parts of the contrast survive, and specifically what the
uncontaminated baseline really does.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      ./mem_scoring_vs_sampling_env/bin/python scripts/rescore_zeroshot_with_boxed_required.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import wandb
from math_verify import parse

sys.path.insert(0, os.getcwd())
from src.scoring import extract_boxed_answer, score_response  # noqa: E402

GRID = "notebooks/11_math_qwen3_pt_math_verify/results/protocol_sensitivity.csv"
PROJECT = "rylan/memorization-scoring-vs-sampling-eval"
OUT_CSV = "notebooks/11_math_qwen3_pt_math_verify/results/protocol_sensitivity_rescored.csv"
OUT_MD = "notebooks/11_math_qwen3_pt_math_verify/results/PROTOCOL_SENSITIVITY_RESCORED.md"


def rescore_run(row: pd.Series) -> dict:
    """Rescore one run.

    MUST run in a separate PROCESS, not a thread: `math_verify.verify()` installs a
    signal-based timeout, and `signal.signal()` raises outside the main thread. Running this
    under a ThreadPoolExecutor makes every call raise and -- if the exception is swallowed --
    silently reports 0.0000 for every run. That happened on the first attempt. Exceptions are
    therefore counted and surfaced below rather than passed over.
    """
    api = wandb.Api(timeout=120)
    run = api.run(f"{PROJECT}/{row['run_id']}")
    n = n_logged = n_strict = n_boxed = n_exc = 0
    first_exc = ""
    for h in run.scan_history(keys=["math_verify_score", "response", "solution"]):
        resp, sol = h.get("response"), h.get("solution")
        if resp is None or sol is None:
            continue
        n += 1
        n_logged += int(bool(h.get("math_verify_score")))
        if extract_boxed_answer(resp) is not None:
            n_boxed += 1
            try:
                n_strict += int(bool(score_response(parse(sol), resp)))
            except Exception as exc:
                n_exc += 1
                first_exc = first_exc or f"{type(exc).__name__}: {exc}"
    out = dict(row)
    out.update(
        n_rescored=n,
        n_score_exceptions=n_exc,
        logged_score=(n_logged / n if n else float("nan")),
        strict_score=(n_strict / n if n else float("nan")),
        boxed_rate=(n_boxed / n if n else float("nan")),
    )
    flag = f"  !! {n_exc} scoring exceptions ({first_exc})" if n_exc else ""
    print(
        f"  {row['protocol']:7s} {str(row['Parameters']):6s} R={int(row['Num. Replicas']):<5d}"
        f" logged={out['logged_score']:.4f} strict={out['strict_score']:.4f}"
        f" boxed={out['boxed_rate']:.4f}{flag}",
        flush=True,
    )
    return out


def main() -> None:
    grid = pd.read_csv(GRID)
    grid = grid[grid["Temp."] == 0.0]
    print(f"Rescoring {len(grid)} runs ({(grid.protocol=='0-shot').sum()} 0-shot, "
          f"{(grid.protocol=='4-shot').sum()} 4-shot)\n", flush=True)

    with ProcessPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(rescore_run, [r for _, r in grid.iterrows()]))

    df = pd.DataFrame(recs)
    df.to_csv(OUT_CSV, index=False)

    # Guard against the silent-failure mode that produced all-zeros on the first attempt.
    tot_exc = int(df["n_score_exceptions"].sum())
    if tot_exc:
        print(f"\n!! {tot_exc} scoring exceptions across the grid -- results are NOT trustworthy")
    if (df["strict_score"] == 0).all():
        print("!! every strict score is exactly 0.0000 -- almost certainly a bug, not a result")
    # 0-shot high-contamination runs regurgitate verbatim, so strict must track logged there.
    chk = df[(df.protocol == "0-shot") & (df.logged_score > 0.5)]
    if len(chk):
        worst = (chk["logged_score"] - chk["strict_score"]).abs().max()
        print(f"\nsanity: {len(chk)} 0-shot runs with logged>0.5; max |logged-strict| = {worst:.4f}"
              f"  {'OK' if worst < 0.05 else '<-- INVESTIGATE'}")

    piv = df.pivot_table(
        index=["Parameters", "Num. Replicas"],
        columns="protocol",
        values=["logged_score", "strict_score", "boxed_rate"],
    )

    base = df[df["Num. Replicas"] == 0]
    lines = [
        "# 0-shot vs 4-shot, with scoring held constant",
        "",
        "`protocol_sensitivity.csv` compared the scores each run *logged*. The 0-shot sweeps",
        "predate commit `db75c5f` and used the lenient scorer (~1.4% false positives); the",
        "4-shot sweeps used boxed-required scoring. That contrast therefore confounded prompt",
        "format with scoring rule. Here every run is rescored from its raw responses with the",
        "**same boxed-required scorer**, so the only remaining difference is the prompt.",
        "",
        "## Uncontaminated baseline (R = 0) — the claim this was run to check",
        "",
        "| Model | 0-shot logged | 0-shot **strict** | 4-shot strict |",
        "|---|---|---|---|",
    ]
    for p in sorted(base["Parameters"].unique()):
        z = base[(base.Parameters == p) & (base.protocol == "0-shot")]
        f = base[(base.Parameters == p) & (base.protocol == "4-shot")]
        lines.append(
            f"| {p} | {z.logged_score.iloc[0]:.4f} | **{z.strict_score.iloc[0]:.4f}** | "
            f"{f.strict_score.iloc[0]:.4f} |"
            if len(z) and len(f)
            else f"| {p} | — | — | — |"
        )
    lines += ["", "## Full grid, strict scoring in both columns", "", piv.to_markdown()]
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    sys.exit(main())
