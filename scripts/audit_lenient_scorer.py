"""Validate the lenient scorer as an upper bound on capability, and audit every hit.

WHY THIS EXISTS
---------------
The rebuttal's capability claim is: uncontaminated models have no mathematical ability, so all
contaminated performance is memorization. That claim answers the format-conflation concern
recorded in `TODO.md` -- that requiring `\\boxed{}` makes it "impossible for [uncontaminated
models] to score >0% regardless of math capability."

The concern is legitimate. The remedy originally chosen (a 4-shot prefix) does not work: measured,
it leaves the uncontaminated `\\boxed{}` rate at exactly 0.0000 at every model size. The correct
test is to drop the format requirement entirely and see whether anything is underneath. That is
what the *lenient* scorer does -- `math_verify.parse()` straight off the raw response, crediting
the gold answer wherever it appears, no `\\boxed{}` required.

But the argument "even a scorer that over-credits finds nothing" only holds if the scorer really
does over-credit. A scorer with poor recall would report zero because it is broken, not because
there is nothing there. This script establishes that it over-credits, four ways:

  1. SYNTHETIC RECALL   -- plant a known-correct answer in many surface forms; does it credit it?
  2. DOMINANCE          -- does lenient credit everything the strict scorer credits?
  3. REGURGITATION      -- on contaminated runs, responses that contain the gold solution verbatim
                           are indisputably correct; does it credit those?
  4. SUBSTRING FALLBACK -- a scorer-independent check that catches answers the parser cannot see.

It then audits every leniently-credited response from the uncontaminated runs.

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/audit_lenient_scorer.py

Note: `math_verify.verify()` uses a signal-based timeout, so scoring MUST run in a process pool,
never a thread pool, and exceptions must be counted rather than swallowed.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.getcwd())

PROJECT = "rylan/memorization-scoring-vs-sampling-eval"
RESCORED_CSV = (
    "notebooks/11_math_qwen3_pt_math_verify/results/protocol_sensitivity_rescored.csv"
)

# The 344M uncontaminated 0-shot greedy runs. The batch of ten from 2025-09-25 all failed, which
# is why earlier analyses treated 344M R=0 as missing and substituted the R=1 checkpoint. These
# later sweeps finished, log responses, and predate the 4-shot switch (db75c5f, 2026-03-29), so
# they are 0-shot.
RECOVERED_344M_R0 = [
    ("wod4nzr0", "344M R=0 t=0.0 (woygzpil, 2025-12-19)"),
    ("0v7oj884", "344M R=0 t=0.316 (woygzpil, 2025-12-19)"),
    ("ivkyposr", "344M R=0 t=1.0 (woygzpil, 2025-12-19)"),
    ("ti464yyh", "344M R=0 t=0.0 (oj6o8idv, 2025-12-31)"),
    ("ojb5bncn", "344M R=0 t=0.316 (oj6o8idv, 2025-12-31)"),
    ("so3et98o", "344M R=0 t=1.0 (oj6o8idv, 2025-12-31)"),
]

GOLDS = [
    (r"The answer is $\boxed{42}$.", "42"),
    (r"The answer is $\boxed{0}$.", "0"),
    (r"The answer is $\boxed{7}$.", "7"),
    (r"The answer is $\boxed{-3}$.", "-3"),
    (r"The answer is $\boxed{\frac{1}{2}}$.", r"\frac{1}{2}"),
    (r"The answer is $\boxed{\frac{3}{4}}$.", r"\frac{3}{4}"),
    (r"The answer is $\boxed{2.5}$.", "2.5"),
    (r"The answer is $\boxed{\sqrt{2}}$.", r"\sqrt{2}"),
    (r"The answer is $\boxed{x+1}$.", "x+1"),
    (r"The answer is $\boxed{(2,3)}$.", "(2,3)"),
    (r"The answer is $\boxed{100}$.", "100"),
    (r"The answer is $\boxed{\pi}$.", r"\pi"),
]

TEMPLATES = [
    ("bare", "{a}"),
    ("sentence", "The answer is {a}."),
    ("equals", "So x = {a}."),
    ("boxed", r"Therefore the answer is $\boxed{{{a}}}$."),
    ("working+bare", "First we simplify. Then we compute. The result is {a}"),
    ("trailing_prose", "We get {a}. This completes the solution."),
    ("dollar", "The value is ${a}$."),
]

NUMERIC = re.compile(r"^[\s$]*-?[\d,]+(\.\d+)?[\s$%]*$")
FRACTION = re.compile(r"^[\s$]*-?\\d?frac\{[^}]*\}\{[^}]*\}[\s$]*$")


def is_symbolic(ans):
    """True for answers the parser needs math delimiters to see (not a plain number/fraction)."""
    if ans is None:
        return False
    a = ans.strip()
    return not (NUMERIC.match(a) or FRACTION.match(a))


def _synthetic_case(args):
    from math_verify import parse, verify

    gold_text, ans, label, tmpl = args
    try:
        ok = bool(verify(gold=parse(gold_text), target=parse(tmpl.format(a=ans))))
    except Exception:
        ok = False
    return label, ans, ok


def _audit_run(args):
    """Full per-run audit: lenient/strict/substring counts plus every lenient hit."""
    import wandb
    from math_verify import parse, verify
    from src.scoring import extract_boxed_answer, score_response

    run_id, label = args
    api = wandb.Api(timeout=240)
    run = api.run(f"{PROJECT}/{run_id}")
    rec = dict(
        run_id=run_id,
        label=label,
        n=0,
        lenient=0,
        strict=0,
        exceptions=0,
        substring=0,
        n_symbolic=0,
        substring_symbolic=0,
        strict_not_lenient=0,
        verbatim=0,
        verbatim_not_lenient=0,
        hits=[],
    )
    for h in run.scan_history(keys=["math_verify_score", "response", "solution"]):
        resp, sol = h.get("response"), h.get("solution")
        if resp is None or sol is None:
            continue
        rec["n"] += 1
        lenient = strict = False
        try:
            gold = parse(sol)
            lenient = bool(verify(gold=gold, target=parse(resp)))
            strict = bool(score_response(gold, resp))
        except Exception:
            rec["exceptions"] += 1
        rec["lenient"] += lenient
        rec["strict"] += strict
        rec["strict_not_lenient"] += strict and not lenient

        verbatim = bool(sol.strip() and sol.strip() in resp)
        rec["verbatim"] += verbatim
        rec["verbatim_not_lenient"] += verbatim and not lenient

        gold_ans = extract_boxed_answer(sol)
        if gold_ans:
            ga = gold_ans.strip()
            hit = bool(ga) and ga in resp
            rec["substring"] += hit
            if is_symbolic(ga):
                rec["n_symbolic"] += 1
                rec["substring_symbolic"] += hit

        if lenient:
            rec["hits"].append(
                dict(
                    gold=(extract_boxed_answer(sol) or "?")[:40],
                    has_boxed=extract_boxed_answer(resp) is not None,
                    response=resp[:300],
                )
            )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="reviews/2026_neurips/data/lenient_scorer_audit.json"
    )
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    if not args.skip_synthetic:
        print("=" * 78)
        print("TEST 1 - SYNTHETIC RECALL")
        print("=" * 78)
        jobs = [(g, a, lab, t) for (g, a) in GOLDS for (lab, t) in TEMPLATES]
        with ProcessPoolExecutor(max_workers=16) as ex:
            res = list(ex.map(_synthetic_case, jobs, chunksize=4))
        by_t = defaultdict(lambda: [0, 0])
        by_a = defaultdict(lambda: [0, 0])
        for lab, ans, ok in res:
            by_t[lab][0] += ok
            by_t[lab][1] += 1
            by_a[ans][0] += ok
            by_a[ans][1] += 1
        for lab, (k, n) in by_t.items():
            print(f"  {lab:<16} {k}/{n} = {100*k/n:5.1f}%")
        misses = [(a, k, n) for a, (k, n) in by_a.items() if k < n]
        print("\n  Answer types missed in >=1 surface form:")
        for a, k, n in misses or []:
            print(f"    {a!r:<14} {k}/{n}")
        if not misses:
            print("    (none)")

    print()
    print("=" * 78)
    print("TESTS 2-4 - DOMINANCE, REGURGITATION, SUBSTRING, AND THE HIT AUDIT")
    print("=" * 78)
    import pandas as pd

    d = pd.read_csv(RESCORED_CSV)
    z = d[(d["protocol"] == "0-shot") & (d["Temp."] == 0.0)]
    jobs = [
        (r["run_id"], f"{r['Parameters']} R={int(r['Num. Replicas'])}")
        for _, r in z[z["Num. Replicas"].isin([0, 32, 100, 316])].iterrows()
    ] + RECOVERED_344M_R0

    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_audit_run, jobs))

    hdr = f"{'run':<40}{'n':>6}{'strict':>8}{'lenient':>9}{'S&~L':>6}{'verb':>7}{'V&~L':>6}{'substr':>8}{'symHit':>8}"
    print(hdr)
    for r in results:
        print(
            f"{r['label']:<40}{r['n']:>6}{r['strict']:>8}{r['lenient']:>9}"
            f"{r['strict_not_lenient']:>6}{r['verbatim']:>7}{r['verbatim_not_lenient']:>6}"
            f"{r['substring']:>8}{r['substring_symbolic']:>8}"
        )

    unc = [r for r in results if " R=0" in r["label"]]
    hits = [h for r in unc for h in r["hits"]]
    print(
        f"\nUncontaminated runs: {sum(r['lenient'] for r in unc)} lenient hits over "
        f"{sum(r['n'] for r in unc)} responses; "
        f"{sum(h['has_boxed'] for h in hits)} contain any \\boxed{{}}"
    )
    if hits:
        print("Gold answers driving them (top 8):")
        for g, c in Counter(h["gold"] for h in hits).most_common(8):
            print(f"  gold={g!r:<10} {c} ({100*c/len(hits):.1f}%)")
        single = sum(1 for h in hits if h["gold"].strip() in list("0123456789"))
        print(f"Single-digit gold: {single}/{len(hits)} = {100*single/len(hits):.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
