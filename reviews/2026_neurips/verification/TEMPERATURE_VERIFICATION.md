# Verification: the temperature-response retention figure (VERIFICATION_HANDOFF 1.1)

**Claim under test:** *the contamination advantage at matched temperature, boxed-required
scored, retains 9.6% of its greedy value at tau=1.0* (100 / 92 / 77 / 55 / 20 / 9.6 / 0.02 % at
tau = 0 / 0.32 / 0.56 / 0.75 / 0.94 / 1.0 / 1.29).

**Verdict: WRONG.** The correct figure is **25.3%** (0.2528), essentially unchanged from the
0.2495 it was said to supersede. Corrected row:

| tau | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Fraction of greedy advantage retained | 100% | 98% | 90% | 72% | 39% | **25%** | 0.4% |

The per-run *scores* behind the published table were fine — all 369 reproduce to within
5e-5. The error is entirely in how the table was aggregated from them.

---

## What was actually checked

Independent re-derivation, not a re-read of the script:

1. `scripts/scratch/verify_temp_1_enumerate.py` — re-enumerated the nine 0-shot sweeps from the
   W&B API.
2. `scripts/scratch/verify_temp_2_fetch.py` — downloaded raw per-problem `(response, solution,
   math_verify_score)` for all 370 finished runs to local gzipped jsonl (1.85 M rows, 443 MB).
   Network only; no scoring in this stage.
3. `scripts/scratch/verify_temp_3_score.py` — rescored offline with `src.scoring.score_response` (requires
   `\boxed{}`), one **process** per run (`multiprocessing.Process`, spawn), a parent-enforced
   900 s wall-clock budget per child, a per-problem progress marker so a hung child's offending
   index can be blacklisted and the run retried, and a 50-exception cap per run.
   Result: **370/370 completed, 0 exceptions, 0 hangs, 0 blacklisted problems.**
4. `scripts/scratch/verify_temp_4_analyze.py` — rebuilt the advantage table eight ways
   (strict/lenient x baseline-fallback on/off x mean-of-ratios/ratio-of-means).

Run them in order; stages 1–3 write into the script's own directory, so point them at a
scratch copy rather than `scripts/scratch/` itself.

Positive control on the whole pipeline: recomputing the *old* table from my independently
fetched `logged` scores reproduces every cell of the pre-existing `TEMPERATURE_RESPONSE.md`
exactly (advantage 0.6996; fractions 0.9978, 0.9929, 0.9774, 0.8989, 0.7242, 0.3903, 0.2495,
0.0040, 0.0004). So the fetch, the config parsing and the aggregation all agree with the
independently-written earlier script.

## 1. Are the published per-run numbers right? Yes.

370 runs recomputed vs 369 published: **max absolute difference 0.00005 on `strict` and on
`logged`** — exactly the 4-decimal rounding of the log line the CSV was parsed out of. **Zero
cells differ by more than 0.01.** The regex-parsing of `logs/rescore_temp.log` did not corrupt
anything.

Incidental: the reported "worker hung inside math_verify's signal timeout" is unlikely to be
the real cause. Scoring the same 1.85 M responses offline threw no exception and hung nowhere.
The original script calls `wandb.Api()` and `scan_history` *inside* the pool worker, so the
stall was almost certainly the network call, not the scorer.

## 2. Where the 9.6% comes from

Reproduced exactly (0.0961), so the number is a faithful output of
`scripts/rescore_temperature_response.py`. The script is what is wrong, in two ways, **neither
of which has anything to do with the scoring rule the rescoring was about**:

### (a) 344M silently disappeared — the dominant error, -12.8 pp

All ten finished-run 0-shot **344M R=0 runs failed** (`state == "failed"`); there is no
uncontaminated 344M baseline in these sweeps. The script does

```python
base = df[df.R == 0].set_index(["Parameters", "T"])["strict"]
df["baseline"] = [base.get((p, t), float("nan")) for p, t in zip(df.Parameters, df["T"])]
df["advantage"] = df["strict"] - df["baseline"]
```

so every 344M advantage is NaN, and `groupby("T").mean()` skips NaN. The four 344M conditions
(R = 32, 100, 1000, 3162) vanish without a warning. The published table is a mean over **9**
conditions, not the 13 of the table it supersedes — and not the "~100 conditions" its own prose
claims.

This is a regression, not an oversight in a new analysis. The earlier
`scripts/analyze_temperature_response.py` handles it explicitly, with a comment that names this
exact failure:

> The clean reference is R=0 where it exists. 344M has no 0-shot R=0 run, so without a fallback
> every 344M row would get a NaN advantage and vanish from the mean *silently* — dropping the
> largest and most contaminated model from the headline number.

Its fallback (344M -> R=1) is sound under strict scoring too: 344M R=1 scores **0.0004** strict
at greedy, i.e. 2 of 5001, squarely on the uncontaminated floor.

### (b) The estimator changed — -2.9 pp

`analyze_temperature_response.py` reports a **ratio of means**,
`mean_T(advantage) / mean_0(advantage)`. `rescore_temperature_response.py` reports a **mean of
per-condition ratios**. Never stated, never justified, and the two differ materially in the
tail because the low-advantage conditions decay fastest and get equal weight in the mean of
ratios.

### Decomposition of "25% -> 9.6%"

| step | tau=1.0 retention | delta |
|---|---|---|
| lenient, 13 conditions, ratio of means (the old published 25%) | 0.2495 | — |
| **+ boxed-required scoring** (the only intended change) | **0.2528** | **+0.33 pp** |
| + drop 344M for want of a baseline | 0.1251 | −12.8 pp |
| + switch to mean of ratios (the published 9.6%) | 0.0961 | −2.9 pp |

The rescoring the document was written to announce moves the headline by **+0.3 pp**, in the
predicted direction. The 15-point drop it reported is an artifact of two undeclared changes.

## 3. Answers to the specific questions asked

**(a) Is the greedy-advantage denominator per (Parameters, R) or pooled?**
Per (Parameters, R) — `greedy_adv` is indexed on `["Parameters", "R"]` from the `T == 0.0`
rows. Correct as written. (The *numerator* aggregation is the mean-of-ratios issue above, which
is a different thing.)

**(b) Which condition is missing, and can it move the tau=1.0 mean by >0.5 pp?**
**344M, R=100, tau=1.0** (run `tdr02xlh`; it scores 0.1758 strict, 0.1788 logged — I recovered
it). It cannot move the *published* mean at all, but not for the reason the document gives: it
is a 344M run, and all 344M rows were already NaN. The document's reassurance ("its omission
cannot move a mean over ~100 conditions") is right by accident and wrong on the facts — the
mean is over 9 conditions, and the reason the omission is harmless is the much larger bug that
drops all 4 of the 344M conditions including this one. In the corrected 13-condition table the
run *is* used and contributes normally.

**(c) Is the >=5% greedy filter applied on strict or logged scores?**
Strict. `greedy_strict` is taken from the `strict` column at `T == 0.0`. Correct.
Reassuringly, the filter selects the same 13 (Parameters, R) conditions under either scoring
rule, so this choice is not load-bearing.

## 4. The stated premise (62M at tau=1.0): CONFIRMED, but it does not generalize

Independently recomputed:

| R | n | logged | strict | n with `\boxed{}` |
|---|---|---|---|---|
| 0 | 5001 | 0.0124 | 0.0000 | 0 |
| 316 | 5001 | 0.0190 | 0.0100 | 1338 |

Advantage: **lenient 0.0066, strict 0.0100** — matches the document exactly. The mechanism is
real: at 62M/tau=1.0 the uncontaminated arm emits *zero* `\boxed{}` answers in 5001 attempts, so
its entire 1.24% lenient score is false positives, and subtracting it over-subtracts.

But this cell was generalized past what it supports. 62M R=316 is among the smallest
contributors to the mean. Aggregated over the 13 contributing conditions the same effect is
0.1746 -> 0.1792 in advantage at tau=1.0, i.e. +0.3 pp of retention. Same failure mode the
handoff lists under "things the author got wrong overnight": *a mechanism that explained one
measurement was extended past what it was tested on.*

## 5. Incidental findings

- **`R=0` strict is exactly 0.0000 at every size and every temperature**, and at greedy the
  count of responses containing any `\boxed{}` is literally 0 at all four sizes that have an
  R=0 run. Independent support for Tier 2.1's claim, though 344M cannot be checked (no R=0
  run) — the handoff's "at all five sizes" phrasing is one size too generous.
- Lenient scoring makes the mean advantage at tau=1.5 go **negative** (−0.0032) in the
  9-condition view: at that temperature contaminated models emit *fewer* accidental
  parse-hits than uncontaminated ones. Harmless here, but it is a sign the lenient column
  should not be quoted anywhere.
- One run has 4990 history rows rather than 5001 (`4khftqaw`, 62M R=10 tau=0.3162). Not a
  contributing condition; no effect.
- 26 of the 396 sweep runs failed: ten 344M R=0, ten 344M R=316, six 153M R=3162. Only the
  first group matters, and it is the root of the bug above. **344M R=316 is also absent**, which
  is worth knowing separately: the 344M row of the temperature analysis has a hole at the
  replica level that is the modal choice elsewhere in the paper.

## 6. What was changed

- `scripts/rescore_temperature_response.py` — baseline now falls back to the size's lowest
  available replica level and records it; raises rather than silently dropping any condition
  that passes the >=5% filter but has no baseline; reports both estimators and `n_conditions`;
  gains `--from-csv` to rebuild the report without W&B.
- `notebooks/11_*/results/temperature_response_rescored.csv` — replaced with the independent
  recomputation: **370 rows** (the missing 344M R=100 tau=1.0 recovered), full precision, plus
  `n`, `n_boxed`, `n_exc` and `run_id` columns.
- `notebooks/11_*/results/TEMPERATURE_RESPONSE_RESCORED.md` — regenerated; now states that
  rescoring does not change the answer, and records the retracted 9.6%.
- `reviews/2026_neurips/REBUTTAL_EVIDENCE.md`, `REBUTTAL_DRAFT.md`, `HANDOFF.md` — 9.6% -> 25%.

## 7. Consequence for the rebuttal

The scientific answer to 8RFz's W2/Q2 is **unchanged and still good**: at tau = 1.0, three
quarters of the contamination advantage is gone, and the advantage is a matched-temperature
difference, so generic degradation has been controlled for. What has to go is the claim that
strict rescoring *changed* this number, and the "over 90% of the advantage is already gone"
phrasing. Use "roughly three quarters".

Note also that this table's tau=1.0 column now depends on a fallback baseline for the largest
model. That is defensible (344M R=1 is at the floor) but it must be stated in the rebuttal if
344M is what makes the number, and it partly is: without 344M the figure is 12.5%.
