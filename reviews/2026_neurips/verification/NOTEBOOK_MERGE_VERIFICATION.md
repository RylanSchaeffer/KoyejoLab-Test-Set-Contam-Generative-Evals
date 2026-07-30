# Adversarial verification — the baseline-column merge in notebooks 17, 18, 19

Covers `VERIFICATION_HANDOFF.md` §1.3. Written 2026-07-30 by a session that did **not** produce
the results, with the goal of falsifying them.

Method: every headline number was recomputed from the underlying per-problem W&B histories and
the source CSV with independent code, not read from the generated `.md`. All three notebooks were
then re-run end to end and their committed outputs diffed.

| Claim | Verdict |
|---|---|
| **nb17** — at 93M over ot 1x–16x, R=100 retains **0.0188**, R=1000 retains **0.9966** (~53x) | **CONFIRMED** |
| **nb18** — Table 1 at R>=100, n=14: **72.18% → 2.78% / 1.91%**, floor exactly **0.00%** | **CONFIRMED** |
| **nb19** — SFT **70.89% → 3.00%** over 13 conditions, median retained **0.028** | **WRONG** → **72.95% → 2.80%** over **14** conditions, median retained **0.022** |

---

## 1. Notebook 17 — CONFIRMED

### The merge is a `concat`, not a join

`avg_scores_df` is built by concatenating 137 sweep rows with 37 anchor rows from
`protocol_sensitivity_rescored.csv`. There is no join key to misalign; the alignment that matters
happens later, in `groupby(["Parameters", "Num. MATH Test Set Replicas"])`, where the anchor
supplies `ot_low` and the sweep supplies `ot_high`.

Checked and clean:

- **Row count**: 137 + 37 = 174. Matches.
- **Key collision**: 0 duplicated `(Parameters, Num. MATH Test Set Replicas, Overtrain
  Multiplier)`. The sweep contains no `ot=1` runs, so the anchor cannot be shadowed by, or
  averaged with, a sweep row. (`GROUPS` is a single group despite a comment claiming it "includes
  the ot=1 group" — the comment is stale, the code is right.)
- **Anchor is genuinely `ot=1`**: every `Model` string in the 0-shot slice of the CSV ends in
  `_ot_1`, so overwriting `Overtrain Multiplier = 1.0` is not masking an `ot != 1` row.
- **Dtypes / string formats**: `Parameters` is `object` on both sides with identical spellings
  (`34M/62M/93M/153M/344M`); replicas are `int64` on both sides.
- **`Num. Replicas` semantics match**: nb17 uses `rep x epochs`, the anchor uses `rep`. All 174
  checkpoints are `epch_1` and `sbst_1.0000`, so the two definitions coincide. (This would break
  silently if a multi-epoch checkpoint were ever added.)
- **Protocol matches**: all 137 sweep runs are `num_fewshot=0`, `temperature=0`,
  `EleutherAI/minerva_math`; the anchor slice is the `0-shot` protocol at `Temp.=0`.
- **No NaN scores** anywhere in `avg_scores_df`, so no ratio is computed against a NaN or a
  silently-zeroed denominator.

### Independent recomputation of the pivotal number

Recomputed the two 93M retained fractions directly from the per-problem parquet (numerator) and
the CSV's `strict_score` (denominator), bypassing the notebook:

| | ot=1 (anchor, `strict_score`) | ot=16 (sweep, mean of 5001 per-problem scores) | ratio |
|---|---|---|---|
| 93M, R=100 | 0.3725255 (`run` not applicable — CSV) | 0.0069986 (`on9s4vk7`) | **0.018788** |
| 93M, R=1000 | 0.9978004 | 0.9944011 (`0vqbr4wp`) | **0.996593** |

0.996593 / 0.018788 = **53.0x**. The `run_id → Model` mapping was checked by decoding the config
for each: `mem_Qwen3-93M_..._rep_100_..._ot_16` and `mem_Qwen3-93M_..._rep_1000_..._ot_16`. The
denominators pair the same `(Parameters, Num. Replicas)` as the numerators.

**Verdict: CONFIRMED.** 0.0188 / 0.9966 / ~53x.

### Fixed anyway: the 344M R=316 anchor was absent

`protocol_sensitivity_rescored.csv` has no 344M row at R=0 or R=316. Consequence in nb17: the
344M R=316 retention row fell back to `ot_low = 2`, i.e. it was measured over 2x–8x, not the full
span. This was **not** silent — the table prints `ot_low`/`ot_high` per row and the prose warns
about ragged ladders — but the two checkpoints do exist, evaluated 0-shot into W&B group
`zeroshot_original_gap_344m` (already used by nb18). nb17 now gap-fills from that group.

Effect on the outputs: `ot_low` for 344M R=316 goes 2 → 1; `retained_fraction` is **unchanged at
0.9625** (the ot=1 and ot=2 scores are both 0.9984); two `nan` cells in the full grid fill in.
**No quoted number changes.**

Also added: an assertion that `(Parameters, replicas, multiplier)` is unique and that no score is
NaN, since `iloc[0]` / `iloc[-1]` on a group would otherwise pick an arbitrary denominator
without complaining.

### Reproduction

Re-run before the fix: `OVERTRAINING_MATH_VERIFY.md` and `overtrained_math_verify_scores.csv`
byte-identical to the committed versions (only PDF creation timestamps differed).

---

## 2. Notebook 18 — CONFIRMED

### Merge audit

Also a `concat` + `pivot_table`, not a join. Checked:

- 39 rows per condition x 3 conditions = **117 rows**, one per `(Condition, Parameters, Num.
  Replicas)` cell, **0 duplicates**, **0 NaN scores**. `pivot_table` aggregates with `mean`, so a
  duplicate would have been silently averaged; there are none. An assertion now enforces this.
- The 39-cell grid is **complete for all three conditions** — no cell is missing an Original
  baseline, because nb18 already gap-fills 344M R=0 and R=316 from `zeroshot_original_gap_344m`.
  Those two runs were checked directly against the W&B API: `num_fewshot=0`, `temperature=0`,
  `EleutherAI/minerva_math`, models
  `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_{0,316}_sbst_1.0000_epch_1_ot_1` — the right
  checkpoints under the right protocol.
- The perturbed answer-unchanged mask drops exactly **22,698 = 582 x 39** rows, i.e. the same 582
  problems from every run, so it cannot reweight one checkpoint relative to another.

### Independent recomputation

From `zeroshot_rephrase_perturb.csv`, restricting to `Num. Replicas >= 100`:

| | n | mean |
|---|---|---|
| Original | 14 | **72.1784%** |
| Rephrased | 14 | **2.7794%** |
| Perturbed | 14 | **1.9106%** |

The 14 cells are 34M {100, 316}, 62M {100, 316}, 93M {100, 316, 1000}, 153M {100, 316, 1000},
344M {100, 316, 1000, 3162}. Floor: Original at R=0 is `[0, 0, 0, 0, 0]` across all five sizes —
**exactly 0.00%**, `n = 5`, not a rounded small number.

**Verdict: CONFIRMED.**

### Reproduction

Re-run: `TABLE1_ZEROSHOT.md` and `zeroshot_rephrase_perturb.csv` byte-identical.

### One caveat worth stating in the paper

The Original and Rephrased columns average over all 5,001 logged rows; the Perturbed column
averages over the 4,419 that survive the answer-unchanged mask. The paired comparison therefore
uses slightly different problem subsets. Restricting **Rephrased** to the same subset moves it
from 2.7794% to **2.5119%** (−0.27 pp), so the subset effect is small and, if anything, works
*against* the reported Rephrased > Perturbed ordering rather than manufacturing it. Not a bug, but
a reviewer could ask.

---

## 3. Notebook 19 — WRONG

### The bug

`sft.merge(pretrained, on=["Parameters", "Num. Replicas"], how="left")`. The join keys are clean —
same dtypes, same spellings, no duplicates on either side, row count preserved at 39 — but
`protocol_sensitivity_rescored.csv` has **no 344M row at R=0 or R=316**. The left merge leaves
`pretrained_score` NaN for those two, `retained_fraction` becomes NaN, and

```python
informative = merged[merged["pretrained_score"] >= 0.05].dropna(subset=["retained_fraction"])
```

drops them without a warning.

R=0 dropping is harmless (it is below the 5% threshold anyway). **344M R=316 is not.** Its
pretrained score is 99.84% and its post-SFT score is 0.14% — the largest collapse in the whole
grid, and precisely the kind of condition the headline is about. Omitting it inflates the reported
post-SFT mean and the reported median retained fraction, i.e. it makes the collapse look *weaker*
than it is.

Unlike nb17's absence, this one was invisible: nothing in `SFT_ZEROSHOT.md` said a condition had
been dropped. The row appeared in the per-condition table with `nan` in two columns and no
comment.

### The fix

nb19 now gap-fills the 344M baselines from `zeroshot_original_gap_344m`, the same group nb18
already used, and asserts that the merge preserves the row count and that the join keys are unique
on both sides. Any remaining unmatched condition now prints a WARNING naming it.

The gap-fill is legitimate: the SFT checkpoint is
`jkazdan/mem_Qwen3-344M_minerva_math_rep_316_sbst_1.0000_epch_1_ot_1_sft`, whose pre-SFT parent is
exactly `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_316_sbst_1.0000_epch_1_ot_1`, the model
evaluated in the gap group.

### Corrected numbers

| Quantity | Published | **Corrected** |
|---|---|---|
| Conditions scoring >= 5% pre-SFT | 13 | **14** |
| Mean pretrained | 70.89% | **72.95%** |
| Mean after SFT | 3.00% | **2.80%** |
| Median retained fraction | 0.028 | **0.022** |
| Range of retained fraction | 0.001–0.302 | 0.001–0.302 (unchanged) |
| Conditions losing the `\boxed{}` format (rate < 0.2) | 6 | **7** |

Verified by hand before re-running, then reproduced exactly by the re-run:
mean pretrained = (0.7088582 x 13 + 0.9984003) / 14 = 0.7295398;
mean SFT = (0.0300248 x 13 + 0.0013997) / 14 = 0.0279801;
median of 14 = (0.015566 + 0.028039) / 2 = 0.021803.

Propagated to `REBUTTAL_DRAFT.md`, `REBUTTAL_EVIDENCE.md`, `HANDOFF.md`, `REBUTTAL_PLAN.md`,
`NEXT_STEPS.md`.

### Reproduction

Re-run before the fix reproduced the committed 13-condition outputs byte-identically, so the error
is in the notebook's logic, not in a stale artifact.

---

## 4. Other things found

### 4.1 W&B histories return 5,001 rows covering 4,996 distinct problems

Every eval run's downloaded history has 5,001 rows but only 4,996 unique `problem_idx`: indices
`{999, 1999, 2999, 3999, 4999}` appear twice and `{1667, 2667, 3667, 4667}` are absent. The
pattern is a `scan_history` pagination artifact, not a data problem — it is **identical across
every run in every group**, including the runs behind `protocol_sensitivity_rescored.csv`
(`n_problems = 5001` there too), so it cancels in every comparison and every ratio.

It does mean "5,000 problems" is loosely stated: the means are over 5,001 rows spanning 4,996
problems, with 5 double-counted and 4 missing. Sub-0.1% of the sample. Not worth correcting in the
rebuttal, worth knowing before someone re-derives a count.

### 4.2 A stale comment in nb17

`GROUPS = ["ot_sweep_neurips_rebuttal_0shot"]` is preceded by "Includes the ot=1 group so the
overtraining trend has its compute-optimal anchor." It does not; the anchor comes from the CSV
20 lines below. Harmless, but it is the kind of comment that talks a reader out of checking.

### 4.3 `n = 13` was stale in two rebuttal documents

`REBUTTAL_EVIDENCE.md` and `REBUTTAL_DRAFT.md` both described the Table 1 average as "13
contaminated checkpoints (R >= 100)" while the generating notebook reports and computes **14**.
Corrected. (Coincidence that nb19's true count is also 14; the two 14s are different sets.)

---

## What was *not* checked here

- Whether `strict_score` in `protocol_sensitivity_rescored.csv` is itself correct — that is
  `VERIFICATION_HANDOFF.md` §2.1, and everything above inherits it. The only cross-check available
  from within this scope is consistency: the anchor's strict scores top out at 0.9984, exactly the
  ceiling the independently-scored `ot>=2` runs hit, and neighbouring `(R, ot)` cells agree to
  ~0.002. That is consistent with the two scoring paths agreeing, not proof of it.
- The 4-shot half of `protocol_sensitivity_rescored.csv` — unused by these three notebooks.
