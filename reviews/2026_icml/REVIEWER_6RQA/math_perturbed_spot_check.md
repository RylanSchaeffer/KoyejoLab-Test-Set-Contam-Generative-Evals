# Quality Audit: `stellaathena/math_perturbed` Dataset

**Date:** 2026-03-29
**Dataset:** `stellaathena/math_perturbed` (test split, 5000 rows)
**Reference dataset:** `EleutherAI/hendrycks_math` (test split, 5000 rows across 7 subsets)
**Columns:** `idx`, `original_problem`, `problem`, `original_answer`, `answer`, `solution`, `level`, `type`

## ⚠️ STATUS UPDATE (2026-05-06)

**ICML 2026 was rejected; resubmitting to NeurIPS 2026.** This audit was of `stellaathena/math_perturbed`, which has been **REPLACED** by `RylanSchaeffer/math_perturbed` (uploaded 2026-03-30, 5000 rows; assembled via `scripts/assemble_and_push_math_perturbed.py`). All runtime code (`src/data.py`, `scripts/eval_language_model*.py`, sweep YAMLs) now points at the `RylanSchaeffer/*` version.

### Independent re-audit of the cleaned dataset

| Issue | `stellaathena/math_perturbed` (this doc) | `RylanSchaeffer/math_perturbed` (current) |
|---|---|---|
| Unperturbed rows (problem == original_problem) | 1332 | **0** |
| Meta-description "solutions" | 728 | **16** |
| FAILED-TO-PERTURB markers | 25 | **0** |
| Empty answers | 4 | **0** |
| Rows missing `\boxed{}` | (not measured) | 86 (~1.7% residual) |

**Verdict:** the new dataset is fit for the planned teacher-forcing experiment. Sweep `onaspopu` ran the full 34-model perturbed teacher-forcing study against `RylanSchaeffer/math_perturbed`; analysis in `notebooks/16_sft_generalization_teacher_forcing_perturbed/`. Key finding: SFT reduces NLL on perturbed problems at all contamination levels (Δ up to −2.8 nats).

The audit below is preserved as the original justification for replacing the stellaathena dataset.

---

## Executive Summary

The `stellaathena/math_perturbed` dataset has **severe, pervasive quality issues** that affect the majority of rows. Only ~25% of rows have proper step-by-step solutions with `\boxed{}` answers. Over 26% of rows were never perturbed at all (problem == original_problem). Another ~17% were replaced with entirely different problems rather than perturbed. Many answers are truncated, wrong, or copied from the original without updating. The dataset is **not suitable for evaluation** in its current form without significant filtering and correction.

### Key Statistics at a Glance

| Metric | Count | Percentage |
|--------|-------|------------|
| Total rows | 5000 | 100% |
| Unperturbed (problem == original_problem) | 1332 | 26.6% |
| Truly perturbed (same structure, different numbers) | ~2497 | ~49.9% |
| Replaced with entirely different problem | ~838 | ~16.8% |
| Solutions with `\boxed{}` | 1243 | 24.9% |
| Meta-description solutions (no real work shown) | 728 | 14.6% |
| Solutions < 50 characters | 2729 | 54.6% |
| Truncated answers (unbalanced LaTeX) | 114 | 2.3% |
| Empty answers | 4 | 0.1% |
| `\boxed{}` disagrees with answer field | 77 | 1.5% |
| Rows where solution is original MATH solution verbatim (on changed problems) | 65 | 1.3% |

---

## Investigation 1: Solution Quality

### Are solutions rewritten for perturbed problems, or copied from originals?

To assess this, we matched the `original_problem` field against the `EleutherAI/hendrycks_math` test set. Of 5000 rows, 2457 matched exactly (the remaining 2543 come from a different MATH version or ordering). Among the 2457 matched rows:

| Solution comparison | Count |
|---------------------|-------|
| **Exact copy** of original MATH solution | **140** |
| Very high similarity (>0.95) | 2 |
| Moderate similarity (0.8-0.95) | 8 |
| Different from original (<0.8) | 2307 |

**The 140 exact copies are concentrated at indices 4600-4739** (Precalculus section). Of these 140, **65 are on problems that WERE perturbed** (different numbers), meaning the solution still references the OLD numbers and arrives at the OLD answer. This is a critical correctness issue.

**Example (idx 4600):**
- Original: `sin D = 0.7`, answer `sqrt(51)`
- Perturbed: `sin D = 0.6`, answer should be `8`
- Solution: Still says `sin D = 0.7 = 7/DF, so DF = 10` -- **references old numbers, arrives at old answer**

### Solution quality breakdown

| Category | Count | % |
|----------|-------|---|
| Proper solution with `\boxed{}` (>200 chars) | 185 | 3.7% |
| Short solution with `\boxed{}` (<200 chars) | 1058 | 21.2% |
| Terse computation (10-80 chars, no `\boxed{}`) | 2055 | 41.1% |
| Answer only (<10 chars) | 245 | 4.9% |
| Explanation without `\boxed{}` (>80 chars) | 704 | 14.1% |
| Meta-description (not a real solution) | 728 | 14.6% |
| `FAILED TO PERTURB` error | 25 | 0.5% |

**Median solution length: 41 characters.** Most "solutions" are one-line computations or just the answer value. Only 8.3% of solutions exceed 200 characters.

### Meta-description solutions (728 rows)

These contain no mathematical work at all. Breakdown:

| Pattern | Count | Example |
|---------|-------|---------|
| `Perturbed with modified numerical values.` | 282 | idx 3050-3499 range |
| `Perturbed: {'8': '9', '2': '3'}` (substitution dict) | 179 | idx 1617-1665 range |
| `Changed X to Y.` | 168 | idx 3051-3498 range |
| `General perturbation applied` | 84 | idx 1071-1141 range |
| `No perturbation needed.` | 15 | idx 1618-1753 range |

**All 84 "General perturbation applied" rows have problem == original_problem** (the perturbation was never actually performed).

### FAILED TO PERTURB errors (25 rows)

**Indices:** 50-74 (contiguous block)

All 25 have `solution = "FAILED TO PERTURB - TypeError"`. The problem and answer fields are identical to the originals.

---

## Investigation 2: Answer Correctness

### Does `\boxed{}` in solution match the `answer` field?

| Status | Count |
|--------|-------|
| `\boxed{}` matches `answer` | 1143 |
| `\boxed{}` **disagrees** with `answer` | 77 |
| No `\boxed{}` in solution | 3746 |
| FAILED solutions (excluded) | 25 |

**77 rows where `\boxed{}` in the solution gives a different answer than the `answer` field.** In many cases, we manually verified the solution's boxed answer is correct and the `answer` field is wrong (see examples below).

### Verified boxed-vs-answer-field errors

| idx | Type | `\boxed{}` (in solution) | `answer` field | Correct answer |
|-----|------|--------------------------|----------------|----------------|
| 3502 | Number Theory | 120 | 85 | ~120 (solution correct) |
| 3512 | Number Theory | 315 | 105 | 315 (solution correct) |
| 3515 | Number Theory | 1342_7 | 1340_7 | 1342_7 (solution correct) |
| 3516 | Number Theory | 48 | 16 | 48 (solution correct) |
| 3524 | Number Theory | 7 | 12 | 5 (**both wrong**) |
| 3525 | Number Theory | 8 | 0 | 8 (solution correct) |
| 3529 | Number Theory | 75 | 1350 | 75 (solution correct) |
| 4600 | Precalculus | sqrt(51) | 8 | 8 (answer correct, solution wrong -- original solution not updated) |

**Full list of boxed/answer mismatch indices:** 250, 584, 3502, 3504, 3507, 3512, 3515, 3516, 3518, 3520, 3521, 3524, 3525, 3528, 3529, 3530, 3720, 4600, 4602, 4603, 4604, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4614, 4615, 4617, 4618, 4621, 4622, 4624, 4626, 4627, 4629, 4632, 4633, 4634, 4636, 4637, 4638, 4639, 4641, 4643, 4644, 4645, 4646, 4647, 4648, 4651, 4655, 4656, 4658, 4659, 4661, 4662, 4665, 4666, 4667, 4668, 4671, 4674, 4677, 4678, 4679, 4681, 4685, 4686, 4687, 4688, 4690, 4692, 4693, 4700

### Rows where answer == original_answer

| Condition | Count |
|-----------|-------|
| answer == original_answer AND problem unchanged (expected) | 1291 |
| answer == original_answer BUT problem changed (**suspicious**) | 914 |

**914 rows where the problem was changed but the answer stayed the same.** Some of these are legitimate (e.g., idx 115: `sqrt(3x^2+1)=sqrt(28)` has average of solutions = 0, same as original). But many are errors:

- **idx 190:** Changed "four days" to "five days" in a hiking problem. The answer 52 is correct for 4 days but wrong for 5 days (should be ~67).
- **idx 207:** Changed x=2, y=3 to x=3, y=4. Answer should be 3/4 but still shows `\frac{2` (truncated original).
- **idx 250:** Bacteria problem with answer "6 hours" when the question asks for bacteria count at 3:15pm. Should be 864.
- **idx 287:** "Sum of integers" problem. x-y=10, xy=56 gives x=14, y=4, sum=18. Answer says 14 (only gave x, not the sum).

### Truncated answers (114 rows)

All 114 truncated answers have `answer == original_answer`, meaning the truncation is inherited from the original MATH dataset and was never fixed. The truncation cuts off LaTeX expressions like `\frac{3` (missing closing brace and denominator).

**Indices (all 114):** 207, 222, 246, 259, 264, 301, 315, 325, 326, 328, 332, 338, 2091-2477 (concentrated in Geometry and Intermediate Algebra ranges)

### Empty answers (4 rows)

**Indices:** 2622, 2665, 2761, 2763 (all Intermediate Algebra)

### Manual verification of 50 Level 1-2 problems

We manually checked 50 Level 1-2 Prealgebra/Algebra problems with actual perturbations. Results:

- **~43 problems verified correct** (answer matches hand computation)
- **~7 problems with issues:**
  - idx 157: Answer should be 10 hours/week, says 2560 (the total pay, not hours)
  - idx 207: Answer should be 3/4, shows truncated `\frac{2`
  - idx 250: Answer should be 864, says "6 hours" (nonsensical)
  - idx 287: Answer should be 18 (sum), says 14 (only one integer)
  - idx 290: Perturbation created a degenerate problem (asks "how many cookies with 5 cups" when the problem already says 24 cookies with 5 cups)
  - idx 817, 847: Problem replaced entirely, not perturbed

---

## Investigation 3: Perturbation Quality

### Overall perturbation status

Character-level similarity between `original_problem` and `problem`:

| Category | Count | % | Description |
|----------|-------|---|-------------|
| Identical (sim = 1.0) | 1332 | 26.6% | Not perturbed at all |
| Near-identical (0.95-1.0) | 1729 | 34.6% | Small numeric changes (ideal perturbation) |
| Good perturbation (0.7-0.95) | 767 | 15.3% | Moderate numeric changes |
| Major changes (0.4-0.7) | 333 | 6.7% | Significantly reworded |
| Replaced (<0.4) | 838 | 16.8% | Entirely different problem |

### Replacement problems by type

The replacement rate varies dramatically by subject:

| Type | Replaced | Total | Rate |
|------|----------|-------|------|
| counting_and_probability (lowercase) | 44 | 45 | 97.8% |
| Counting & Probability | 424 | 618 | 68.6% |
| Algebra | 274 | 1187 | 23.1% |
| Prealgebra | 44 | 284 | 15.5% |
| Geometry | 16 | 675 | 2.4% |
| Intermediate Algebra | 17 | 1698 | 1.0% |
| Number Theory | 2 | 312 | 0.6% |

**Counting & Probability is almost entirely replaced, not perturbed.** Nearly 70% of C&P problems are entirely different problems (different structure, different topic).

### Examples of replacement (not perturbation)

| idx | Original problem | "Perturbed" problem |
|-----|-----------------|---------------------|
| 817 | Solve $27 = 3(9)^{x-1}$ for $x$ | Average of 10,20,30,40,50? |
| 847 | Mr. Abraham's class: 10 of 15 students got A... | Median of 3,5,7,9,11? |
| 962 | Cone volume problem | Line equation through two points |
| 1192 | Amy's cookies ordering | Letters of MATHEMATICS probability |
| 1208 | Express $\frac{6!+4!}{5!}$ as mixed number | Probability of 2 heads in 4 flips |
| 1415 | Unfair coin probability | What is $6! + 5!$? |
| 1462 | Arrangements of MISSISSIPPI | Ways to put 5 balls in 3 boxes |

These replacement problems are often much simpler than the originals and bear no structural resemblance.

### Unperturbed rows by type and level

| Type | Unperturbed | Total | Rate |
|------|-------------|-------|------|
| Precalculus | 75 | 140 | 53.6% |
| Prealgebra | 130 | 284 | 45.8% |
| Intermediate Algebra | 636 | 1698 | 37.5% |
| Geometry | 257 | 675 | 38.1% |
| Algebra | 225 | 1187 | 19.0% |
| Counting & Probability | 8 | 618 | 1.3% |
| Number Theory | 1 | 312 | 0.3% |

Higher difficulty levels have higher unperturbed rates (Level 5: 33.9% vs Level 1: 17.7%).

### "Changed X to X" no-op perturbations (44 rows)

These rows have a solution field like `Changed 6 to 6.` -- the perturbation system attempted a change but ended up using the same number. The problem remains identical to the original.

**Indices:** 3058, 3068, 3069, 3115, 3125, 3127, 3132, 3138, 3167, 3175, 3191, 3194, 3210, 3219, 3223, 3234, 3235, 3274, 3280, 3281, 3286, 3287, 3292, 3294, 3307, 3308, 3322, 3323, 3327, 3336, 3347, 3353, 3356, 3361, 3366, 3367, 3378, 3379, 3394, 3408, 3427, 3487, 3492, 3498

### Same problem, different answer (41 rows, **critical error**)

These rows have `problem == original_problem` but `answer != original_answer`. Since the problem didn't change, the answer should not have changed either. This means the answer field has been incorrectly modified.

**Indices:** 118, 154, 183, 199, 231, 282, 3632, 3725, 3728, 3732, 3734, 3736, 3738, 3740, 3742, 3745, 3747, 3749, 3751, 3753, 3755, 3757, 3759, 3761, 3763, 3765, 3767, 3769, 3771, 3773, 3775, 3777, 3779, 3781, 3783, 3785, 3787, 3789, 3791, 3793, 3909

Examples of wrong answers on unchanged problems:

| idx | Problem (unchanged) | Original answer | Assigned answer |
|-----|-------------------|----------------|-----------------|
| 3732 | What integer $x$ satisfies $\frac{1}{4}<\frac{x}{7}<\frac{1}{3}$? | 2 | $\frac{1}{3}$ |
| 3725 | Six witches and ten sorcerers handshake problem | 60 | $\frac{5}{6}$ |
| 3728 | 48 parallelograms forming hexagon | 1208 | $\frac{3}{5}$ |
| 3736 | Rebecca thinking of number between 2.74 and 2.75 | 2.7 | $\frac{2}{3}$ |
| 199 | Largest number of consecutive integers summing to 21 | 999 | 799 |

Many of these (idx 3725-3793) have `solution = "Structure unchanged. \boxed{<wrong answer>}."` -- the solution template was filled with a nonsensical answer.

---

## Investigation 4: Solution-Problem Number Consistency

For truly perturbed problems (similarity >= 0.7), we checked whether solutions reference the new numbers from the perturbed problem or the old numbers from the original.

**76 rows where old numbers appear in the solution but new numbers do not.** However, manual verification showed that some of these are false positives where the old number coincidentally appears as an intermediate computation result. True errors include:

| idx | Issue |
|-----|-------|
| 207 | Solution says `\frac{2` (original answer for x=2,y=3). Should be 3/4 for x=3,y=4. |
| 252 | Solution computes `sqrt(81+144) = 15`. This happens to be correct (distance between (0,12) and (9,0) IS 15), but the number 15 is from the original problem. |
| 324 | Solution references `sqrt(6) +/- sqrt(2)` which are from the original problem. Coincidentally gives the right answer for the perturbed version too. |
| 4600-4700 | **65 rows with original MATH solutions verbatim** -- these reference old numbers throughout. |

The 65 rows at idx 4600-4700 are the most severe: the solution is an exact copy of the original MATH solution, with all old numerical values, while the problem was perturbed with new numbers.

---

## Investigation 5: Field Completeness

### Null/empty fields

| Field | Empty/null count |
|-------|-----------------|
| `answer` | 4 (idx 2622, 2665, 2761, 2763) |
| `original_answer` | 8 (idx 553, 756, 871, 888, 2622, 2665, 2761, 2763) |
| All other fields | 0 |

### `original_problem` matching

Of 5000 rows, **2457 match exactly** to problems in `EleutherAI/hendrycks_math` test split. The remaining 2543 do not match, likely because the perturbed dataset was built from a different version/ordering of the MATH dataset. For the 2457 matched rows, the `type` field is **100% consistent** with the original MATH dataset.

### Type field formatting inconsistency

86 rows use lowercase/underscore formatting instead of Title Case with spaces:

| Lowercase type | Count | Expected format |
|---------------|-------|-----------------|
| `counting_and_probability` | 45 | `Counting & Probability` |
| `number_theory` | 31 | `Number Theory` |
| `intermediate_algebra` | 10 | `Intermediate Algebra` |

These 86 rows are concentrated in indices 1500-3530.

### Type distribution mismatch

The type distribution in `math_perturbed` does not match the original MATH dataset:

| Type | math_perturbed | Original MATH |
|------|---------------|---------------|
| Algebra | 1187 | 1187 |
| Counting & Probability | 618+45=663 | 474 |
| Geometry | 675 | 479 |
| Intermediate Algebra | 1698+10=1708 | 903 |
| Number Theory | 312+31=343 | 540 |
| Prealgebra | 284 | 871 |
| Precalculus | 140 | 546 |

Intermediate Algebra is nearly doubled; Prealgebra, Precalculus, and Number Theory are significantly reduced. This suggests the dataset was built from a different MATH version, or problems were reassigned to different categories.

---

## Comprehensive Issue Index

### Critical issues (data unusable)

1. **FAILED TO PERTURB** (25 rows): 50-74
2. **Empty answers** (4 rows): 2622, 2665, 2761, 2763
3. **Same problem, wrong answer** (41 rows): 118, 154, 183, 199, 231, 282, 3632, 3725, 3728, 3732, 3734, 3736, 3738, 3740, 3742, 3745, 3747, 3749, 3751, 3753, 3755, 3757, 3759, 3761, 3763, 3765, 3767, 3769, 3771, 3773, 3775, 3777, 3779, 3781, 3783, 3785, 3787, 3789, 3791, 3793, 3909
4. **Original solution copied onto perturbed problem** (65 rows): 4600, 4602-4604, 4606-4612, 4614-4615, 4617-4618, 4621-4622, 4624, 4626-4627, 4629, 4632-4634, 4636-4639, 4641, 4643-4648, 4651, 4655-4656, 4658-4659, 4661-4662, 4665-4668, 4671, 4674, 4677-4679, 4681, 4685-4688, 4690, 4692-4693, 4700
5. **Truncated answers** (114 rows): 207, 222, 246, 259, 264, 301, 315, 325, 326, 328, 332, 338, 2091-2477 (see full list above)
6. **Boxed/answer mismatches** (77 rows): see full list above

### Moderate issues (data quality degraded)

7. **Meta-description solutions** (728 rows): No mathematical work shown; solution field contains only a description like "General perturbation applied" or "Changed 12 to 13."
8. **No-op changes** (44 rows): Solution says "Changed X to X" with the same number. Indices: 3058-3498 (see full list above)
9. **Replaced with different problem** (~838 rows): Not a perturbation at all; entirely different problem structure and topic. Concentrated in Counting & Probability.
10. **Unperturbed rows** (1332 rows): 26.6% of the dataset has `problem == original_problem`.

### Estimated usable rows

Filtering out all critical issues, meta-descriptions, and FAILED rows: approximately **3287 rows** appear to have actual perturbations with non-meta solutions. However, of these:
- 835 are replaced problems (not perturbations)
- 612 have the same answer as the original (some legitimate, some errors)
- 1628 have solutions under 50 characters (answer-only, no verification possible)

**Conservatively, ~2100-2500 rows are truly perturbed with plausible answers**, but even among these, answer correctness cannot be fully verified without the step-by-step solutions that are largely missing.

---

## Recommendations

1. **Regenerate solutions** for all 5000 problems using a capable model (GPT-4, Claude, etc.), ensuring solutions reference the correct (perturbed) numerical values and include `\boxed{}` answers.
2. **Fix the 838 replacement problems** -- either generate actual perturbations of the originals, or clearly mark these as replacements, not perturbations.
3. **Fix the 1332 unperturbed rows** -- either perturb them or remove them from the dataset.
4. **Fix the 41 rows** where problem is unchanged but answer was incorrectly modified.
5. **Fix truncated answers** (114 rows) by completing the LaTeX expressions.
6. **Standardize type field** formatting (86 rows with lowercase types).
7. **Verify answer correctness** for all rows, especially the 914 where the problem changed but the answer stayed the same.
8. **Remove or fix FAILED rows** (25 rows).
9. **For our evaluation pipeline**: filter the dataset before use. At minimum, exclude rows where `solution` contains "FAILED TO PERTURB", meta-descriptions, empty answers, and truncated answers. Consider only using rows where `problem != original_problem` and character similarity > 0.7.
