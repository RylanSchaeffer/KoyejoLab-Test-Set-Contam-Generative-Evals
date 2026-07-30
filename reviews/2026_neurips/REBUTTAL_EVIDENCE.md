# Rebuttal Evidence Map

Every number you need, mapped to the reviewer question it answers, with the artifact that
produced it. Written to be worked from directly.

**Read [`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md) before quoting any Math Verify number.**
All figures below are **0-shot greedy** unless stated otherwise. 0-shot is the protocol behind
the manuscript's Finding #1 figure and every teacher-forced notebook; the 4-shot numbers in
notebooks 13 and 15 measure something different and are not interchangeable.

---

## AC's five weakness bullets

| AC bullet | Status | Evidence |
|---|---|---|
| Loss vs correctness (8RFz's pivotal critique) | **Answered** | Finding #4 measured in accuracy space, below |
| MATH-specific | Conceded; camera-ready commitment | — |
| No uncertainty quantification | **Answered** | Bootstrap CIs, below |
| SFT result absent from paper | **Answered** | Finding #5 at 0-shot, below |
| External validity / scale | Conceded; pass@k reframes it | pass@k, below |

---

## Finding #4 — overtraining and contamination (8RFz's pivotal critique)

**137/137 overtrained checkpoints evaluated.** Accuracy tracks loss, so the objection that
"loss on exact solution text is not correctness" is answered on its own terms. The
stealth-contamination alternative — accuracy persisting while loss rises — does not occur.

The result is sharper than the paper's current claim: **dilution is threshold-dependent.**

| | Retained after overtraining |
|---|---|
| 93M, R=100, ot 1→16 | **0.0188** |
| 93M, R=1000, ot 1→16 | **0.9966** |

Same model, same multiplier range, ~53x difference. Below the memorization threshold
overtraining suppresses contamination by more than an order of magnitude; above it, 16x more
training does essentially nothing. Mechanism is dilution of the contaminated *token fraction*,
which is why it stops working once that fraction stays high.

*Do not* state Finding #4 as "overtraining dilutes contamination" unqualified — it reads as a
mitigation, and it is false in the regime that matters most (heavy leakage).

- `notebooks/17_math_qwen3_pt_overtrained_math_verify/results/OVERTRAINING_MATH_VERIFY.md`
- `reviews/2026_neurips/data/CONTAMINATED_TOKEN_FRACTION.md`

## Finding #2 / Table 1 — memorization, not generalization (8RFz Q4, aPBL Q2)

Measured at 0-shot, where the gains exist. Averaged over 13 contaminated checkpoints (R >= 100):

| Condition | Score | Advantage removed | Residual above floor |
|---|---|---|---|
| Original | 72.18% | — | — |
| Rephrased | 2.78% | 96.1% | +2.78 pp |
| Perturbed | 1.91% | 97.4% | +1.91 pp |

⚠️ Updated 2026-07-29. The earlier version of this table paired a *leniently* scored Original
column against strictly scored Rephrased/Perturbed columns. Both are now boxed-required scored.
The uncontaminated floor is **exactly 0.00%** at every size, so the residual is quoted in
percentage points; the old "2-3x the floor" phrasing divided by a floor that was entirely
lenient-scorer false positives.

Two things to carry into the text:

1. **Perturbed excludes 582 problems (11.64%) whose perturbation leaves the ground-truth answer
   unchanged.** Those score a memorizing model correct by construction. Including them inflates
   Perturbed to 4.78% and inverts the expected ordering. A reviewer checking the dataset will
   find this, so state it.
2. **Do not say performance "collapses to baseline."** Under matched strict scoring the
   uncontaminated floor is exactly 0.00%, and rephrased/perturbed sit 1.9-2.8 percentage points
   above it. "Removes the large majority of the contamination advantage, leaving a small
   residual" is what the data support.

Table 1's printed values (0.00-0.04%) do not reproduce — they predate the current datasets and
correspond to no run in this W&B account. Replace them with these.

- `notebooks/18_math_qwen3_pt_rephrase_perturb_zeroshot/results/TABLE1_ZEROSHOT.md`
- `reviews/2026_neurips/data/PERTURBED_ANSWER_OVERLAP.md`

## Finding #5 — SFT (8RFz Q1)

0-shot SFT evaluation in progress at time of writing; see
`notebooks/13_*/results/FORMAT_SANITY_CHECK.md` for the matched-protocol 4-shot comparison.

**The "~60x collapse" in `REBUTTAL_PLAN.md` P0.1 is an artifact** — it compares 0-shot
pretrained (~100%) against 4-shot SFT (~1-2%). Matched at 4-shot the figures are 0.40% and
0.20%. Do not use it.

## Temperature (8RFz W2/Q2)

The clean control is the contamination advantage at **matched** temperature,
`score(R) - score(R=0)` with both terms at the same tau, so uniform degradation cancels.

| tau | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Fraction of greedy advantage | 100% | 92% | 77% | 55% | 20% | **9.6%** | 0.02% |

⚠️ Rescored 2026-07-30 with boxed-required scoring. The previous row (100/90/72/39/25/0.4) came
from leniently scored runs; the matched-temperature difference does NOT cancel that artifact,
because the uncontaminated arm is almost all false positives while the contaminated arm is mostly
real. Source: `notebooks/11_*/results/TEMPERATURE_RESPONSE_RESCORED.md`.

tau = 1.0 is the model's own distribution, not a hot setting, and >90% of the advantage is
already gone. Restrict the claim to tau <= 1 and concede that everything degrades above it.

- `notebooks/11_math_qwen3_pt_math_verify/results/TEMPERATURE_RESPONSE.md`

## Capability baseline (aPBL "small models", supports the P3.1 reconciliation)

**0 correct out of 5,000,000 samples** — uncontaminated 344M, 5,000 problems x 1,000 samples,
tau = 1.0. Not one sample contains even a well-formed `\boxed{}`. Every point of contaminated
performance is therefore unambiguously memorization, and the models provably have no latent
capability to bridge surface-form changes.

- `results/pass_at_k/mem_Qwen3-344M_..._ot_1/temp=1.0/summary.md`

## Finding #3 — irreducible error (aPBL Q3)

**The claim survives its own uncertainty.** State the logical structure explicitly: contaminated
losses are *measured*, only the uncontaminated asymptote E(0) is *extrapolated*, so the claim
needs only a conservative lower bound on E(0) to exceed the measured contaminated losses — not
an exactly correct functional form.

- E(0) point estimate **3.5942**, 95% interval **[3.5359, 3.6639]** (reproduces the
  manuscript's 3.594 exactly, which validates the refit).
- **33 of 35 contaminated runs (94.3%)** have measured loss below the *lower* end of that
  interval.

Three things not to overstate:

1. The intervals are **optimistically narrow** — each resample is refit by local optimization
   seeded at the full-data solution, since the repo's 5,760-point grid search cannot be
   bootstrapped in reasonable time. Quote the conclusion, not the interval widths.
2. **R = 32 returns E = 0.0000, which is not a measurement** — the optimizer drove `e_0` toward
   negative infinity, meaning the data admit no identifiable asymptote. Flagged unreliable.
3. R = 1000 rests on 3 points with 79/300 resamples converging. Also flagged unreliable.

- `reviews/2026_neurips/data/IRREDUCIBLE_ERROR_ROBUSTNESS.md`

## Uncertainty (aPBL W3, AC bullet 3)

95% percentile bootstrap over the 5,000 test problems, 10,000 resamples. **Median CI half-width
0.33 percentage points** against effects spanning ~1% to 100%.

State plainly that this is test-set sampling error, **not** multi-seed variance, and commit to
seeds for camera-ready. Presenting it as covering seed variance would be worse than reporting
no intervals.

- `notebooks/11_math_qwen3_pt_math_verify/results/BOOTSTRAP_CIS.md`

## "Contrived" contamination levels (aPBL Q1)

At R = 1 the test set is 0.02%-0.21% of the training budget depending on model size — at or
below published real-world leakage estimates, and effects are measured there. At the top of the
ladder it reaches 67%-92%, which is deliberately extreme and should be described that way. The
honest framing is a dose-response curve spanning below-realistic to saturating.

- `reviews/2026_neurips/data/CONTAMINATED_TOKEN_FRACTION.md`

## Upside item — temperature response as a detector (not requested)

AUC 0.99 separating contaminated from clean checkpoints using only a model compared to itself
at two temperatures — no corpus, no reference model.

**Pitch as proof of concept only.** In this grid every clean checkpoint scores near zero, so any
feature keying on high greedy accuracy separates perfectly; the table cannot show the drop
features beat raw accuracy. The regime where they should win — a capable clean model — contains
no checkpoints here. Overclaiming would repeat the framing error that earned Originality = 2.

- `reviews/2026_neurips/data/CONTAMINATION_DETECTOR.md`

---

## Things that would embarrass you if a reviewer found them first

1. **Fig. 1 is 0-shot; Table 1 and the SFT figures are 4-shot.** Same checkpoint scores 1.0000
   vs 0.0052 across that difference.
2. **Notebook 16's "14/17 conditions, up to -4.72 nats"** predates the token-weighting fix.
   Corrected: **17/17 conditions, max -2.18 nats.** Stronger on consistency, weaker on magnitude.
3. **11.64% of perturbed problems keep the original answer.**
4. **Table 1's printed values are not reproducible** from any run in this account.
