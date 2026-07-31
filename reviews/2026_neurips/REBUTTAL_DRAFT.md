# Rebuttal draft — NeurIPS 2026 submission 32216

Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) · **1wx9 4** · **aPBL 3**. The AC named
8RFz's loss-vs-correctness objection the pivotal critique.

**Written for a 10-minute read.** Four sections, ~2.5–3k characters each. Every number is 0-shot
greedy with boxed-required scoring. Supporting detail — provenance, caveats, validation — lives in
[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md) and belongs in the paper, not here.

**Editing rule:** if a sentence explains our process rather than answering a question, cut it.

---

## General response

We thank the reviewers and the AC. The AC identified 8RFz's loss-vs-correctness objection as the
critique that questions whether our evidence supports our claims. We answer it with new
measurements, and we correct one thing in our own submission.

**1. Findings 4 and 5, now measured in accuracy.** We evaluated all 137 overtraining and all 39 SFT
checkpoints with Math Verify. Accuracy tracks loss, so the scenario 8RFz raises — accuracy
persisting while loss rises — does not occur. The measurement also forces us to weaken Finding 4:
dilution is **threshold-dependent**. At 93M over ot 1→16, the contamination advantage retained is
**0.0188 at R = 100 but 0.9966 at R = 1000**. Overtraining is not a mitigation in the heavy-leakage
regime, and we no longer claim it is. For Finding 5, SFT takes mean Math Verify from **72.95% to
2.80%** across the 14 conditions with something to lose.

**2. A correction to Table 1.** 8RFz's Q4 caused us to audit it. The printed values do not
reproduce; they predate our current rephrased/perturbed datasets. We also found Table 1 was
evaluated 4-shot while Figure 1 is 0-shot — which matters enormously here, because the few-shot
prefix displaces the memorized context (the same checkpoint scores **1.0000 at 0-shot and 0.0052 at
4-shot**). We standardised on 0-shot and re-measured across 39 checkpoints, averaging over the 14
with R ≥ 100:

| Condition | Math Verify | Advantage removed |
|---|---|---|
| Original | 72.18% | — |
| Rephrased | 2.78% | 96.1% |
| Perturbed | 1.91% | 97.4% |

**3. A new experiment, and the result we would most like read.** 1wx9 asked for contamination that
is paraphrased rather than exact. We ran it, and it produced a clean dissociation between loss and
correctness — the exact distinction 8RFz's objection rests on. A model pretrained on **rephrased
problems with verbatim solutions** has cross-entropy 78% of the way from clean to
fully-contaminated, yet scores **1.52%** where the exact-replica model scores **7.22%**, and never
reproduces the gold solution (0 of 5,000). A positive control shows it memorizes just as strongly
on its own training items (7.56%, regurgitating verbatim 5.34% there).

So **memorization is of the solution text; retrieval is keyed on the problem text.** The model holds
the answer and cannot reach it. This unifies Finding 2 with the new experiment, and it inverts the
detection concern: perplexity flags these models loudly while their benchmark scores stay near
clean.

**4. Also addressed.** Bootstrap confidence intervals on every Math Verify number (median
half-width 0.33 pp; test-set sampling error, not seed variance, and we say so). Five missing
references added and our findings situated against them. New appendices giving SFT hyperparameters
and the construction and validation of the modified test sets.

---

## Reviewer 8RFz

**W1/Q1 — do Findings 4 and 5 hold for Math Verify score?**

Yes, and we now measure it rather than inferring it from Figure 11. All 137 overtraining and 39 SFT
checkpoints, in accuracy space. You raised the sharper possibility that accuracy might persist while
loss rises; we looked for it and it does not occur.

The measurement does force us to weaken Finding 4. Dilution is threshold-dependent: at 93M over
ot 1→16, the advantage retained is **0.0188 at R = 100** and **0.9966 at R = 1000**. The mechanism
is dilution of the contaminated token fraction, which stops working once that fraction stays high.
Our unqualified phrasing was misleading and we have replaced it. For Finding 5, SFT takes mean Math
Verify from **72.95% to 2.80%** (14 conditions, median retained fraction 0.022, range 0.001–0.302).

**We also found a regime where your objection is exactly right.** Pretraining on rephrased problems
paired with verbatim solutions, then evaluating on the originals:

| R | Loss, exact | Loss, rephrased | Acc, exact | Acc, rephrased |
|---|---|---|---|---|
| 316 | 0.5243 | 1.9573 | 7.22% | **1.52%** |

(Uncontaminated: loss 7.1437, accuracy 0.00%.) By any loss-based measure that model is heavily
contaminated; it nonetheless scores 1.52% and reproduces the gold solution verbatim 0 times in
5,000. A positive control on its *own* training items gives 7.56% and a 5.34% verbatim rate, so the
failure is retrieval, not learning. **Memorization is of the solution text; retrieval is keyed on
the problem text.** Rephrasing at training time stores the solution without the key; rephrasing at
evaluation time (Table 1) withholds the key from a model that has one.

One consequence runs opposite to your concern: rather than contamination evading perplexity, we see
perplexity flagging models whose benchmark scores are barely inflated — a false-positive mode for
loss-based detection.

**W2/Q2 — temperature.** You are right that Finding 6 does not separate the two explanations. The
control is the contamination *advantage* at matched temperature, so uniform degradation cancels:

| τ | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Advantage retained | 100% | 98% | 90% | 72% | 39% | **25%** | 0.4% |

The effect is contamination-specific: at τ = 1.0, three quarters of the advantage is already gone.
Above τ ≈ 1.3 the explanations are inseparable, so we restrict the claim to τ ≤ 1.

**W3/Q3 — related work, and the conflict with prior rephrasing results.** We accept this and remove
the "first targeted examination" framing. We add Palavalli et al. (2024), Mehrbakhsh et al. (2024),
Dekoninck et al. (2024a, 2024b) and Godey et al. (2025), and we now state that Finding 1
**replicates** the repeat-count effect rather than presenting it as new.

On the apparent conflict with Mehrbakhsh et al. and Dekoninck et al.: we think it is a regime
boundary, and we can now support that. Those studies contaminate models **already capable** of the
task, which can bridge a surface-form change. Ours have no such capability: an uncontaminated 344M
model produces 0 correct answers in 5,000,000 samples, and under a deliberately over-generous scorer
that ignores output format entirely, uncontaminated models score at or below that scorer's own
false-positive rate. With no latent capability, all contaminated performance is verbatim
memorization, which cannot survive rephrasing. This predicts that paraphrase transfer is a function
of capability and should switch on between our scale and theirs.

One respectful correction: **Jiang et al. (2024) is already cited**, in the introduction and at
paragraph length in the appendix. We mention it only because it bears on the Originality assessment.

**Q4 — how are the values in Table 1 calculated?** Directly: **they do not reproduce.** They were
produced on separate infrastructure and predate our current datasets. We re-measured across 39
checkpoints at 0-shot (table in the general response). Two disclosures: the perturbed figure
excludes 582 problems (11.64%) whose perturbation leaves the answer unchanged, since those score a
memorizing model correct by construction; and we no longer say performance "collapses to baseline",
since the floor is 0.00% and the modified conditions sit 1.9–2.8 pp above it.

---

## Reviewer 1wx9

**W1/Q1 — paraphrased rather than exact contamination.** We agree this was the most important
missing condition, and we ran it: new 34M models pretrained with the *injected* text differing from
the text loss is measured on, at R = 32, 100, 316.

Setting it up surfaced something that improved the experiment. Our rephrased MATH set rephrases the
problem but keeps the solution — 4,991 of 5,000 solutions are byte-identical — so injecting it is
*solution-verbatim* leakage. Our perturbed set differs on both sides. Rather than one condition we
report an ablation over **which component of a leaked document carries the effect**:

| Arm | Problem | Solution | Loss @ R=316 | Accuracy @ R=316 |
|---|---|---|---|---|
| Uncontaminated | — | — | 7.1437 | 0.00% |
| Exact replicas | same | same | 0.5243 | **7.22%** |
| Rephrased | differs | **same** | 1.9573 | 1.52% |
| Perturbed | differs | differs | 3.3705 | 1.60% |

Two findings. **Only exact-replica contamination produces a dose-response in accuracy** — it climbs
13× from R = 32 to R = 316, while both arms whose problem text differs plateau at ~1.5% and stay
there. But **loss calls the perturbed model 57–88% as contaminated as the exact one.** The two
metrics disagree about the same models, and accuracy is the one tracking what a benchmark reports.
This is the clearest argument we can give for why contamination work should report accuracy and not
loss alone.

Two caveats we state rather than bury: our perturbed set is 21.7% smaller per copy, so at fixed R it
delivers a smaller dose than its label implies — the bias runs against us. And both modified corpora
are still MATH-domain text, so part of any reduction is domain adaptation; separating that cleanly
needs a fourth arm contaminated with *disjoint* mathematics, which we have not run.

**W2 — "if we have rephrased contamination in training, evaluation looks uncontaminated. Which is
quite surprising."** Thank you — this identified a real ambiguity. Table 1 does **not** test that
direction; it tests exact contamination in training with *modified* evaluation. The direction you
describe is the experiment above, and it was untested in the submission. We have clarified the
caption.

Your accompanying hypothesis — that scale may be insufficient to see generalization from
contaminated data — is, we believe, correct. An uncontaminated 344M model produces 0 correct answers
in 5,000,000 samples, and shows no capability even under a scorer that ignores output format
entirely. There is no latent ability for contamination to combine with, so anything that breaks
verbatim surface-form match removes the whole effect.

**W3 — scale, single benchmark.** Conceded; raised by all three reviewers. We keep the
scale-for-control argument, state the limitation more prominently, and commit to a second benchmark
and model family for the camera-ready.

**W4 — exact leakage makes loss results less surprising.** Agreed, and we say so. We would note the
irreducible-error result is not in that category — the surprise is not that loss drops but that a
*single* replica pushes measured loss below the extrapolated uncontaminated asymptote.

---

## Reviewer aPBL

**W1/W2 — small models, single dataset.** Conceded without argument. We keep the deliberate
scale-for-control tradeoff, state the limitation more prominently, and commit to a second benchmark
and model family. One observation from this rebuttal: the *reason* our models show no paraphrase
transfer is that they have zero baseline capability, which makes scale a **substantive boundary
condition** on Finding 2 rather than a caveat.

**W3 — single seed, no error bars.** Added. 95% percentile bootstrap intervals over the test
problems (10,000 resamples); median half-width **0.33 pp** against effects spanning ~1% to 100%. We
state explicitly that this is test-set sampling error and **not** seed-to-seed variance, and commit
to multiple seeds at the R ≈ 10–100 transition for the camera-ready.

**W4 — missing SFT hyperparameters.** You are right. We add an appendix with the full configuration
(AdamW, lr 1×10⁻⁴, cosine schedule, 0.2 warmup ratio, 1 epoch, effective batch 64, max length 2048,
bfloat16, trained on the MATH *train* split). All 39 runs share these; only the initial checkpoint
varies.

**Q1 — multiple replicates seem contrived.** Quantifying it helped. As a fraction of tokens actually
trained on, at **R = 1** the test set is **0.03%–0.30%** depending on model size — at or below
published real-world leakage estimates, and we measure effects there. At the top of the ladder it
reaches 74%, which is deliberately extreme. The ladder is a **dose-response curve from
below-realistic to saturating**, and we now describe it that way.

**Q2 — how were the modified sets validated, and are difficulty and length matched?** We
under-documented this; a new appendix covers it. Difficulty is matched exactly by construction —
both sets are index-aligned rewrites, so the Level 1–5 distribution is identical across all three.
Problem lengths are close (median 47 → 45, 46 tokens); perturbed solutions are shorter (mean 162.8
vs 215.3), which we report because it means a perturbed replica delivers ~22% fewer tokens at the
same nominal dose. Validation combined automated checks with manual inspection; an earlier pair of
candidate datasets failed the audit and was regenerated.

That appendix also reports a problem your question was aimed at: **11.64% of perturbed problems (582
of 5,000) have a perturbation that leaves the ground-truth answer unchanged**, scoring a memorizing
model correct by construction. We exclude them and report that including them inflates the perturbed
score from 1.91% to 4.84%.

**Q3 — does the irreducible error come from fitting an asymptotic scaling law?** The right question.
The claim's structure makes it more robust than it looks: the contaminated losses are **measured**,
not fitted; only the uncontaminated asymptote E(0) is extrapolated. So the claim needs a
*conservative lower bound* on E(0) that still exceeds the measured contaminated losses — not a
correct functional form. Bootstrap refitting gives E(0) = 3.5942, 95% interval [3.5359, 3.6639], and
**33 of 35 contaminated runs (94.3%) fall below the lower end of that interval**; the two exceptions
are the smallest model at R = 1 and R = 3. We report the fit's limitations in the appendix (the
intervals are optimistically narrow; the R = 32 and R = 1000 fits are flagged unreliable).

**Q4 — cross-domain contamination.** The natural extension, and we now specify the design: pretrain
with MATH contamination as here, then evaluate on held-out tasks sharing competence but not items
(GSM8K, MMLU mathematics, a code benchmark requiring arithmetic), measuring whether the advantage
transfers across *domain* when it does not transfer across *surface form*. Our result predicts it
will not at this scale, which makes it a clean test of the capability-boundary hypothesis once run
larger.

---

## Posting checklist

- [ ] Confirm the OpenReview per-comment character limit. Sections are ~2.6–3.4k characters.
- [ ] Do not use the "~60× SFT collapse" figure — an artifact of comparing 0-shot pretrained against
      4-shot SFT.
- [x] 4-shot is not used as evidence anywhere. It appears only in the Table 1 disclosure, which is
      required because the submitted numbers change.
- [x] Five citations added **and `\citep`'d**; bibliography verified (128 entries, 0 undefined).
- [x] SFT and modified-test-set appendices written; manuscript compiles (47 pages).

### Numbers that changed — do not paste from an older draft

| Quantity | Superseded | **Use** |
|---|---|---|
| Table 1 (R≥100) | 70.19 → 2.74 / 1.89% | **72.18 → 2.78 / 1.91%** |
| Uncontaminated floor | ~1% | **0.00%** (0.00–0.06% at 344M, all hits inspected & spurious) |
| SFT | 72.31 → 3.00% | **72.95 → 2.80%** (14 conditions) |
| Finding #4 retention | 0.019 / 0.995 | **0.0188 / 0.9966** |
| Temperature at τ=1.0 | 9.6% (retracted) | **25%** |
| Answer-overlap inflation | 4.78% | **4.84%** |
| Contaminated fraction at R=1 | 0.02–0.21% | **0.03–0.30%** (of tokens actually trained) |
| 344M R=0 0-shot | "all ten runs failed" | **recovered**: sweeps `woygzpil`/`oj6o8idv` |

`notebooks/11_*/results/protocol_sensitivity_rescored.csv` (`strict_score`) is authoritative.
