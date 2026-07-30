# Rebuttal draft — NeurIPS 2026 submission 32216

Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) · **1wx9 4** · **aPBL 3**. The AC named
8RFz's loss-vs-correctness objection the pivotal critique.

**Every number is 0-shot greedy with boxed-required scoring unless labelled otherwise.** Read
[`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md) before editing any figure; sources in
[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md).

**Rewritten 2026-07-30** for length (each section now ≈5–6k characters, against a likely 6,000-char
OpenReview cap) and to correct the format-conflation argument — see the posting checklist.

---

## General response (post once, to all reviewers and the AC)

We thank all three reviewers and the AC. The AC singled out 8RFz's loss-vs-correctness objection as
the one critique questioning whether our evidence supports our claims. That was the right thing to
elevate, and we answer it with new measurements rather than argument: **we evaluated all 137
overtraining checkpoints and all 39 SFT checkpoints in Math Verify (accuracy) space.** In doing so
we found and corrected two problems in our own submission.

**1. Findings 4 and 5 now rest on accuracy, not loss.** Accuracy tracks loss, so the
stealth-contamination scenario 8RFz raises — accuracy persisting while loss rises — does **not**
occur here. The measurement also obliges us to weaken Finding 4: dilution is
**threshold-dependent**. For 93M over ot 1→16, the contamination advantage retained is **0.0188 at
R = 100 but 0.9966 at R = 1000** — same model, same range, ~53× difference. Below the memorization
threshold overtraining suppresses contamination by more than an order of magnitude; above it, 16×
more fresh data does essentially nothing. "Overtraining dilutes contamination" reads as a
mitigation and is false exactly in the heavy-leakage regime that matters most. The revision states
the threshold behaviour instead.

**2. A protocol inconsistency in our own paper.** Our generative evaluations were not run under one
protocol. Figure 1 and every teacher-forced result are **0-shot**; Table 1 and the SFT figures are
**4-shot**. At our scale this dominates everything: the same checkpoint (344M, R = 3162, greedy,
identical scoring) scores **1.0000 at 0-shot and 0.0052 at 4-shot**.

The cause is mechanistic. At 0-shot the prompt reproduces the opening of the memorized training
document and the model emits the stored solution. At 4-shot, four unrelated worked examples precede
the problem, the prompt matches no memorized context, and the model produces fluent unrelated text.
Median prompts are ~687 tokens against a 2,048-token pretraining length, so this is not overflow.
We had already recognised this for teacher forcing — our code notes that "a 4-shot prefix would
change the conditioning context and dilute the memorization signal" — and simply failed to carry
the reasoning to generation.

**We have standardised on 0-shot and re-measured everything affected.** This is why the values in
Table 1 and the SFT figures change substantially in the revision, and we flag it prominently rather
than let the new numbers appear without explanation. The 4-shot measurements are not reported in the
revision: they conflate the memorization signal with the conditioning context, so they answer a
different question than the one the paper asks.

4-shot was originally adopted on the theory that uncontaminated models never see `\boxed{}`, so
requiring it makes them unable to score above zero regardless of mathematical ability, conflating
output format with reasoning. **We think that concern is legitimate**, and we now address it
directly — but the right test is not to teach the format, it is to **remove the requirement
entirely**.

We therefore rescored the uncontaminated models with a deliberately over-generous scorer that
ignores formatting completely, crediting the gold answer wherever it appears in the output. Nothing
appears underneath. Across the uncontaminated 0-shot checkpoints that scorer credits **178 of
20,004 responses (0.89%)** — and we inspected **every one**: **not a single one contains a
`\boxed{}` expression, and all are false positives.** They are driven by the base rate of small
numbers occurring in text (75.8% have a single-digit gold answer; gold = 1 alone accounts for
44.4%), and the responses are degenerate repetition loops such as `" 1. 1. 1. 1. 1. 1. ..."` on
problems whose answer is 1. The same signature holds in sampled output: of ~14,300 credited samples
out of ~1,038,000, **zero** contained a `\boxed{}`, and answers 0–5 accounted for 77.6%.

We also validated that this scorer is genuinely generous rather than merely insensitive, since
otherwise a null result would be uninformative. It credits **229 of 229** responses in which a
contaminated model reproduces the gold solution verbatim — indisputably correct output, perfect
recall; it credits every numeric answer we plant in it across seven surface forms; and it is a
superset of our strict scorer (1 exception in 20,004). Its one blind spot is a bare *symbolic*
answer written without math delimiters, which we closed with a scorer-independent check: on the
1,153 problems with symbolic answers, the gold answer appears anywhere in an uncontaminated
model's output at most **0.78%** of the time.

So the format requirement is not hiding any capability — there is none to hide, and 0-shot does not
disadvantage the uncontaminated baseline. This is a stronger answer to the concern than the 4-shot
prefix was: measured, that prefix leaves the uncontaminated `\boxed{}` rate at exactly 0.0000 at
every model size, so it never removed the format barrier it was adopted to remove.

**One consequence for Figure 1.** Our 0-shot and 4-shot sweeps also straddled a commit that
tightened scoring (bare-number extraction → required `\boxed{}`), so we rescored every run with the
same strict scorer. This moves the uncontaminated and low-dose points from 0.38–1.26% to **0.00%**
(0.06% at 344M, three responses, all inspected and spurious), leaving high-contamination saturation
untouched (<0.2 pp). The corrected figure states our
claim more cleanly: the baseline is not a small positive number that contamination lifts, it is
zero, and *every* point of measured performance is contamination-derived. Supporting this, at 0-shot
the `\boxed{}` rate *rises with dose* (153M: 0.000 → 0.009 → 0.047 → 0.72 → 0.98 → 1.000 as R goes
0 → 316): contamination supplies format and answer together.

**3. Uncertainty** (aPBL W3, AC bullet 3). We add 95% percentile bootstrap intervals over the test
problems (10,000 resamples); median half-width **0.33 pp** against effects spanning ~1% to 100%. We
state plainly that this is test-set sampling error, **not** multi-seed variance, and commit to seeds
at pivotal configurations for camera-ready.

**4. Related work** (8RFz W3, AC bullet 5). We accept this. We remove the "first targeted
examination" framing, add five references, and situate Findings 1 and 2 against the work they
respectively replicate and appear to contradict. Details to 8RFz.

**5. Realistic leakage** (1wx9 W1/Q1, aPBL Q1, AC bullet 1). We ran it — a three-arm ablation over
*which component of a leaked document* carries the effect. Details to 1wx9.

**6. A result we did not go looking for.** Pursuing (5) produced a clean dissociation between loss
and correctness — exactly the distinction 8RFz's W1 rests on. A model contaminated with rephrased
problems and verbatim solutions has cross-entropy 78% of the way from clean to fully-contaminated,
yet scores 1.52% where the exact-replica model scores 7.22%, and reproduces the gold solution
verbatim **0 times in 5,000**. A positive control shows it memorizes just as strongly on its own
training items (7.56%), regurgitating verbatim 5.34% there. So **memorization is of the solution
text; retrieval is keyed on the problem text.** The model holds the answer and cannot reach it.
This unifies Finding 2 with the ablation, concedes 8RFz's premise by demonstrating it, and inverts
their detection worry: perplexity flags these models loudly while their benchmark scores stay near
clean — a false-positive mode for loss-based detection.

Every criticism we could address with data, we did.

---

## Response to Reviewer 8RFz

Thank you — your W1 identified the weakest link, and it changed one of our conclusions.

**W1 / Q1 — do Findings 4 and 5 hold for Math Verify score?**

Yes, and we now measure it rather than inferring it from Figure 11's correlation: **all 137
overtraining and all 39 SFT checkpoints**, in accuracy space. You raised the sharper possibility
that accuracy might persist while loss rises, making contamination harder to detect while still
inflating scores. We looked for exactly that and **it does not occur**.

But the measurement forces us to weaken Finding 4. Dilution is **threshold-dependent**: for 93M
over ot 1→16, the advantage retained is **0.0188 at R = 100** and **0.9966 at R = 1000** — a ~53×
difference over the same range. The mechanism is dilution of the contaminated *token fraction*,
which is why it stops working once that fraction stays high. Our original unqualified phrasing is
misleading, and we replace it with the threshold formulation. We think this is the more useful
finding: overtraining is not a mitigation for the leakage regimes practitioners should worry about.

For Finding 5, at matched 0-shot protocol and scoring, SFT takes mean Math Verify from **72.95% to
2.80%** across the 14 conditions with something to lose (≥5% before SFT), median retained fraction
**0.022** (range 0.001–0.302). We quote the range rather than a single multiplier.

**We also found a regime where your objection is exactly right, and it is the most interesting
result in this response.** Pursuing 1wx9's request, we pretrained models on rephrased problems
paired with verbatim original solutions, then evaluated 0-shot on the original problems:

| R | Loss, exact | Loss, rephrased | Acc, exact | Acc, rephrased | Verbatim rate |
|---|---|---|---|---|---|
| 100 | 1.4526 | 2.0077 | 1.70% | 1.58% | 0.000 |
| 316 | 0.5243 | 1.9573 | 7.22% | **1.52%** | **0.000** |

(Uncontaminated: loss 7.1437, accuracy 0.00%.) At R = 316 the rephrased model's cross-entropy is
78% of the way from baseline to the exact-replica model's — by any loss measure it is heavily
contaminated. Yet it scores 1.52% and reproduces the gold solution verbatim **0 times in 5,000**.
Loss and correctness genuinely come apart, precisely as you argued they could.

The mechanism: **memorization is of the solution text while retrieval is keyed on the problem
text.** Rephrasing at training time stores the solution without the key; rephrasing at evaluation
time (our Table 1) withholds the key from a model that has one. Both collapse generation for the
same reason, which unifies Finding 2 with this result.

The positive control (general response, item 6) rules out the alternative that nothing was learned:
measured on the items each model was *actually trained on*, the rephrased arm matches the exact arm
at every dose (7.56% vs 7.22% at R = 316) and regurgitates the gold solution verbatim 5.34% of the
time there, against 0.000% on the original problems, with byte-identical solutions in both. It is
fully capable of regurgitation; the original problem does not trigger it. The failure is retrieval,
not learning. We scope this honestly: the gap is high-dose only — at R = 100 the conditions are
indistinguishable (1.56% vs 1.58%) and at R = 32 both sit near the floor.

One consequence runs opposite to your concern: accuracy persisting while loss rises would evade
perplexity-based detection, but we observe the *other* asymmetry — **perplexity would flag these
models loudly while their benchmark scores are barely inflated.** Since realistic leakage rarely
reproduces problems verbatim, that may be the more common case.

None of this rescues Findings 4–5 by itself; those stand on the accuracy measurements above. But we
accept your framing rather than merely answering it, and report both metrics everywhere.

**W2 / Q2 — the temperature confound.**

You are right that Finding 6 does not separate "temperature reduces contamination effects" from
"temperature degrades generation." The clean control is the contamination *advantage* at **matched**
temperature — score(R) − score(R=0) with both at the same τ — so uniform degradation cancels:

| τ | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Fraction of greedy advantage retained | 100% | 98% | 90% | 72% | 39% | **25%** | 0.4% |

The effect is contamination-specific: at τ = 1.0 — the model's own distribution, not a hot setting —
roughly three quarters of the advantage is gone with general degradation controlled for. Above
τ ≈ 1.3 the explanations are no longer separable; we restrict the claim to τ ≤ 1 and say so.

**W3 / Q3 — related work and the conflict with prior rephrasing results.**

We largely accept this and remove the "first targeted examination" claim. We add **Palavalli et al.
(2024)**, **Mehrbakhsh et al. (2024)**, **Dekoninck et al. (2024a, evading detection)**, **Dekoninck
et al. (2024b, ConStat)**, and **Godey et al. (2025, Gaperon)** — the last closely related, since its
"late deliberate contamination" is the large-scale analogue of our overtraining and SFT conditions.

One respectful correction: **Jiang et al. (2024) is already cited** — in the introduction twice and
at paragraph length in the appendix related work, where we describe their text-only versus
ground-truth conditions. We mention it only because it bears on the Originality assessment; the
substantive point about *situating* our findings stands and we have acted on it.

- **Finding 1 replicates** the repeat-count effect (Jiang et al.; Dekoninck et al.), and we say so.
- **Finding 2 appears to conflict** with Mehrbakhsh et al. and Dekoninck et al., who find rephrased
  contamination *does* transfer. We think this is a regime boundary, and can now support that with
  a measurement. Prior studies contaminate models **already capable** of the task, which can bridge
  a surface-form change. Our models have **no competence to bridge with**, established on the
  uncontaminated 344M checkpoint two ways. The first does not depend on output format at all: under
  the lenient scorer, which ignores formatting entirely, these models score **0.38–1.26%**, at or
  below that scorer's own measured **~1.38% false-positive rate**. The second is sampling at scale:
  **0 correct in 5,000,000 samples** (5,000 problems × 1,000 at τ = 1.0), and **pass@25 = 0 on all
  2,500 problems** in a separate 0-shot run. We note that the sampling figures use boxed-required
  scoring and none of those samples contained a well-formed `\boxed{}`, so it is the format-free
  bound that carries the argument; the sampling result establishes that no amount of resampling
  uncovers latent ability.

  With zero latent capability, all contaminated performance is verbatim memorization, which cannot
  transfer across rephrasing. This makes the results complementary and yields a prediction: transfer
  across paraphrase is a function of capability and should switch on between our scale and theirs.

**Q4 — how are the values in Table 1 calculated?**

This question caused us to audit Table 1, and we owe you a direct answer: **the printed values
(0.00%–0.04%) do not reproduce.** They were produced on separate infrastructure by a co-author and
predate our current rephrased/perturbed datasets; they correspond to no run we can point at. We
should have verified them before submission. Re-measured across 39 checkpoints × 2 modified datasets
at 0-shot, averaged over the 14 contaminated checkpoints with R ≥ 100:

| Condition | Math Verify | Advantage removed | Residual above floor |
|---|---|---|---|
| Original | 72.18% | — | — |
| Rephrased | 2.78% | 96.1% | +2.78 pp |
| Perturbed | 1.91% | 97.4% | +1.91 pp |

Two disclosures. First, the perturbed figure **excludes 582 problems (11.64%)** whose perturbation
leaves the ground-truth answer unchanged; those score a memorizing model correct by construction.
Including them inflates Perturbed to 4.84% and inverts the ordering against Rephrased. Second, we no
longer write that performance "collapses to baseline": under strict scoring the uncontaminated floor
is **0.00%** (0.00% at four model sizes; 0.00–0.06% at 344M, and we inspected those three responses
individually — all are false positives on problems whose answer is 1), and the modified conditions
sit 1.9–2.8 pp above it. The supportable claim is
that modification removes the large majority of the advantage while leaving a small residual.

We hope the accuracy evidence, the temperature control, and the related-work reframing address the
concerns behind Quality = 2 and Originality = 2.

---

## Response to Reviewer 1wx9

Thank you — your W1/Q1 prompted the main new experiment in this response.

**W1 / Q1 — paraphrased rather than exact contamination in pretraining.**

We agree this is the most important missing condition, and we ran it. We modified our pipeline so
*what is injected* can differ from *what loss is measured on*, and pretrained new 34M models
(R = 32, 100, 316) while measuring cross-entropy on the **original** test set. To make these
comparable to the paper, we reconstructed the pretraining configuration behind Figure 3, so the
published exact-replica runs at the same doses serve as controls. The injected text is the only
variable.

Setting this up surfaced something we had not appreciated, and it improved the experiment. **Our
rephrased MATH set rephrases the problem but keeps the solution: 4,991 of 5,000 solutions (99.8%)
are byte-identical.** Since benchmark loss is cross-entropy on the solution, injecting it is
*solution-verbatim* leakage, not paraphrased leakage; reporting it as the latter would have
overstated paraphrase transfer considerably. Our perturbed set differs on both sides (4/5,000
identical). So instead of one condition we report an ablation over *which component of a leaked
document carries the effect*:

| Arm | Problem | Solution | R=32 | R=100 | R=316 |
|---|---|---|---|---|---|
| Uncontaminated | — | — | 7.1437 | 7.1437 | 7.1437 |
| Exact replicas | same | same | 2.5138 | 1.4526 | 0.5243 |
| Rephrased (solution-verbatim) | differs | **same** | 2.6125 | 2.0077 | 1.9573 |
| Perturbed (nothing verbatim) | differs | differs | 3.0741 | 3.0113 | 3.3705 |

1. **The problem text contributes little at low dose and more at high dose.** Replacing every
   problem while keeping solutions recovers **97.9%** of the exact-replica loss reduction at R = 32,
   **90.2%** at R = 100, **78.4%** at R = 316. At light contamination almost the entire effect is
   carried by solution text alone. As dose rises, exact replicas pull away — consistent with also
   learning the problem→solution *association*, which is what lets the model retrieve the right
   memorized solution. The rephrased arm nearly saturates (2.0077 → 1.9573) while exact keeps
   improving (1.4526 → 0.5243).
2. **This replicates a distinction from prior work we failed to cite.** Jiang et al. (2024) separate
   "text-only" from "ground-truth" contamination and find the latter far more damaging; our
   exact-versus-rephrased contrast reaches the same conclusion from the opposite direction.
3. **The perturbed arm never improves with dose, which bounds the confound rather than flagging
   it.** Perturbed loss goes 3.0741 → 3.0113 → 3.3705 — flat, then worse — while exact falls
   monotonically. That is the signature of domain adaptation: once MATH style and templates are
   learned from 32 replicas, further replicas of *different* items add nothing. The perturbed
   plateau (≈3.0 nats) estimates how much loss reduction genre learning alone buys; the exact arm's
   descent to 0.5243 is what requires verbatim solution text.

**Accuracy makes the point far more sharply than loss.** On the original problems, 0-shot:

| R | Exact | Rephrased | Perturbed |
|---|---|---|---|
| 32 | 0.56% | 0.24% | 1.34% |
| 100 | 1.70% | 1.58% | 1.16% |
| 316 | **7.22%** | 1.52% | 1.60% |

**Only exact-replica contamination produces a dose-response** — exact climbs 13×. Both arms whose
problem text differs plateau at ~1.5%, with all plateau movement inside the ±0.33 pp bootstrap
half-width and a verbatim solution rate of 0.000 throughout. Meanwhile *loss* calls the perturbed
model 57–88% as contaminated as the exact one. The two metrics disagree about the same models, and
accuracy is the one tracking what a benchmark reports. This is the clearest argument we can offer
for why contamination work should report accuracy, not loss alone — the same lesson as 8RFz's W1,
reached from the realistic-leakage direction.

**A dosing caveat we state rather than bury.** Our perturbed set is 21.7% smaller in tokens per copy
(1,132,643 vs 1,446,312), so at fixed R it delivers less contaminated text — perturbed R = 316 is
exact R ≈ 247 in contaminated tokens. The bias runs against us: the arm showing no effect receives a
smaller dose than its label implies, so the conclusion is conservative. We report dose in tokens as
well as replicas.

**A second caveat:** both modified corpora are still MATH-domain text, so part of any reduction is
domain adaptation rather than item-level leakage. Our R = 0 baseline saw no mathematics and does not
separate the two; cleanly separating them needs a fourth arm contaminated with *disjoint* math
problems, which we have not run. The perturbed number is an upper bound on realistic-leakage
transfer.

**These combine into one mechanism**, set out in item 6 of our general response: the ablation shows
models memorize the *solution string* largely independently of the problem attached to it, while
Table 1 shows accuracy collapsing to ~2.8% when the *evaluation* problem is rephrased. Together:
**memorization is of the solution text; retrieval is keyed on the exact problem text.**

**W2 — "rephrased contamination in training gives evaluation similar to an uncontaminated model.
Which is quite surprising."**

Thank you — this identified a genuine ambiguity in our presentation. Table 1 does **not** test that
direction. It tests **exact** contamination in training with **modified** evaluation. The symmetric
direction you describe is exactly the experiment above, and it was untested in the submission. We
have clarified this in the caption and text.

Your accompanying hypothesis — that model/train scale may be insufficient to see generalization from
contaminated data — is, we believe, correct, and we can now support it directly. An uncontaminated
344M model produced **0 correct answers in 5,000,000 samples**, with not one well-formed `\boxed{}`,
and 0 again in a separate **62,500-sample 0-shot** run (pass@25 = 0 on all 2,500 problems) — so this
is not a prompt-format artifact. There is no latent capability at this scale for contamination to
combine with, so anything breaking verbatim surface-form match removes the entire effect. We use
this to reconcile Finding 2 with prior work finding that rephrased contamination *does* transfer, in
already-capable models — a different regime rather than a contradictory result.

**W3 — small models, single benchmark, single mixture.** Conceded; all three reviewers raised it. We
keep the scale-for-control argument, state the limitation more prominently, and commit to a second
benchmark and model family for camera-ready.

**W4 — exact leakage makes loss results less surprising.** Agreed for the loss results specifically,
and we now say so. We would note the *irreducible-error* result is not in that category — the
surprise is not that loss drops but that a **single** replica pushes measured loss below the
extrapolated uncontaminated asymptote — and neither are the inference-time regimes, which concern
how memorized content is *emitted* rather than how well it is stored.

---

## Response to Reviewer aPBL

Thank you for the careful reading; Q2 and Q3 caught real gaps.

**W1 / W2 — small models, single dataset.** Conceded without argument. We retain the deliberate
scale-for-control tradeoff and the scaling-law bridge, state the limitation more prominently, and
commit to a second benchmark and model family for camera-ready. One observation from this rebuttal:
the *reason* our models show no paraphrase transfer is that they have zero baseline capability —
under format-free lenient scoring they sit at or below the scorer's own ~1.38% false-positive rate,
and sampling produces 0 correct answers in 5,000,000 draws — which makes the scale limitation a
**substantive boundary condition** on Finding 2 rather than a caveat. We present it that way.

**W3 — single seed, no error bars.** Added. We report 95% percentile bootstrap intervals over the
test problems (10,000 resamples); median half-width **0.33 pp** against effects spanning ~1% to
100%. We state explicitly that this quantifies **test-set sampling error, not seed-to-seed
variance** — presenting it as the latter would be worse than reporting nothing — and commit to
multiple seeds at pivotal configurations (the R ≈ 10–100 transition) for camera-ready.

**W4 — missing SFT hyperparameters.** You are right, they were absent. We add an appendix with the
full configuration, verified against the logged config of every run rather than transcribed from
defaults: AdamW, lr 1×10⁻⁴, cosine schedule with 0.2 warmup ratio, weight decay 0, gradient clipping
1.0, 1 epoch, per-device batch 16 with 4 gradient accumulation steps, max length 2048, bfloat16,
trained on the MATH **train** split with best-on-eval-loss checkpoint selection. All 39 runs share
these; only the initial checkpoint varies.

**Q1 — "multiple replicates seems somewhat contrived."** A fair challenge, and quantifying it
helped. As a fraction of the training tokens actually consumed: at **R = 1** the test set is
**0.03%–0.30%** depending on model size — at or below published real-world leakage estimates, and we
measure effects there. At the **top of our ladder** it reaches **74% and above**, which is
deliberately extreme. The ladder is best described as a **dose-response curve spanning
below-realistic to saturating**, and we describe it that way rather than implying every dose is
realistic. The paraphrased-contamination experiment addresses the complementary question of leakage
*mode* rather than dose.

**Q2 — how were the rephrasings/perturbations validated, and are difficulty/length distributions
matched?** We under-documented this and now add an appendix covering generation, validation, and
distributions. Difficulty is matched exactly by construction — both sets are index-aligned rewrites,
so the Level 1–5 distribution is identical across all three sets (8.7 / 17.9 / 22.6 / 24.3 / 26.5%).
Problem lengths are close (median 47 → 45, 46 tokens); rephrased solutions are unchanged, while
perturbed solutions are shorter (mean 162.8 vs 215.3), which we report because it means a perturbed
replica delivers ~22% fewer tokens at the same nominal dose. Validation combined automated checks
(answer-to-`\boxed{}` consistency, unmodified-problem detection, generation-failure markers, empty
answers) with manual inspection for faithfulness and problem-solution consistency. An earlier pair of
candidate datasets failed this audit — 26.6% of "perturbed" problems were unperturbed — and was
discarded and regenerated; the replacements have 0 unperturbed problems.

That appendix also reports a validation issue we found while preparing this response, which is
exactly what your question was aimed at: **11.64% of perturbed problems (582 of 5,000) have a
perturbation that leaves the ground-truth answer unchanged.** Those score a memorizing model correct
by construction. We exclude them, report the exclusion, and note that including them inflates the
perturbed score from 1.91% to 4.84% and inverts the ordering against rephrased (2.78%).

**Q3 — does the irreducible error come from fitting an asymptotic scaling law?** This is the right
question, and the claim's logical structure makes it more robust than it may appear: the
contaminated losses are **measured**, not fitted. Only the uncontaminated asymptote E(0) is
extrapolated. So the claim requires only a *conservative lower bound* on E(0) exceeding the measured
contaminated losses — not a correct functional form.

We now report that bound. Bootstrap refitting gives E(0) = **3.5942**, 95% interval **[3.5359,
3.6639]** (the point estimate reproduces the manuscript's 3.594, validating the refit). **33 of 35
contaminated runs (94.3%) have measured loss below the lower end of that interval**; the two
exceptions are the smallest model at R = 1 and R = 3. Three honest caveats: the intervals are
optimistically narrow, since each resample is refit by local optimization seeded at the full-data
solution rather than repeating the grid search; the R = 32 fit returns a degenerate asymptote (the
optimizer drives e₀ toward −∞, meaning the data admit no identifiable asymptote) and is flagged
unreliable; and the R = 1000 fit rests on 3 points with 79/300 resamples converging, also flagged.
We quote the conclusion, not the interval widths.

**Q4 — cross-domain contamination.** A good suggestion and the natural extension. The design we
would use: pretrain with MATH contamination as here, then evaluate on held-out mathematical tasks
sharing competence but not items (GSM8K, MMLU mathematics, a code benchmark requiring arithmetic),
measuring whether the advantage transfers across *domain* even when it does not transfer across
*surface form*. Our pass@k result predicts it will not at this scale — there is no capability for it
to transfer through — making it a clean test of the capability-boundary hypothesis once run at
larger scale. We add this to future work with the design specified rather than gestured at.

---

## Checklist before posting

- [x] All experiments complete; no placeholders.
- [x] **4-shot dropped as evidence entirely (2026-07-30).** An earlier draft argued 4-shot "does
      successfully teach the output format" (rates 0.43–0.89) yet buys no accuracy, therefore no
      capability hides beneath the format barrier. Those rates are from **contaminated (R ≥ 100)**
      models; at **R = 0 the 4-shot boxed rate is exactly 0.0000 at every size**, so the barrier was
      never removed for the baseline and the argument did not close — it invited the reply that our
      0.00% floor is itself a format artifact.

      Rather than patch it, we removed 4-shot from the argument. The format-conflation concern is
      answered by the **lenient, format-free** scorer bound (0.38–1.26% against a ~1.38%
      false-positive rate), which never requires the model to emit `\boxed{}`. The 4-shot vs 0-shot
      contrast ("96–192× reduction") is also cut: it is a confounded version of the retrieval-key
      result, since the prefix changes conditioning context *and* adds ~687 tokens *and* is
      out-of-distribution, whereas the contaminant ablation varies exactly one thing.

      **What is retained, and must be:** the disclosure that the submitted Table 1 and SFT figures
      were 4-shot, one illustrative magnitude (1.0000 vs 0.0052), and the mechanism in two
      sentences. Table 1's Original column moves from ~0% to 72.18% in the revision; numbers cannot
      move that far unexplained, and 8RFz's Q4 asks directly how Table 1 was computed.
      Consequence: the boxed-rate table (whose 34M and 344M cells were also wrong — 0.33–0.65 and
      0.57–0.66 should have been **0.40–0.52** and **0.59–0.66**) is gone from the response.
- [x] Sections condensed to ≈5–6k characters each for OpenReview.
- [x] Five citations added **and now `\citep`'d** in the appendix related work; bibliography
      regenerated and verified (128 entries, 0 undefined citations). Jiang 2024 was already cited.
- [x] SFT hyperparameter appendix and modified-test-set validation appendix written; manuscript
      compiles (47 pages).
- [ ] Confirm the exact OpenReview per-comment character limit before posting.
- [ ] Do not use the "~60× SFT collapse" figure — an artifact of comparing 0-shot pretrained against
      4-shot SFT. Matched at 4-shot it is 0.40% vs 0.20%.

### Numbers that changed — do not paste from an older draft

| Quantity | Superseded | **Use** |
|---|---|---|
| Table 1 (R≥100) | 70.19 → 2.74 / 1.89% | **72.18 → 2.78 / 1.91%** |
| Uncontaminated floor | ~1% | **0.00%** (0.00–0.06% at 344M, all hits inspected & spurious) |
| SFT | 72.31 → 3.00% | **72.95 → 2.80%** (14 conditions) |
| Finding #4 retention | 0.019 / 0.995 | **0.0188 / 0.9966** (~53×) |
| Temperature at τ=1.0 | 9.6% (retracted) | **25%** |
| Notebook 16 | 14/17, −4.72 nats | **17/17, −2.18 nats** |
| Answer-overlap inflation | 4.78% | **4.84%** |
| Contaminated fraction at R=1 | 0.02–0.21% | **0.03–0.30%** (of tokens actually trained) |
| pass@k | 4-shot only | **0-shot pass@25 = 0 on all 2,500 problems** |
| 344M R=0 0-shot | "all ten runs failed" | **recovered**: sweeps `woygzpil`/`oj6o8idv`, strict 0.00–0.06% |
| 4-shot boxed rate, 34M | 0.33–0.65 | **0.40–0.52** |
| 4-shot boxed rate, 344M | 0.57–0.66 | **0.59–0.66** |

`notebooks/11_*/results/protocol_sensitivity_rescored.csv` (`strict_score`) is authoritative; never
quote the 0-shot column of `protocol_sensitivity.csv`.
