# Rebuttal draft — NeurIPS 2026 submission 32216

Drafted 2026-07-29. Paste-ready per-reviewer comments. Scores: **8RFz 3** (Quality 2,
Originality 2, conf 4) · **1wx9 4** · **aPBL 3**. The AC named 8RFz's loss-vs-correctness
objection as the pivotal critique.

**Every number below is 0-shot greedy unless explicitly labelled otherwise.** Read
[`PROTOCOL_CONFOUND.md`](PROTOCOL_CONFOUND.md) before editing any figure. Sources are in
[`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md).

**[PENDING]** marks numbers still running as of 2026-07-30 00:05:
- contaminant ablation, sweeps `mxamktp0` (rephrased, 2 of 3 done) and `vrxwx4dz` (perturbed);
  analysis is ready at `notebooks/21_paraphrased_contamination/` — just run it.

Every other claim stands without them. If a run does not land, delete the row rather than
softening it — a table with an honest gap beats a hedge.

---

## General response (post once, addressed to all reviewers and the AC)

We thank all three reviewers and the AC. The AC identified 8RFz's loss-vs-correctness objection
as the one critique that questions whether our evidence supports our claims. We agree that was
the right thing to single out, and we have answered it with new measurements rather than
argument: **we evaluated all 137 overtraining checkpoints and all 39 SFT checkpoints in Math
Verify (accuracy) space**, which is the metric the objection asks for. We summarise below, and
in doing so we correct two things in our own submission that we found while doing this work.

**1. Findings 4 and 5 now rest on accuracy, not loss.** Accuracy tracks loss, so the
stealth-contamination scenario 8RFz raises — accuracy persisting while loss rises, making
contamination harder to detect but no less inflationary — does **not** occur in our setting. The
new measurement is also sharper than our original claim, and it obliges us to weaken that claim:
dilution is **threshold-dependent**. For 93M over ot 1→16, the contamination advantage retained
is **0.0188 at R = 100 but 0.9966 at R = 1000**. Below the memorization threshold, overtraining
suppresses contamination by more than an order of magnitude; above it, 16× more fresh data does
essentially nothing. We were wrong to state "overtraining dilutes contamination" without
qualification — it reads as a mitigation, and it fails precisely in the heavy-leakage regime
that matters most. The revision states the threshold behaviour instead.

**2. A correction we owe the reviewers: an evaluation-protocol inconsistency in our own paper.**
While re-running these evaluations we discovered that our generative evaluations were not run
under a single protocol. Figure 1 and every teacher-forced result are **0-shot**; Table 1 and
the SFT figures are **4-shot** (the EleutherAI `minerva_math` default, which we adopted partway
through the project). This matters enormously at our scale: the same checkpoint
(344M, R = 3162, greedy, identical scoring code) scores **1.0000 at 0-shot and 0.0052 at
4-shot**.

The cause is mechanistic, not a bug. At 0-shot the prompt reproduces the opening of the
memorized training document and the model emits the stored solution verbatim. At 4-shot, four
unrelated worked examples precede the problem, the prompt no longer matches any memorized
context, and the model produces fluent but unrelated text. Prompts are ~687 tokens at the median
against a 2,048-token pretraining sequence length, so this is not context overflow. We had in
fact already recognised this mechanism for teacher forcing — our code carries a note that "adding
a 4-shot prefix would change the conditioning context and dilute the memorization signal" — and
we simply failed to carry that reasoning to the generative evaluations.

We have standardised on **0-shot** and re-run everything affected. We want to be explicit that
this is not merely a convenience. The reason 4-shot was adopted was to let uncontaminated models
demonstrate the `\boxed{}` output format, on the theory that 0-shot conflates format knowledge
with reasoning. We tested that rationale directly, and it does not hold at this scale.

Doing so required removing a second confound of our own making: our 0-shot and 4-shot sweeps sat
on either side of the same commit that also tightened scoring (from `math_verify.parse()`, which
extracts bare numbers from free text, to requiring a well-formed `\boxed{}`). Comparing the
originally logged scores therefore compared prompt format *and* scoring rule at once. Because raw
generations are retained, we rescored every run with the **same** boxed-required scorer. With
scoring held constant:

| Model | R = 0, 0-shot | R = 0, 4-shot | 4-shot `\boxed{}` rate, R ≥ 100 |
|---|---|---|---|
| 34M | 0.0000 | 0.0000 | 0.33–0.65 |
| 62M | 0.0000 | 0.0000 | 0.65 |
| 93M | 0.0000 | 0.0000 | 0.43–0.70 |
| 153M | 0.0000 | 0.0000 | 0.60–0.89 |
| 344M | 0.0000 | 0.0000 | 0.57–0.66 |

The four-shot prefix **does** successfully teach the output format — the rate of well-formed
`\boxed{}` responses rises from near zero to 0.43–0.89 — and uncontaminated accuracy nonetheless
remains **exactly 0.0000** at every model size. Removing the format barrier reveals no capability
underneath it. (The apparent 0.4–1.3% we previously saw for uncontaminated models at 0-shot was
entirely lenient-scorer false positives, consistent with its measured ~1.4% rate.) Format
conflation therefore cannot be what drives our 0-shot results, and the headline contrast is
undisturbed: 153M at R = 316 scores **0.9984 at 0-shot versus 0.0078 at 4-shot** under identical
strict scoring.

A mechanistic detail supports the reading. At 0-shot the `\boxed{}` rate *rises with
contamination dose* (153M: 0.000 → 0.009 → 0.047 → 0.72 → 0.98 → 1.000 as R goes 0 → 1 → 10 → 32
→ 100 → 316). The contaminated model learns the output format from the injected solutions
themselves. Contamination supplies format and answer together; four in-context examples supply
format alone, and format alone is worth nothing.

We report both protocols in the revision, and we think the sensitivity is itself a result worth
stating: **contamination-driven gains in small from-scratch models are memorization brittle
enough that four in-context examples erase them** (96–192× reduction at high contamination),
while cross-entropy stays low under both. That is a stronger and more precise version of our
Finding 2 than the original rephrase/perturbation table, and it speaks directly to the
loss-versus-correctness distinction the AC highlighted.

**3. Uncertainty quantification** (aPBL W3, AC bullet 3). We add 95% percentile bootstrap
intervals over the 5,000 test problems (10,000 resamples) for every Math Verify number; median
half-width is **0.33 percentage points** against effects spanning ~1% to 100%. We state plainly
in the paper that this is test-set sampling error and **not** multi-seed variance, and we commit
to multiple seeds at pivotal configurations for the camera-ready.

**4. Related-work framing** (8RFz W3, AC bullet 5). We accept this criticism in full. We remove
the "first targeted examination" phrasing, add the missing references, and — most importantly —
situate our Findings 1 and 2 against the work they respectively replicate and appear to
contradict. Details in our response to 8RFz.

**5. Realistic leakage** (1wx9 W1/Q1, aPBL Q1, AC bullet 1). We have run the experiment: injecting
**paraphrased** MATH into pretraining while measuring loss on the original test set. Details in
our response to 1wx9.

We are grateful for reviews that were specific enough to be actionable. Every criticism above
that we could address with data, we did.

---

## Response to Reviewer 8RFz

Thank you — your W1 identified the weakest link in the paper, and the AC was right to elevate
it. We have addressed it with new measurements, and it changed one of our conclusions.

**W1 / Q1 — "Do Findings 4 and 5 also hold for Math Verify score?"**

Yes, and we now measure it directly rather than inferring it from Figure 11's correlation. We
evaluated **all 137 overtraining checkpoints** and **all 39 SFT checkpoints** in Math Verify
space.

You raised the sharper possibility that accuracy might persist while loss rises — which would
make contamination *harder* to detect by perplexity while still inflating scores, and would make
"dilution" the wrong word entirely. We looked for exactly that and **it does not occur**:
accuracy tracks loss across the grid.

But the accuracy-space measurement does force us to weaken Finding 4. Dilution is
**threshold-dependent**. For 93M over ot 1→16, the fraction of the contamination advantage
retained is:

| Configuration | Advantage retained after 16× overtraining |
|---|---|
| 93M, R = 100 | **0.0188** |
| 93M, R = 1000 | **0.9966** |

Same model, same multiplier range, ~53× difference. The mechanism is dilution of the
contaminated *token fraction*, which is why it stops working once that fraction stays high. Our
original phrasing — "the performance boost from contamination diminishes when overtraining with
fresh data" — is therefore misleading as an unqualified statement, and we have replaced it with
the threshold formulation. We think this is a more useful finding than the one we submitted: it
says overtraining is not a mitigation for the leakage regimes practitioners should worry about.

For Finding 5, at matched 0-shot protocol and matched scoring, SFT takes mean Math Verify from
**70.89% to 3.00%** across the 13 conditions that had something to lose (≥ 5% before SFT), with a
median retained fraction of **0.028** (range 0.001–0.302). The range spans more than two orders of
magnitude, so we quote it rather than a single multiplier.

**We also found a regime where your objection is exactly right, and we think it is the most
interesting result in this response.** Pursuing 1wx9's paraphrased-contamination request, we
pretrained models on rephrased problems paired with verbatim original solutions, then evaluated
them 0-shot on the original problems:

| R | Loss, exact | Loss, rephrased | Accuracy, exact | Accuracy, rephrased | Verbatim solution rate |
|---|---|---|---|---|---|
| 100 | 1.4526 | 2.0077 | 1.70% | 1.58% | 0.000 |
| 316 | 0.5243 | 1.9573 | 7.22% | **1.52%** | **0.000** |

(Uncontaminated: loss 7.1437, accuracy 0.00%.)

At R = 316 the rephrased model's cross-entropy on the original solutions is 78% of the way from
the uncontaminated baseline to the exact-replica model's — by any loss-based measure it is
heavily contaminated. Yet it scores 1.52% and reproduces the gold solution verbatim **0 times out
of 5,000**. It holds the answer and cannot retrieve it. Loss and correctness genuinely come
apart, precisely as you argued they could.

The mechanism this exposes is that **memorization is of the solution text while retrieval is keyed
on the problem text**. Rephrasing at training time stores the solution without the key;
rephrasing at evaluation time (our Table 1) withholds the key from a model that has one. Both
collapse generation for the same reason, which unifies Finding 2 with this new result.

One further consequence worth flagging, because it runs opposite to the concern you raised: you
noted that accuracy persisting while loss rises would let contamination evade perplexity-based
detection. We observe the *other* asymmetry — **perplexity would flag these models loudly while
their benchmark scores are barely inflated.** That is a false-positive mode for loss-based
detection, and since realistic leakage rarely reproduces benchmark problems verbatim, it may be
the more common case in practice.

None of this rescues Findings 4–5 by itself; those stand on the accuracy-space measurements
above, where the metrics do track. But it means we accept your framing rather than merely
answering it: loss and correctness are not interchangeable, we can now show a case where they
diverge sharply, and we report both metrics everywhere in the revision.

**W2 / Q2 — the temperature confound.**

You are right that Finding 6 as written does not separate "temperature reduces contamination
effects specifically" from "temperature degrades generation generally." The clean control is the
contamination *advantage* at **matched** temperature — score(R) − score(R = 0) with both terms at
the same τ — so any uniform degradation cancels:

| τ | 0 (greedy) | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Fraction of greedy advantage retained | 100% | 92% | 77% | 55% | 20% | **9.6%** | 0.02% |

So the effect is contamination-specific: at τ = 1.0 — the model's own distribution, not a hot
setting — **over 90% of the advantage is already gone**, while general degradation has been
controlled for. Above τ ≈ 1.3 we agree the two explanations are no longer separable and
everything is degrading; we now restrict the claim to τ ≤ 1 and say so explicitly.

**W3 / Q3 — related work, and the conflict with prior rephrasing results.**

We largely accept this criticism; the framing complaint is fair and we remove the "first targeted
examination of contamination in generative tasks" claim. We have added **Palavalli et al. (2024)**,
**Mehrbakhsh et al. (2024)**, **Dekoninck et al. (2024a, "Evading Data Contamination Detection for
Language Models is (too) Easy")**, **Dekoninck et al. (2024b, "ConStat")**, and **Godey et al.
(2025, "Gaperon")**.

One respectful correction: **Jiang et al. (2024) is already cited** in the submission — in the
introduction's list of controlled contamination studies, again in the introduction's discussion of
which benchmarks prior work used, and at paragraph length in the appendix related work, where we
describe their text-only versus ground-truth contamination conditions and their finding that
n-gram detection can be bypassed by paraphrasing or partial leaks. We mention this only because it
bears on the Originality assessment; the substantive point about *situating* our findings against
that literature stands, and we have acted on it.

We also thank you for Godey et al. (2025) in particular, which we had missed and which is closely
related: their "late deliberate contamination" — continued training on mixtures containing test
sets — is the large-scale analogue of our overtraining and SFT conditions, and we now discuss it
alongside Finding 4.

More substantively, we now situate both findings:

- **Finding 1 replicates** the repeat-count effect reported by Jiang et al. (2024) and Dekoninck
  et al. (2024), and we say so rather than presenting it as new.
- **Finding 2 appears to conflict** with Mehrbakhsh et al. (2024) and Dekoninck et al. (2024),
  who find that rephrased contamination *does* produce contamination effects on mathematical
  generative evaluations. We do not think this is a contradiction; we think it is a regime
  boundary, and we can now support that with a measurement rather than a hypothesis.

  The prior studies inject contamination into models that are **already capable** of the task.
  Such a model can bridge a surface-form change: it has the underlying competence, and
  contamination supplies an advantage that survives paraphrase. Our models have **no such
  competence to bridge with**, which we establish three independent ways on the uncontaminated
  344M checkpoint:

  - **Sampling, 4-shot:** 5,000 problems × 1,000 samples at τ = 1.0 = **5,000,000 samples, 0
    correct**, and not one containing a well-formed `\boxed{}` — despite four worked examples
    demonstrating the format in every prompt.
  - **Sampling, 0-shot:** 62,500 samples over 2,500 problems at τ = 1.0 — **pass@25 = 0 on every
    problem**, and again not one well-formed `\boxed{}`. We ran this separately because the figure
    above uses the 4-shot prefix and so cannot, on its own, speak to 0-shot capability.
  - **Greedy, both protocols:** exactly 0.0000 under boxed-required scoring at every model size
    (table in the general response).

  With zero latent capability, every point of contaminated performance is verbatim memorization,
  and verbatim memorization cannot transfer across a rephrasing.

  This makes the two results complementary and yields a concrete prediction: the transfer of
  contamination across paraphrase should be a function of the model's underlying capability, and
  should switch on somewhere between our scale and theirs. We state this as the reconciliation
  and flag it as the natural next experiment.

**Q4 — "How are the values in Table 1 calculated?"**

This question caused us to audit Table 1, and we owe you a direct answer: **the printed values
(0.00%–0.04%) do not reproduce**, and we are replacing them. They were produced on separate
infrastructure by a co-author (credited in the acknowledgements) and predate our current
rephrased/perturbed datasets; they correspond to no run we can now point at. We should have
verified them before submission.

We have re-measured the table from scratch across 39 checkpoints × 2 modified datasets at 0-shot,
averaged over the 13 contaminated checkpoints with R ≥ 100:

| Condition | Math Verify | Advantage removed | Residual above floor |
|---|---|---|---|
| Original | 72.18% | — | — |
| Rephrased | 2.78% | 96.1% | +2.78 pp |
| Perturbed | 1.91% | 97.4% | +1.91 pp |

Two disclosures that go with these numbers:

1. The perturbed figure **excludes 582 problems (11.64%)** whose numerical perturbation leaves
   the ground-truth answer unchanged. Those score a memorizing model correct by construction.
   Including them inflates Perturbed to 4.84% and inverts the expected ordering relative to
   Rephrased. We now report the exclusion and both numbers.
2. We no longer write that performance "collapses to baseline." Under boxed-required scoring
   the uncontaminated floor is **exactly 0.00%** at every model size, and the modified conditions
   sit **1.9–2.8 percentage points above it**. The supportable claim is that modification removes
   the large majority of the contamination advantage while leaving a small but consistent
   residual.

The re-measurement also extends the table to model sizes the submitted version asserted but did
not show, and gives it provenance we can point at.

We hope the accuracy-space evidence for Findings 4–5, the temperature control, and the
related-work reframing address the concerns behind Quality = 2 and Originality = 2.

---

## Response to Reviewer 1wx9

Thank you — we appreciate the assessment of the controlled design, and your W1/Q1 prompted the
main new experiment in this response.

**W1 / Q1 — paraphrased rather than exact contamination in pretraining.**

We agree this is the most important missing condition, and we ran it. We modified our pretraining
pipeline so that *what is injected* can differ from *what loss is measured on*, and pretrained new
34M models (R = 32, 100, 316) while measuring cross-entropy on the **original** test set
throughout.

To make these comparable to the paper rather than merely internally consistent, we reconstructed
the exact pretraining configuration behind Figure 3, so the published exact-replica runs at the
same three doses serve as the control without retraining. The injected text is the only variable.

Setting this up surfaced something we had not appreciated about our own data, and it changed the
experiment for the better. **Our rephrased MATH set rephrases the problem but keeps the solution:
4,991 of 5,000 solutions (99.8%) are byte-identical to the original.** Since the benchmark loss is
cross-entropy on the solution, injecting that set is *solution-verbatim* leakage, not paraphrased
leakage. Reporting it as the latter would have overstated paraphrase transfer considerably. Our
perturbed set, which changes the numbers, differs on both sides (4/5,000 identical solutions).

So instead of one condition we report an ablation over *which component of a leaked document
carries the effect* — which we think is a more useful answer to your question than the one we set
out to give:

| Arm | Problem | Solution | R = 32 | R = 100 | R = 316 |
|---|---|---|---|---|---|
| Uncontaminated | — | — | 7.1437 | 7.1437 | 7.1437 |
| Exact replicas | same | same | 2.5138 | 1.4526 | 0.5243 |
| Rephrased (solution-verbatim) | differs | **same** | 2.6125 | 2.0077 | 1.9573 |
| Perturbed (nothing verbatim) | differs | differs | [PENDING] | [PENDING] | [PENDING] |

Two things follow from the rows we have:

1. **The problem text contributes little at low dose, and increasingly more at high dose.**
   Replacing every problem statement while keeping the solutions recovers **97.9%** of the
   exact-replica loss reduction at R = 32, **90.2%** at R = 100, and **78.4%** at R = 316. At light
   contamination almost the entire effect is carried by the solution text alone. As dose rises,
   exact replicas pull away — consistent with them additionally learning the problem→solution
   *association*, which is what lets the model retrieve the right memorized solution rather than
   merely having memorized solutions in general. Notably the rephrased arm nearly saturates
   (2.0077 → 1.9573 from R = 100 to R = 316) while the exact arm keeps improving
   (1.4526 → 0.5243).
2. **This replicates a distinction from prior work we had failed to cite.** Jiang et al. (2024)
   separate "text-only" from "ground-truth" contamination and find the latter far more damaging;
   our exact-versus-rephrased contrast reaches the same conclusion from the opposite direction. We
   now cite them for it (see also our response to 8RFz on related work).

The perturbed arm — where nothing is leaked verbatim — is the one that speaks directly to
realistic leakage, and we report it above.

**These two results combine into a single mechanism**, which we think is the most useful thing to
come out of this rebuttal. The ablation says contaminated models memorize the *solution string*
largely independently of the problem it was attached to. Table 1 says that when the *evaluation*
problem is rephrased, accuracy collapses to ~2.8%. Both are true at once, and together they say:
**memorization is of the solution text; retrieval is keyed on the exact problem text.** The model
holds the answer and cannot get to it when the question is reworded. That is why contamination
inflates benchmark scores so dramatically and yet survives no surface-form change whatsoever, and
it explains the loss-versus-accuracy gap the AC asked about — cross-entropy sees the stored
solution, generation needs the retrieval key.

**A caveat we state rather than hide:** both modified corpora are still MATH-domain text with
MATH-style solutions, so part of any reduction is domain adaptation rather than item-level
leakage. Our R = 0 baseline saw no mathematics at all and therefore does not separate the two.
Cleanly separating them needs a fourth arm contaminated with *disjoint* math problems, which we
have not run. The perturbed number should be read as an upper bound on realistic-leakage transfer.

On the design question behind the criticism: we chose exact replicas as a causal testbed — the
same control-for-realism tradeoff we articulate against Bordt et al. — and we now frame that
result as what it is, an **upper bound** on contamination effects whose *lifecycle dynamics*
(pretraining → overtraining → SFT → inference) are what the paper characterises. Partial,
translated, and discussion-embedded leakage remain untested and are now listed explicitly as scope
limits rather than left implicit.

**W2 — "if we have a rephrased test set contamination in training, the evaluation on test set is
similar to an uncontaminated model. Which is quite surprising."**

Thank you for this — it identified a genuine ambiguity in our presentation. Table 1 does **not**
test that direction. It tests **exact** contamination in training with **modified** evaluation
(rephrased/perturbed test sets). The symmetric direction you describe — rephrased contamination
in training, original test set at evaluation — is exactly the experiment above, and it was
untested in the submission. We have clarified the direction in the caption and text so the
inference is not invited.

Your accompanying hypothesis — "maybe model/train scale is not enough to see the generalization
from contaminated data" — is, we believe, correct, and we can now support it directly rather
than speculatively. An uncontaminated 344M model produced **0 correct answers in 5,000,000
samples** (5,000 problems × 1,000 samples at τ = 1.0), with not one well-formed `\boxed{}`, and
0 correct again in a separate **62,500-sample 0-shot** run (pass@25 = 0 on all 2,500 problems) — so this is not an artifact of the
prompt format. There
is no latent capability at this scale for contamination to combine with, so anything that breaks
verbatim surface-form match removes the entire effect. We now use this to reconcile our Finding 2
with prior work (Mehrbakhsh et al. 2024; Dekoninck et al. 2024) that finds rephrased
contamination *does* transfer — in already-capable models, which is a different regime rather
than a contradictory result.

**W3 — small models, single benchmark, single mixture.**

Conceded; all three reviewers raised it and we do not contest it. We keep the scale-for-control
argument but state the limitation more prominently, and we commit to a second benchmark and a
second model family for the camera-ready.

**W4 — exact leakage makes loss/perplexity results less surprising.**

Agreed for the loss results specifically, and we now say so. We would note that the
*irreducible-error* result is not in that category — the surprise there is not that loss drops
but that a **single** replica pushes measured loss below the extrapolated uncontaminated
asymptote — and neither are the inference-time regimes, which concern how memorized content is
*emitted* rather than how well it is stored.

---

## Response to Reviewer aPBL

Thank you for the careful reading, particularly Q2 and Q3, which caught real gaps.

**W1 / W2 — small models, single dataset.**

Conceded without argument; raised independently by all three reviewers. We retain the
deliberate scale-for-control tradeoff and the scaling-law bridge, but we state the limitation
more prominently and commit to a second benchmark and a second model family for the
camera-ready. We would add one observation that emerged from this rebuttal: the *reason* our
models show no paraphrase transfer is that they have zero baseline capability (0 correct in
5,000,000 samples at 4-shot and 62,500 more at 0-shot), which makes the scale limitation a
**substantive boundary condition** on
Finding 2 rather than merely a caveat. We now present it that way.

**W3 — single seed, no error bars.**

Added. We report 95% percentile bootstrap intervals over the 5,000 test problems (10,000
resamples) for every Math Verify number; the median half-width is **0.33 percentage points**
against effects spanning ~1% to 100%. We state explicitly in the paper that this quantifies
**test-set sampling error and not seed-to-seed variance**, since presenting it as the latter
would be worse than reporting nothing, and we commit to multiple seeds at pivotal configurations
(the R ≈ 10–100 transition, where variance should matter most) for the camera-ready.

**W4 — missing SFT hyperparameters.**

You are right, they were absent. We add an appendix with the full SFT configuration
(optimizer, learning-rate schedule, batch size, sequence length, epochs, and the train/test split
handling), matching the detail already given for pretraining.

**Q1 — "multiple replicates seems somewhat contrived."**

A fair challenge, and quantifying it helped us. Expressed as the fraction of the training token
budget occupied by contaminated text:

- At **R = 1**, the test set is **0.02%–0.21%** of the budget depending on model size — at or
  below published real-world leakage estimates. We measure effects at this dose.
- At the **top of our ladder** it reaches **67%–92%**, which is deliberately extreme.

So the ladder is best described as a **dose-response curve spanning from below-realistic to
saturating**, and we now describe it that way rather than implying every dose is realistic. The
paraphrased-contamination experiment above addresses the complementary question of realistic
leakage *mode* rather than dose.

**Q2 — how were the rephrasings/perturbations validated, and are difficulty/length distributions
matched?**

We under-documented this. We add an appendix covering the generation procedure and validation,
including a manual spot-check protocol over sampled problems (answer-`\boxed{}` consistency,
faithfulness of rephrasing, solution-problem consistency) and the length/difficulty-level
distributions against the original test set.

That appendix also reports a validation issue we found while preparing this response, which we
think is exactly the kind of thing your question was aimed at: **11.64% of perturbed problems
(582 of 5,000) have a numerical perturbation that leaves the ground-truth answer unchanged.**
Those problems score a memorizing model correct by construction and therefore cannot support a
memorization-versus-generalization claim. We exclude them, report the exclusion, and note that
including them inflates the perturbed score from 1.91% to 4.84% and inverts the ordering against
rephrased (2.78%).

**Q3 — "Does the irreducible error come from fitting an asymptotic scaling law? It's a strong
claim ... may depend on assumptions of the functional form and extrapolation."**

This is the right question to ask, and the answer is that the claim's logical structure makes it
more robust than it may appear. The contaminated losses are **measured**, not fitted. Only the
uncontaminated asymptote E(0) is extrapolated. So the claim requires only a *conservative lower
bound* on E(0) that still exceeds the measured contaminated losses — not a correct functional
form.

We now report that bound. Refitting with bootstrap resampling gives E(0) = **3.5942**, 95%
interval **[3.5359, 3.6639]** (the point estimate reproduces the manuscript's 3.594, which
validates the refit). **33 of 35 contaminated runs (94.3%) have measured loss below the lower
end of that interval.**

Three honest caveats we state alongside it: the intervals are optimistically narrow, because each
resample is refit by local optimization seeded at the full-data solution rather than by repeating
the full grid search; the R = 32 fit returns a degenerate asymptote (the optimizer drives e₀
toward −∞, meaning the data admit no identifiable asymptote) and is flagged unreliable; and the
R = 1000 fit rests on 3 points with only 79/300 resamples converging, also flagged. We quote the
conclusion, not the interval widths.

**Q4 — cross-domain contamination.**

A good suggestion and we agree it is the natural extension. The design we would use: pretrain
with MATH contamination as here, then evaluate on held-out mathematical tasks that share
competence but not items (GSM8K, a subset of MMLU mathematics, and a code benchmark requiring
arithmetic reasoning), measuring whether the contamination advantage transfers across *domain*
even when it does not transfer across *surface form*. Our pass@k result predicts it will not at
this scale — there is no capability for it to transfer through — which makes it a clean test of
the capability-boundary hypothesis above once run at larger scale. We add this to future work
with the design specified rather than gestured at.

---

## Checklist before posting

- [ ] Replace both **[PENDING]** blocks with measured paraphrased-contamination results, or
      delete them.
- [ ] Confirm OpenReview per-comment character limits; the general response may need trimming
      or splitting.
- [ ] Do not use the "~60× SFT collapse" figure anywhere — it is an artifact of comparing 0-shot
      pretrained against 4-shot SFT. Matched at 4-shot it is 0.40% vs 0.20%.
- [x] Citations added to `references_rylan.bib` and metadata verified against the ACL Anthology
      and arXiv: `palavalli2024taxonomy`, `mehrbakhsh2024confounders`, `dekoninck2024evading`
      (2402.02823), `dekoninck2024constat` (2405.16281), `godey2025gaperon` (2510.25771). Bib
      validates at 151 entries, no duplicate keys. Jiang 2024 was already present *and cited*.
- [ ] Actually `\citep` the five new keys in the related-work rewrite — added to the bib is not
      the same as cited, and an uncited entry will not appear in the bibliography.
- [ ] Cross-check every number here against `REBUTTAL_EVIDENCE.md` one final time.
