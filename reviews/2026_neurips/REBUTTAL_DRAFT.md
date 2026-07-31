# Rebuttal draft — NeurIPS 2026 submission 32216

Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) · **1wx9 4** (all 3s) · **aPBL 3** (all 3s).

**Where the score movement is.** 8RFz is dragged by two subscores: Quality 2, from their W1 (the
evidence doesn't support Findings 4–5), and Originality 2, from their W3 (framing understates prior
work; our findings replicate and conflict with it undiscussed). Those are the two biggest levers in
this response. aPBL's subscores are all 3s, so their rating turns on *completeness of reporting* —
seeds, SFT details, validation — and three of their four concerns are now fully addressed. 1wx9 is
already at 4 and asked for one experiment, which we ran and which produced the best result here.

**Writing rule.** Every paragraph presents a result or answers a question. No process narration, no
provenance archaeology, no account of how we found our own errors. Corrections appear once, in a
clause, and we move on. Supporting detail lives in [`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md)
and in the paper. A reviewer dragged through our measurement hygiene is a reviewer wondering why we
are wasting their time.

---

## General response

We thank the reviewers. Two new experiments, run in response to these reviews, address the central
criticisms.

**1. Findings 4 and 5, now measured in accuracy** (8RFz W1/Q1; the AC's pivotal point). We evaluated
all 137 overtraining and all 39 SFT checkpoints under Math Verify. Accuracy tracks loss, so the
possibility 8RFz raises — accuracy persisting while loss rises, hiding contamination from perplexity
while still inflating scores — does not occur here.

The measurement also sharpens Finding 4 into something more useful than what we submitted. Dilution
is **threshold-dependent**: at 93M over ot 1→16, the contamination advantage retained is **0.0188 at
R = 100 but 0.9966 at R = 1000**. Overtraining suppresses light contamination by more than an order
of magnitude and does essentially nothing to heavy contamination. It is not a mitigation for the
leakage regimes that matter most, and we now say so.

**2. Contamination that is paraphrased rather than exact** (1wx9 Q1; aPBL Q1; AC). We pretrained new
models in which the *injected* text differs from the text loss is measured on, giving a three-arm
ablation over which component of a leaked document carries the effect. **Only exact replicas produce
a dose-response in accuracy** — 0.56% → 7.22% from R = 32 to 316, while both arms whose problem text
differs plateau near 1.5%.

The rephrased arm — rephrased problems, verbatim solutions — separates loss from correctness, which
is precisely the distinction 8RFz's objection rests on. Its cross-entropy on the original solutions
is 78% of the way from clean to fully-contaminated, yet it scores **1.52%** against the exact
model's **7.22%** and never reproduces a gold solution. Measured on its *own* training items it
scores 7.56% and regurgitates verbatim 5.34% of the time. It has memorized the solutions and cannot
reach them.

**Memorization is of the solution text; retrieval is keyed on the problem text.** Rephrasing at
training time stores the solution without the key; rephrasing at evaluation time (Table 1) withholds
the key from a model that has one. One mechanism now explains Finding 2, the new ablation, and the
loss-versus-accuracy gap the AC asked about — and it inverts the usual detection worry, since
perplexity flags these models loudly while their benchmark scores stay near clean.

**3. Reporting.** We add confidence intervals to every Math Verify number (median half-width 0.12
pp), five missing references with our findings situated against them, and appendices giving the SFT
configuration and the construction and validation of the modified test sets. Table 1 now includes an
Original column and is measured at the same protocol as Figure 1, so the comparison is
self-contained.

Together these strengthen the paper's central claim: contamination inflates generative benchmark
scores enormously while creating no capability whatsoever, and the inflation is keyed so tightly to
the exact problem text that it survives no surface-form change — in training or in evaluation.

---

## Reviewer 8RFz

**W1/Q1 — do Findings 4 and 5 hold for Math Verify score?**

Yes. We evaluated all 137 overtraining and all 39 SFT checkpoints in accuracy space rather than
inferring it from Figure 11's correlation. You raised the sharper possibility that accuracy might
persist while loss rises; we looked for that specifically and it does not occur.

The measurement improves Finding 4. Dilution is **threshold-dependent**: at 93M over ot 1→16, the
advantage retained is **0.0188 at R = 100** and **0.9966 at R = 1000**. The mechanism is dilution of
the contaminated *token fraction*, which stops working once that fraction stays high — so
overtraining is not a mitigation for heavy leakage. Our unqualified phrasing was misleading and is
replaced by the threshold formulation. For Finding 5, SFT takes mean Math Verify from **72.95% to
2.80%** across the 14 conditions scoring ≥ 5% beforehand (median retained fraction 0.022).

**We also found a regime where your objection is exactly right, and we think it is the most
interesting result in this response.** Pretraining on rephrased problems paired with verbatim
solutions, then evaluating on the original problems:

| R = 316 | Loss | Accuracy | Reproduces gold solution |
|---|---|---|---|
| Uncontaminated | 7.1437 | 0.00% | — |
| Exact replicas | 0.5243 | 7.22% | frequently |
| Rephrased problems, verbatim solutions | 1.9573 | **1.52%** | **0 / 5,000** |

By any loss-based measure that model is heavily contaminated. It nonetheless scores 1.52% and never
reproduces a gold solution. A positive control on its *own* training items gives 7.56% with a 5.34%
verbatim rate, so this is a retrieval failure rather than a learning failure: **memorization is of
the solution text while retrieval is keyed on the problem text.** Loss and correctness come apart
sharply, as you argued they could, and we report both metrics throughout the revision.

One consequence runs opposite to the concern you raised. Rather than contamination evading
perplexity, we observe perplexity flagging models whose benchmark scores are barely inflated — a
false-positive mode for loss-based detection, and plausibly the more common one in practice, since
real leakage rarely reproduces benchmark problems verbatim.

**W2/Q2 — does temperature reduce contamination effects specifically, or degrade generation
generally?**

The clean control is the contamination *advantage* at matched temperature — score(R) − score(R = 0)
with both terms at the same τ — so any uniform degradation cancels:

| τ | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Advantage retained | 100% | 98% | 90% | 72% | 39% | **25%** | 0.4% |

The effect is contamination-specific: by τ = 1.0 — the model's own distribution, not a hot setting —
three quarters of the advantage is gone with general degradation controlled for. Above τ ≈ 1.3 the
two explanations are no longer separable, so we restrict the claim to τ ≤ 1 and say so explicitly.

**W3/Q3 — related work, and why Finding 2 conflicts with prior rephrasing results.**

We accept this criticism and have acted on it. We remove the "first targeted examination" framing,
add Palavalli et al. (2024), Mehrbakhsh et al. (2024), Dekoninck et al. (2024a, 2024b) and Godey et
al. (2025), and state plainly that **Finding 1 replicates** the repeat-count effect rather than
presenting it as new.

On the conflict, we believe there is a real reconciliation, and it strengthens both sets of results.
Mehrbakhsh et al. and Dekoninck et al. inject contamination into models **already capable** of the
task; contamination there supplies an advantage on top of existing competence, and that advantage
survives a paraphrase. Our models have no competence for it to sit on top of: an uncontaminated 344M
model produces **0 correct answers in 5,000,000 samples**, and scores no better than scorer noise
even when output formatting is ignored entirely. With no latent ability, all contaminated
performance is verbatim memorization, and verbatim memorization cannot transfer across a rephrasing.

This makes the two results complementary rather than contradictory, and it yields a testable
prediction: **paraphrase transfer should be a function of the model's underlying capability,
switching on somewhere between our scale and theirs.** We now present this as the reconciliation,
and we think it is a more useful contribution than either result alone.

(One small correction: Jiang et al. (2024) is already cited, in the introduction and at paragraph
length in the appendix related work. We note it only because it bears on the Originality
assessment.)

**Q4 — how are the values in Table 1 calculated?**

Each contaminated checkpoint is evaluated on the rephrased and numerically perturbed MATH test sets
under greedy decoding, 0-shot, and a response counts as correct only if it contains a well-formed
`\boxed{}` answer that Math Verify accepts. The revision reports this across 39 checkpoints and adds
an **Original** column measured the same way, so the comparison no longer requires reading a
baseline off Figure 1. Averaged over the 14 contaminated checkpoints with R ≥ 100:

| Condition | Math Verify | Advantage removed |
|---|---|---|
| Original | 72.18% | — |
| Rephrased | 2.78% | 96.1% |
| Perturbed | 1.91% | 97.4% |

The perturbed column excludes 582 problems (11.64%) whose numerical perturbation leaves the
ground-truth answer unchanged, since those score a memorizing model correct by construction; we
report the exclusion and note that including them gives 4.84%. We also no longer describe
performance as collapsing "to baseline": the uncontaminated floor is 0.00%, and the modified
conditions sit 1.9–2.8 percentage points above it.

---

## Reviewer 1wx9

**W1/Q1 — contamination that is paraphrased, partial, or synthetic rather than exact.**

We agree this was the most important missing condition, and we ran it. We modified the pretraining
pipeline so the *injected* text can differ from the text loss is measured on, and pretrained new
models at R = 32, 100, 316 while measuring cross-entropy on the original test set throughout.

This proved more informative than a single paraphrased arm. Our rephrased MATH set rephrases the
problem but keeps the solution (4,991 of 5,000 solutions are byte-identical), so injecting it is
*solution-verbatim* leakage, while our perturbed set shares nothing verbatim. That gives a three-arm
ablation over **which component of a leaked document carries the effect**:

| Arm | Problem | Solution | Loss @ R=316 | Accuracy @ R=316 |
|---|---|---|---|---|
| Uncontaminated | — | — | 7.1437 | 0.00% |
| Exact replicas | same | same | 0.5243 | **7.22%** |
| Rephrased | differs | **same** | 1.9573 | 1.52% |
| Perturbed | differs | differs | 3.3705 | 1.60% |

Two results follow. **Only exact-replica contamination produces a dose-response in accuracy** — it
climbs 13× from R = 32 to R = 316, while both arms whose problem text differs plateau near 1.5% and
stay there. But **loss calls the perturbed model 57–88% as contaminated as the exact one.** The two
metrics disagree about the same models, and accuracy is the one tracking what a benchmark actually
reports — the clearest argument we can offer that contamination work should not rely on loss alone.

The rephrased arm supplies the mechanism. It memorizes just as strongly as the exact arm when
measured on its own training items (7.56% vs 7.22%), yet scores 1.52% on the originals and never
reproduces a gold solution there. **Memorization is of the solution text; retrieval is keyed on the
problem text** — which is also why Table 1's rephrased *evaluation* collapses. One mechanism, both
directions.

Two limits we state in the paper: our perturbed set is 21.7% smaller per copy, so at fixed R it
delivers a smaller dose than its label implies (the bias runs against us); and both modified corpora
are still MATH-domain text, so cleanly separating item-level leakage from domain adaptation would
need a fourth arm contaminated with *disjoint* mathematics, which we have not run.

**W2 — "if we have rephrased contamination in training, evaluation looks uncontaminated. Which is
quite surprising."**

Thank you — this identified a genuine ambiguity in our presentation. Table 1 does **not** test that
direction; it tests exact contamination in training with *modified* evaluation. The direction you
describe is the experiment above, and it was untested in the submission. We have clarified the
caption so the inference is not invited.

Your accompanying hypothesis — that model or training scale may be insufficient to see
generalization from contaminated data — is, we believe, correct, and we can now support it directly:
an uncontaminated 344M model produces 0 correct answers in 5,000,000 samples. There is no latent
ability for contamination to combine with, so anything breaking verbatim surface-form match removes
the entire effect. We use this to reconcile Finding 2 with prior work reporting that rephrased
contamination *does* transfer — in already-capable models, a different regime rather than a
contradictory result.

**W3 — small models, single benchmark and mixture.** Conceded; raised by all three reviewers. We
retain the deliberate scale-for-control tradeoff, state the limitation more prominently, and commit
to a second benchmark and a second model family for the camera-ready.

**W4 — exact leakage makes the loss results less surprising.** Agreed, and we now say so directly.
We would note that the irreducible-error result is not in that category: the surprise is not that
loss drops, but that a *single* replica pushes measured loss below the extrapolated uncontaminated
asymptote. Nor are the inference-time regimes, which concern how memorized content is emitted rather
than how well it is stored.

---

## Reviewer aPBL

**W1/W2 — small models, single dataset.** Conceded without argument. We keep the scale-for-control
tradeoff and the scaling-law bridge, state the limitation more prominently, and commit to a second
benchmark and model family. One observation that emerged from this rebuttal: the *reason* our models
show no paraphrase transfer is that they have zero baseline capability, which makes scale a
**substantive boundary condition** on Finding 2 rather than a caveat, and yields a testable
prediction that transfer switches on with capability.

**W3 — single seed, no error bars.** Added. We report exact binomial 95% intervals over the 5,000
test problems for every Math Verify number: median half-width **0.12 pp**, maximum 1.35 pp, against
effects spanning 0% to 100%. We state explicitly that this quantifies test-set sampling error and
**not** seed-to-seed variance, and we commit to multiple seeds at the R ≈ 10–100 transition — where
variance should matter most — for the camera-ready.

**W4 — missing SFT hyperparameters.** You are right that they were absent. A new appendix gives the
full configuration: AdamW, learning rate 1×10⁻⁴, cosine schedule with 0.2 warmup ratio, one epoch,
effective batch size 64, maximum sequence length 2048, bfloat16, trained on the MATH *train* split
with best-on-validation checkpoint selection. All 39 runs share these; only the initial pretrained
checkpoint varies.

**Q1 — multiple replicates seem contrived; are there more realistic settings?**

Quantifying the dose helps. As a fraction of the tokens actually trained on, at **R = 1** the test
set is **0.03%–0.30%** depending on model size — at or below published real-world leakage estimates,
and we measure effects there. At the top of our ladder it reaches 74%, which is deliberately
extreme. The ladder is best described as a **dose-response curve spanning below-realistic to
saturating**, and we now describe it that way rather than implying every dose is realistic.

On the realism of the leakage *mode* rather than its size, the three-arm ablation reported to 1wx9
is our direct answer: contamination that is paraphrased, or that shares no verbatim text at all,
produces no dose-response in accuracy. Realistic non-verbatim leakage appears to inflate loss
substantially while barely inflating benchmark scores.

**Q2 — how were the modified sets validated, and are difficulty and length distributions matched?**

A new appendix covers this. Difficulty is matched exactly by construction: both sets are
index-aligned rewrites of the original problems, so the Level 1–5 distribution is identical across
all three sets. Problem lengths are close (median 47 → 45 and 46 tokens); rephrased solutions are
unchanged, while perturbed solutions are shorter (mean 162.8 vs 215.3 tokens), which we report
because it means a perturbed replica delivers ~22% fewer tokens at the same nominal dose. Validation
combined automated checks — answer-to-`\boxed{}` consistency, detection of unmodified problems and
of generation failures — with manual inspection for faithfulness and problem-solution consistency.
An earlier pair of candidate datasets failed this audit and was regenerated.

The appendix also reports something your question was aimed at: **11.64% of perturbed problems (582
of 5,000) have a numerical perturbation that leaves the ground-truth answer unchanged**, and so
score a memorizing model correct by construction. We exclude them, report the exclusion, and note
that including them raises the perturbed score from 1.91% to 4.84%.

**Q3 — does the irreducible error depend on the functional form and on extrapolation?**

This is the right question to ask, and the claim's structure makes it more robust than it may
appear: the contaminated losses are **measured**, not fitted. Only the uncontaminated asymptote E(0)
is extrapolated. The claim therefore requires only a *conservative lower bound* on E(0) that still
exceeds the measured contaminated losses — not a correct functional form.

We now report that bound. Bootstrap refitting gives E(0) = **3.5942**, 95% interval **[3.5359,
3.6639]**, and **33 of 35 contaminated runs (94.3%) have measured loss below the lower end of that
interval**; the two exceptions are the smallest model at R = 1 and R = 3. The appendix reports the
fit's limitations — the intervals are optimistically narrow, and the R = 32 and R = 1000 fits are
flagged unreliable — so we quote the conclusion rather than the interval widths.

**Q4 — how would you set up cross-domain experiments?**

The natural extension, and we now specify the design rather than gesturing at it: pretrain with MATH
contamination as here, then evaluate on held-out tasks that share competence but not items — GSM8K,
a mathematics subset of MMLU, and a code benchmark requiring arithmetic reasoning — measuring
whether the contamination advantage transfers across *domain* even where it does not transfer across
*surface form*. Our capability result predicts it will not at this scale, which makes it a clean
test of the capability-boundary hypothesis above once run at larger scale.

---

## Posting checklist

- [ ] Confirm the OpenReview per-comment character limit. Sections are 2.9–4.4k characters.
- [ ] Do not use the "~60× SFT collapse" figure — an artifact of comparing 0-shot pretrained against
      4-shot SFT.
- [x] No protocol archaeology, provenance narration, or scorer methodology in the response.
      Corrections appear once, in a clause.
- [x] Five citations added and `\citep`'d; bibliography verified (128 entries, 0 undefined).
- [x] SFT and modified-test-set appendices written; manuscript compiles (47 pages).

### Numbers to use

| Quantity | **Use** |
|---|---|
| Table 1 (R ≥ 100) | **72.18 → 2.78 / 1.91%** |
| Uncontaminated floor | **0.00%** |
| SFT | **72.95 → 2.80%** (14 conditions) |
| Finding #4 retention | **0.0188 / 0.9966** |
| Temperature at τ = 1.0 | **25%** |
| Answer-overlap inflation | **4.84%** |
| Contaminated fraction at R = 1 | **0.03–0.30%** |
| CI median half-width | **0.12 pp** (exact binomial, strict scores) |

`notebooks/11_*/results/protocol_sensitivity_rescored.csv` (`strict_score`) is authoritative.
