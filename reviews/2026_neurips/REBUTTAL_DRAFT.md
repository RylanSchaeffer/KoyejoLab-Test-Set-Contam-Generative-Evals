# Rebuttal draft: NeurIPS 2026 submission 32216

Scores: **8RFz 3** (Quality 2, Originality 2, conf 4) · **1wx9 4** (all 3s) · **aPBL 3** (all 3s).

**Where the score movement is.** All three reviewers score Significance 3; nobody disputes the
problem matters, so do not argue that. **8RFz's Quality 2 and Originality 2 are the only sub-"good"
marks in the entire review set**, and both trace to specific, fixable complaints: Quality to their W1
(loss-based evidence does not establish a claim about generation) and Originality to their W3
(framing understates prior work; the replication and the conflict go undiscussed). Those two are the
single biggest lever available. aPBL's subscores are all 3s, so their rating rests on the four
limitations they list: two are now fully resolved (seeds, SFT details) and two conceded (scale,
benchmark). 1wx9 is at 4 with all 3s and asked for exactly one experiment, which ran and produced the
strongest result here.

**The AC published their checklist early.** The metareview lists five weaknesses under (c). The
general response therefore opens with a five-row table mapping each metareview weakness, in the
AC's own order, to its resolution, so the AC can verify the whole response in one pass. The
retrieval mechanism is the synthesis that follows the table, not the organizing principle.

Each per-reviewer section closes by tallying what they raised against what is now done, and asking.
Reviewers do not raise scores for work they have to reconstruct themselves.

**Writing rule.** Every paragraph presents a result or answers a question. No process narration, no
provenance archaeology, no account of how we found our own errors. Corrections appear once, in a
clause, and we move on. Supporting detail lives in [`REBUTTAL_EVIDENCE.md`](REBUTTAL_EVIDENCE.md)
and in the paper. A reviewer dragged through our measurement hygiene is a reviewer wondering why we
are wasting their time. No em dashes anywhere in the posted text.

---

## General response

We thank the reviewers and the AC. Two new experiments, run in response to these reviews, address
the central criticisms. Against the metareview's five weaknesses, in order:

| Metareview weakness | Resolution |
|---|---|
| Exact replicas are not realistic leakage | **New experiment**: three-arm pretraining ablation with paraphrased and perturbed contamination (details in reply to 1wx9) |
| ≤350M params, one family, one benchmark | **Conceded**; second benchmark and model family committed for the camera-ready, and scale reframed as a substantive boundary condition (below) |
| Single seed, no error bars | **Added**: exact binomial 95% CIs on every Math Verify number; median half-width 0.12 pp against effects spanning 0–100% |
| Findings 4–5 rest on loss, not generation correctness | **New measurement**: all 137 overtraining and 39 SFT checkpoints evaluated under Math Verify (details in reply to 8RFz) |
| Related work understated; replication and conflict undiscussed | **Fixed**: five references added, Finding 1 marked as a replication, the conflict reconciled (reply to 8RFz) |

**One mechanism unifies the new results.** Contamination in generative evaluation turns out to be a
*retrieval* phenomenon, not a learning one: models store leaked solutions and emit them only when
the exact problem text cues retrieval. That one mechanism accounts for why scores inflate to
near-100%, why they collapse under any rephrasing of the problem in either training or evaluation,
and why loss and accuracy dissociate.

**Findings 4 and 5, now measured in accuracy** (the metareview's pivotal point; 8RFz W1/Q1).
Accuracy tracks loss, so the possibility 8RFz raises (accuracy persisting while loss rises, hiding
contamination from perplexity while still inflating scores) does not occur here. The measurement
also sharpens Finding 4: dilution is **threshold-dependent** (advantage retained 0.0188 at R = 100
but 0.9966 at R = 1000, for 93M over ot 1→16), so overtraining is not a mitigation for the heavy
leakage regimes that matter most, and we now say so.

**Contamination that is paraphrased rather than exact** (1wx9 Q1; aPBL Q1). **Only exact replicas
produce a dose-response in accuracy**, climbing from 0.56% at R = 32 to 7.22% at R = 316, while
both arms whose problem text differs plateau near 1.5%. The rephrased arm (rephrased problems,
verbatim solutions) is 78% of the way from clean to fully-contaminated by loss, yet scores 1.52%,
and never reproduces a gold solution; on its *own* training items it scores 7.56% and regurgitates
verbatim 5.34% of the time. It has memorized the solutions and cannot reach them.

**Scale as a boundary condition.** An uncontaminated 344M model produces 0 correct answers in
5,000,000 samples, so at our scale every point of contaminated performance is memorization, and
nothing that breaks verbatim surface-form match survives. Prior work finding that rephrased
contamination transfers used already-capable models; our reconciliation predicts paraphrase
transfer switches on with capability: a testable boundary condition, not a contradiction.

**In practice:** detection built on loss or perplexity has a false-positive mode: it flags
heavily-memorizing models whose benchmark scores are barely inflated. Benchmark scores are more
robust to realistic non-verbatim leakage than loss is. And overtraining on fresh data does
essentially nothing once leakage is heavy.

We are grateful for reviews specific enough to be actionable. Every criticism we could address with
data, we did, and the paper is stronger for it.

---

## Reviewer 8RFz

**W1/Q1: do Findings 4 and 5 hold for Math Verify score?**

Yes. We evaluated all 137 overtraining and 39 SFT checkpoints in accuracy space, not inferred
from Figure 11's correlation. We looked specifically for the sharper possibility you raise,
accuracy persisting while loss rises, and it does not occur.

The measurement improves Finding 4. Dilution is **threshold-dependent**: at 93M over ot 1→16, the
advantage retained is **0.0188 at R = 100** and **0.9966 at R = 1000**. The mechanism is dilution
of the contaminated *token fraction*, which stops working once that fraction stays high; the
revision replaces our unqualified claim with this threshold behaviour, since overtraining is not a
mitigation for heavy leakage. For Finding 5, SFT takes mean Math Verify from **72.95% to 2.80%**
across the 14 conditions scoring ≥ 5% beforehand.

**We also found a regime where your objection is exactly right.** Pretraining on rephrased problems
paired with verbatim solutions, then evaluating on the original problems (34M, R = 316):

| | Loss | Accuracy | Reproduces gold solution |
|---|---|---|---|
| Uncontaminated | 7.1437 | 0.00% | n/a |
| Exact replicas | 0.5243 | 7.22% | frequently |
| Rephrased problems, verbatim solutions | 1.9573 | **1.52%** | **0 / 5,000** |

By any loss-based measure that model is heavily contaminated, yet it scores 1.52% and never
reproduces a gold solution. On its *own* training items it scores 7.56% with a 5.34% verbatim
rate. This is a retrieval failure, not a learning failure: **memorization is of the solution text;
retrieval is keyed on the problem text.** Loss and correctness come apart sharply, as you argued
they could; we now report both metrics throughout. One consequence runs opposite to your concern:
perplexity here flags models whose benchmark scores are barely inflated, a false-positive mode for
loss-based detection and plausibly the common one in practice. (This ablation is at our smallest
scale; our account below predicts the rephrased arm's transfer grows with capability.)

**W2/Q2: does temperature reduce contamination effects specifically, or degrade generally?**

The clean control is the contamination *advantage* at matched temperature, score(R) − score(R = 0)
at the same τ, so uniform degradation cancels:

| τ | 0 | 0.32 | 0.56 | 0.75 | 0.94 | 1.0 | 1.29 |
|---|---|---|---|---|---|---|---|
| Advantage retained | 100% | 98% | 90% | 72% | 39% | **25%** | 0.4% |

By τ = 1.0, the model's own distribution rather than a hot setting, three quarters of the
advantage is gone with degradation controlled for. Above τ ≈ 1.3 the two explanations are no
longer separable, so we restrict the claim to τ ≤ 1.

**W3/Q3: related work, and why Finding 2 conflicts with prior rephrasing results.**

We accept this criticism and have acted on it: we remove the "first targeted examination" framing,
add Palavalli et al. (2024), Mehrbakhsh et al. (2024), Dekoninck et al. (2024a, 2024b) and Godey et
al. (2025), and mark **Finding 1 as replicating** the repeat-count effect. (Jiang et al. (2024) is
already cited, including at paragraph length in the appendix.)

On the conflict: Mehrbakhsh et al. and Dekoninck et al. inject contamination into models **already
capable** of the task, so the advantage sits on top of competence and survives a paraphrase. Our
models have no competence for it to sit on: an uncontaminated 344M model produces **0 correct
answers in 5,000,000 samples**. With no latent ability, all contaminated performance is verbatim
memorization, which cannot transfer across a rephrasing. The results are complementary and yield a
testable prediction: **paraphrase transfer is a function of underlying capability, switching on
between our scale and theirs.**

**Q4: how are the values in Table 1 calculated?**

Each contaminated checkpoint is evaluated on the modified test sets under greedy decoding, 0-shot;
a response counts only if a well-formed \boxed{} answer passes Math Verify. The revision reports
all 39 checkpoints and adds an **Original** column at the same protocol. Averaged over the 14
checkpoints with R ≥ 100: Original **72.18%**, Rephrased **2.78%**, Perturbed **1.91%**, against
an uncontaminated floor of 0.00%. The perturbed column excludes 582 problems (11.64%) whose
perturbation leaves the ground-truth answer unchanged; these score a memorizing model correct by
construction. Including them gives 4.84%.

**In summary.** Your two lowest marks rested on specific objections we have acted on. Quality
rested on W1: we replaced loss-based evidence with direct accuracy measurements across all 176
checkpoints, and where the metrics genuinely diverge we show the divergence rather than assume it
away. Originality rested on W3: we added the missing references, marked Finding 1 as a replication,
and turned the conflict into a capability-boundary account with a testable prediction. We would be
grateful if you would consider whether these changes warrant revisiting those two scores.

---

## Reviewer 1wx9

**W1/Q1: contamination that is paraphrased, partial, or synthetic rather than exact.**

We agree this was the most important missing condition, and we ran it. We modified the pretraining
pipeline so the *injected* text can differ from the text loss is measured on, and pretrained new
models at R = 32, 100, 316, measuring cross-entropy on the original test set throughout.

This proved more informative than a single paraphrased arm. Our rephrased MATH set rephrases the
problem but keeps the solution (4,991 of 5,000 solutions byte-identical), so injecting it is
*solution-verbatim* leakage, while our perturbed set shares nothing verbatim. That gives a
three-arm ablation over **which component of a leaked document carries the effect**:

| Arm | Problem | Solution | Loss @ R=316 | Accuracy @ R=316 |
|---|---|---|---|---|
| Uncontaminated | n/a | n/a | 7.1437 | 0.00% |
| Exact replicas | same | same | 0.5243 | **7.22%** |
| Rephrased | differs | **same** | 1.9573 | 1.52% |
| Perturbed | differs | differs | 3.3705 | 1.60% |

Two results follow. **Only exact-replica contamination produces a dose-response in accuracy**,
rising 13× from R = 32 to 316, while both arms whose problem text differs plateau near 1.5%. But
**loss calls the perturbed model 57–88% as contaminated as the exact one.** The two metrics
disagree about the same models, and accuracy is the one tracking what a benchmark reports, which is
the clearest argument we can offer that contamination work should not rely on loss alone.

The rephrased arm supplies the mechanism. It memorizes as strongly as the exact arm on its own
training items (7.56% vs 7.22%), yet scores 1.52% on the originals and never reproduces a gold
solution there. **Memorization is of the solution text; retrieval is keyed on the problem text**,
which is also why Table 1's rephrased *evaluation* collapses. One mechanism, both directions.

Three limits we state in the paper: our perturbed set is 21.7% smaller per copy, so at fixed R it
delivers a smaller dose than its label implies (the bias runs against us); both modified corpora
are still MATH-domain text, so separating item-level leakage from domain adaptation would need a
fourth arm with *disjoint* mathematics, which we have not run; and the ablation is at our smallest
model size, where by the capability account below the rephrased arm's transfer is precisely what
should grow with scale, which is why we present Finding 2 as a boundary rather than a universal.

**W2: "if we have rephrased contamination in training, evaluation looks uncontaminated. Which is
quite surprising."**

Thank you; this identified a genuine ambiguity in our presentation. Table 1 does **not** test that
direction; it tests exact contamination in training with *modified* evaluation. The direction you
describe is the experiment above, untested in the submission. We have clarified the caption so the
inference is not invited.

We believe your accompanying hypothesis (that model or training scale may be insufficient to see
generalization from contaminated data) is correct, and we can now support it directly: an
uncontaminated 344M model produces 0 correct answers in 5,000,000 samples. There is no latent
ability for contamination to combine with, so anything breaking verbatim surface-form match removes
the entire effect. This reconciles Finding 2 with prior work reporting that rephrased contamination
*does* transfer: in already-capable models, a different regime rather than a contradictory result.

**W3: small models, single benchmark and mixture.** Conceded; raised by all three reviewers. We
retain the deliberate scale-for-control tradeoff, state the limitation more prominently, and commit
to a second benchmark and a second model family for the camera-ready.

**W4: exact leakage makes the loss results less surprising.** Agreed, and we now say so directly.
The irreducible-error result is not in that category: the surprise is not that loss drops, but that
a *single* replica pushes measured loss below the extrapolated uncontaminated asymptote. Nor are
the inference-time regimes, which concern how memorized content is emitted rather than how well it
is stored.

**In summary.** You asked for one thing, contamination that is paraphrased rather than exact, and
identified it as what would determine whether our conclusions extend beyond blatant leakage. We ran
it, and it produced what we think is the strongest result in the paper: the retrieval-key
mechanism, which unifies Finding 2 with the new ablation and explains the loss-versus-accuracy gap
the metareview highlighted. It also answers your W4, by showing which results survive when leakage
is not verbatim. If that addresses the concern behind your score, we would be glad if you would
consider whether the paper now merits a stronger one.

---

## Reviewer aPBL

**W1/W2: small models, single dataset.** Conceded without argument. We keep the scale-for-control
tradeoff and the scaling-law bridge, state the limitation more prominently, and commit to a second
benchmark and model family for the camera-ready. One new observation: the *reason* our models show
no paraphrase transfer is that they have zero baseline capability, which makes scale a
**substantive boundary condition** on Finding 2 rather than a caveat, and yields a testable
prediction that transfer switches on with capability.

**W3: single seed, no error bars.** Added. We report exact binomial 95% intervals over the 5,000
test problems for every Math Verify number: median half-width **0.12 pp**, maximum 1.35 pp, against
effects spanning 0–100%. These quantify test-set sampling error, **not** seed-to-seed variance; we
commit to multiple seeds at the R ≈ 10–100 transition, where variance should matter most, for the
camera-ready.

**W4: missing SFT hyperparameters.** They were absent; a new appendix gives the full
configuration: AdamW, learning rate 1×10⁻⁴, cosine schedule with 0.2 warmup ratio, one epoch,
effective batch size 64, maximum sequence length 2048, bfloat16, trained on the MATH *train* split
with best-on-validation checkpoint selection. All 39 runs share these; only the initial pretrained
checkpoint varies.

**Q1: multiple replicates seem contrived; are there more realistic settings?**

Quantifying the dose helps. As a fraction of the tokens actually trained on, at **R = 1** the test
set is **0.03%–0.30%** depending on model size, at or below published real-world leakage
estimates, and we measure effects there. The top of the ladder reaches 74%, deliberately extreme.
The ladder is a **dose-response curve spanning below-realistic to saturating**, and we now
describe it that way. On the realism of the leakage *mode*, the three-arm ablation in our reply to
1wx9 answers directly: paraphrased contamination, or contamination sharing no verbatim text,
produces no dose-response in accuracy; realistic non-verbatim leakage inflates loss substantially
while barely inflating benchmark scores.

**Q2: how were the modified sets validated; are difficulty and length matched?**

A new appendix covers this. Both sets are index-aligned rewrites, so the Level 1–5 difficulty
distribution is identical by construction. Problem lengths are close (median 47 vs 45 and 46
tokens); rephrased solutions are unchanged; perturbed solutions are shorter (mean 162.8 vs 215.3
tokens), so a perturbed replica delivers ~22% fewer tokens at the same nominal dose (the bias runs
against us). Validation combined automated checks (answer-to-\boxed{} consistency, detection of
unmodified problems and of generation failures) with manual inspection; an earlier candidate pair
failed this audit and was regenerated.

The audit also caught a real defect: **11.64% of perturbed problems (582 of 5,000) have a
numerical perturbation that leaves the ground-truth answer unchanged**, and so score a memorizing
model correct by construction. We exclude and report them; including them raises the perturbed
score from 1.91% to 4.84%.

**Q3: does the irreducible error depend on the functional form and on extrapolation?**

The right question; the claim is more robust than it may appear: the contaminated losses are
**measured**, not fitted. Only the uncontaminated asymptote E(0) is extrapolated, so the claim
requires only a *conservative lower bound* on E(0) that still exceeds the measured contaminated
losses, not a correct functional form. Bootstrap refitting gives E(0) = **3.5942**, 95% interval
**[3.5359, 3.6639]**, and **33 of 35 contaminated runs (94.3%) have measured loss below the lower
end of that interval**; the exceptions are the smallest model at R = 1 and 3. The appendix reports
the fit's limitations, so we quote the conclusion rather than the interval widths.

**Q4: how would you set up cross-domain experiments?**

Pretrain with MATH contamination as here, then evaluate held-out tasks that share competence but
not items (GSM8K, MMLU mathematics, a code benchmark requiring arithmetic), asking whether the
advantage transfers across *domain* where it does not transfer across *surface form*. Our
capability result predicts it will not at this scale, a clean test of the capability-boundary
hypothesis once run larger.

**In summary.** Of your four limitations, two are fully resolved: every Math Verify number carries
a confidence interval, and the SFT configuration is documented in a new appendix. Your Q2 proved
well-aimed: auditing the modified sets turned up a real defect, the 11.64% of answer-unchanged
perturbations, now excluded and reported. Scale and the single benchmark we concede and commit to
addressing for the camera-ready; the zero-capability result makes scale a substantive boundary
condition on Finding 2 rather than a caveat. We hope the resolved items shift the balance enough
for you to reconsider your score.

---

## Posting checklist

- [x] **OpenReview limit confirmed: 5,000 characters per box.** All four sections fit (re-measure
      after any edit; the splitter below counts every section body).
      `python3 -c "import re;[print(len(p.strip()),p.strip().split(chr(10))[0][:40]) for p in re.split(r'\n---\n',open('reviews/2026_neurips/REBUTTAL_DRAFT.md').read())]"`
- [x] No em dashes anywhere in the posted sections (verify: `grep -c $'\\u2014' REBUTTAL_DRAFT.md`
      should return 0).
- [ ] Do not use the "~60× SFT collapse" figure, an artifact of comparing 0-shot pretrained against
      4-shot SFT.
- [x] General response opens with the metareview-mapping table, in the AC's own weakness order.
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
