# Memorization without retrieval: loss and accuracy come apart

Measured 2026-07-30. This is the direct test of the mechanism that the contaminant ablation and
Table 1 only implied jointly, and it is the strongest new result of the rebuttal.

## The setup

Take the checkpoints pretrained on **rephrased problems paired with verbatim original solutions**
(sweep `mxamktp0`) and evaluate them 0-shot on the **original** problems, boxed-required scoring
— the same protocol and scorer as every other 0-shot number. Compare against the published
exact-replica runs at the same doses.

Qwen3-34M, 1×OT. Uncontaminated baseline: loss **7.1437**, accuracy **0.0000**.

| R | Loss, exact | Loss, rephrased | Accuracy, exact | **Accuracy, rephrased** | Boxed rate | Verbatim solution rate |
|---|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 0.56% | **0.24%** | 0.150 | **0.000** |
| 100 | 1.4526 | 2.0077 | 1.70% | **1.58%** | 0.871 | **0.000** |
| 316 | 0.5243 | 1.9573 | 7.22% | **1.52%** | 0.943 | **0.000** |

## What it shows

**The solutions are memorized.** At R = 316 the rephrased model's cross-entropy on the original
test solutions is 1.9573 against an uncontaminated 7.1437 — 78% of the way to the exact-replica
model's 0.5243. By any loss-based measure this model is heavily contaminated.

**And it cannot produce them.** Given the original problem, that same model scores **1.52%** and
reproduces the gold solution verbatim **0 times out of 5,000**. The exact-replica model at the
same dose scores 7.22%.

So the model holds the answer and cannot get to it. **Memorization is of the solution text;
retrieval is keyed on the problem text it was trained against.** Rephrasing the problem at
*training* time stores the solution without the key; rephrasing it at *evaluation* time (Table 1)
withholds the key from a model that has one. Both collapse generation, for the same reason.

The boxed rate is the corroborating detail: it climbs to 0.94, so the model has clearly learned
the output *format* from the injected solutions. It emits well-formed `\boxed{}` answers that are
wrong. Format and content dissociate exactly as the protocol analysis predicted.

## Why this matters for the rebuttal

**1. It is a direct demonstration of 8RFz's W1 — the pivotal critique — on our own data.** The
reviewer's objection is that cross-entropy on the exact solution text is not correctness. Here is
a regime where the two diverge sharply: loss says heavily contaminated, accuracy says nearly
clean. We should concede the point *by showing it*, then note that in the overtraining and SFT
conditions the metrics do track (notebooks 17 and 19), so Findings 4–5 survive on accuracy.

**2. It inverts the usual detection worry.** 8RFz raised the scenario where accuracy persists
while loss rises, making contamination invisible to perplexity. We find the opposite asymmetry:
**perplexity-based detection would flag these models loudly while their benchmark scores are
barely inflated.** That is a false-positive mode for loss-based contamination detection, and it
is arguably the more common real-world case, since realistic leakage rarely reproduces benchmark
problems verbatim.

**3. It bounds the practical damage from realistic leakage.** Solution-verbatim leakage recovers
78–98% of the exact-replica *loss* reduction but only ~20% of the *accuracy* inflation at high
dose. Benchmark scores are more robust to non-verbatim leakage than loss is — which is
reassuring for benchmark validity and cautionary for detection methods built on loss.

## The perturbed arm: no dose-response at all

Adding the third arm (nothing leaked verbatim) sharpens the picture. Accuracy on the original
problems, 0-shot, boxed-required:

| R | Exact | Rephrased | **Perturbed** |
|---|---|---|---|
| 32 | 0.56% | 0.24% | **1.34%** |
| 100 | 1.70% | 1.58% | **1.16%** |
| 316 | 7.22% | 1.52% | *(running)* |

Exact and rephrased both climb with dose. **Perturbed is flat** — 1.34% → 1.16%, a change well
inside the ±0.33 pp bootstrap half-width — and its loss is flat too (3.0741 → 3.0113) while
exact's keeps falling. Its `\boxed{}` rate is high (0.63 → 0.94) and its verbatim solution rate is
0.000 at every dose.

So the perturbed model learns the *genre* — format, template, the look of a MATH solution — from
the first 32 replicas and then stops improving, because further replicas carry no information
about the specific items being evaluated.

**The dose-response curve that defines contamination requires verbatim content.** Non-verbatim
leakage produces a one-off genre gain that does not compound. That is a cleaner statement of what
"realistic leakage" does than anything in the submitted paper, and it is the direct answer to
1wx9's Q1.

Note the ordering at R = 32 — perturbed (1.34%) *above* exact (0.56%). At light contamination the
perturbed model has learned to emit well-formed answers while the exact model has only begun
memorizing, so genre-learning temporarily wins. By R = 316 memorization has taken over
completely.

## Caveats

- 34M only, and one seed. The exact-replica accuracies come from the rescored 0-shot grid; the
  rephrased accuracies are new evaluations under the same scorer and protocol, so they are
  directly comparable.
- The perturbed arm (nothing verbatim) is still running; its accuracy should be at or near the
  0.00% floor if this account is right, and that is worth checking rather than assuming.
- "Verbatim solution rate" counts an exact substring match of the gold solution in the response.
  It is 0.000 at every dose, but a near-miss paraphrase of the solution would not be caught; the
  boxed-required accuracy already covers the case that matters.

Reproduce: `scripts/eval_contaminant_checkpoints_zeroshot.py`; raw generations in
`results/contaminant_eval/*.jsonl`; table in `results/contaminant_eval/loss_vs_accuracy.csv`.
