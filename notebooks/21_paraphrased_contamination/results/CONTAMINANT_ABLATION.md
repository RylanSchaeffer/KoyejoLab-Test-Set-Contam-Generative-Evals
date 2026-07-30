# Which part of a leaked document carries the effect?

Qwen3-34M, 1xOT. Cross-entropy is always measured on the **original** `EleutherAI/minerva_math` test set; only the injected contaminant changes. Run from `scripts/pretrain_language_model_v1.py`, which reproduces the published (pre-`934546a`) optimizer configuration, so the published exact-replica runs serve as the control without retraining.

| Arm | Problem | Solution | Isolates |
|---|---|---|---|
| Exact | same | same | full verbatim leakage |
| Rephrased | differs | **same** (99.8% identical) | solution-only leakage |
| Perturbed | differs | differs (0.1% identical) | no verbatim leakage |

⚠️ The rephrased arm is **not** a paraphrase condition — `math_rephrased` keeps the original solution, and the loss is measured on solution text. See `reviews/2026_neurips/CONTAMINANT_ABLATION.md`.

Uncontaminated baseline (R = 0): **7.1437**

| R | Exact | Rephrased | Perturbed | Transfer: rephrased | Transfer: perturbed |
|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 3.0741 | 0.979 | 0.879 |
| 100 | 1.4526 | 2.0077 | 3.0113 | 0.902 | 0.726 |
| 316 | 0.5243 | 1.9573 | — | 0.784 | — |

`Transfer` = (L(R=0) - L_arm) / (L(R=0) - L_exact): the share of the exact-replica loss reduction the arm achieves. 1.0 means as damaging as verbatim leakage; 0.0 means it buys nothing.

## Caveat to state in the paper

Both modified corpora are still MATH-domain text with MATH-style solutions, so part of any reduction is **domain adaptation** rather than item-level leakage. The R = 0 baseline saw no mathematics at all and so does not separate the two. A clean separation needs a fourth arm contaminated with *disjoint* math problems; that is not run here. Treat the perturbed number as an upper bound on realistic-leakage transfer.
