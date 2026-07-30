# What "paraphrased contamination" actually leaks

Established 2026-07-30 while interpreting the first paraphrased pretraining result. **Read this
before quoting any number from the paraphrased arm.**

## The problem

The first result looked spectacular and was nearly written up as such:

| | Benchmark cross-entropy on the *original* test set, 34M, R=32 |
|---|---|
| Uncontaminated (R = 0) | 7.1437 |
| **Paraphrased contaminant** | **2.6125** |
| Exact replicas (published) | 2.5138 |

That is **97.9%** of the exact-replica loss reduction, which would say paraphrased leakage is
almost as damaging as verbatim leakage. It is not what it appears.

## Why: `math_rephrased` rephrases the problem, not the solution

Measured directly against `EleutherAI/minerva_math` test, index-aligned:

| Dataset | Problems identical to original | **Solutions identical to original** |
|---|---|---|
| `RylanSchaeffer/math_rephrased` | 0 / 5000 (0.0%) | **4991 / 5000 (99.8%)** |
| `RylanSchaeffer/math_perturbed` | 0 / 5000 (0.0%) | **4 / 5000 (0.1%)** |

`eval_after/eval_benchmark_loss` is cross-entropy on the original **solution** text. The
rephrased arm injects documents whose solution field is byte-identical to the thing the loss is
measured on. So it is a **solution-verbatim** leakage condition, not a paraphrase condition, and
its transfer number is largely explained by that.

Reporting it as "paraphrased contamination transfers 98% of the effect" would be wrong, and a
reviewer who spent five minutes with the dataset would find it. Reviewer 1wx9 asked about
contamination that is "paraphrased, partial, translated, synthetic" — meaning the *leaked content
itself* differs.

## What we do instead: a three-arm ablation

This is a better experiment than the one originally planned, because it isolates *which component
of a leaked document* carries the effect.

| Arm | Problem | Solution | W&B sweep | What it isolates |
|---|---|---|---|---|
| **Exact** | same | same | published (`-pt`, lost; losses in the notebook-11 cache) | full verbatim leakage |
| **Rephrased** | differs | **same** | `mxamktp0` | solution-only leakage |
| **Perturbed** | differs | **differs** | `vrxwx4dz` | no verbatim leakage at all |

- **Exact vs Rephrased** answers: does the *problem* text matter? First data point at R=32 says
  almost not at all — 2.5138 → 2.6125, i.e. 97.9% of the effect survives losing the problem.
- **Rephrased vs Perturbed** answers: does the *solution* text matter? This is the load-bearing
  comparison and it is what the perturbed arm was launched for.
- **Perturbed vs Uncontaminated** is the realistic-leakage number reviewers asked for, and it
  still carries a caveat (below).

This maps directly onto Jiang et al. (2024), who distinguish "text-only" from "ground-truth"
contamination and find the ground-truth condition far more damaging. Our exact-vs-rephrased
contrast is the same distinction reached from the other direction, and citing it converts an
overlooked reference into a replication.

## Final results — all three arms, all three doses

Qwen3-34M, 1×OT. Loss is cross-entropy on the **original** test set; accuracy is 0-shot greedy,
boxed-required, also on the original problems. Uncontaminated: loss **7.1437**, accuracy **0.00%**.

| R | Loss: exact | Loss: rephrased | Loss: perturbed | Acc: exact | Acc: rephrased | Acc: perturbed |
|---|---|---|---|---|---|---|
| 32 | 2.5138 | 2.6125 | 3.0741 | 0.56% | 0.24% | 1.34% |
| 100 | 1.4526 | 2.0077 | 3.0113 | 1.70% | 1.58% | 1.16% |
| 316 | **0.5243** | 1.9573 | 3.3705 | **7.22%** | 1.52% | 1.60% |

Loss transfer (share of the exact-replica reduction achieved): rephrased 0.979 / 0.902 / 0.784;
perturbed 0.879 / 0.726 / 0.570.

**Only exact-replica contamination produces a dose-response in accuracy.** Exact climbs 0.56 →
1.70 → 7.22% (13×). Both arms whose *problem text* differs from the benchmark plateau at ~1.5%
and stay there: rephrased 0.24 → 1.58 → 1.52%, perturbed 1.34 → 1.16 → 1.60%. All the plateau
movement is inside the ±0.33 pp bootstrap half-width.

**The perturbed arm never improves with dose in loss either** — 3.0741 → 3.0113 → 3.3705, flat
and then slightly worse, while exact falls monotonically. That is the signature of domain
adaptation: once MATH style and problem templates are learned from 32 replicas, further replicas
of *different* items add nothing, and at high dose they displace corpus text that was doing useful
work.

So the perturbed plateau (≈ 3.0 nats) estimates how much loss reduction is available from genre
and template learning alone; the exact arm's further descent to 0.5243 is what requires verbatim
solution text. The confound can be bounded from the data rather than merely acknowledged.

## ⚠️ Replicas are not an equal dose across arms

One copy of each contaminant, tokenized exactly as injected (values corrected 2026-07-30 — the
figures first printed here were each 5,000 tokens low because they omitted the per-document EOS;
these ones divide exactly into the runs' own logged contaminant token counts, see
`verification/PERTURBED_INJECTION_VERIFICATION.md`):

| Contaminant | Tokens per copy | vs original |
|---|---|---|
| `EleutherAI/minerva_math` (original) | 1,446,312 | 100.0% |
| `RylanSchaeffer/math_rephrased` | 1,392,475 | 96.3% |
| `RylanSchaeffer/math_perturbed` | **1,132,643** | **78.3%** |

`math_perturbed` is **21.7% smaller in tokens**, so at fixed R the perturbed arm injects
proportionally less contaminated text. In contaminated tokens, perturbed R = 316 corresponds to
exact R ≈ 247.

This also explains a token-budget check that notebook 21 flagged automatically: perturbed R = 316
consumed 589.0M training tokens against the published run's 617.2M (**−4.56%**), because a smaller
contaminant leaves more to make up from the corpus and the corpus-trimming step undershoots. The
rephrased arm is within 0.03–0.81% and is unaffected.

**What this threatens and what it does not.** Within-arm comparisons across dose are unaffected.
Cross-arm comparisons at fixed R *understate* the perturbed arm, which is getting a smaller dose
than its label implies — and since the perturbed arm is the one showing no effect, the bias runs
against our conclusion, making it conservative. The one thing not to over-read is the R = 316 loss
uptick: part of that is the 4.56% token deficit rather than crowding-out.

## The caveat that survives even the perturbed arm

`math_perturbed` is still MATH-domain text with MATH-style solutions, and it perturbs numbers
within the *same* problems, so template-level correspondence remains. Part of any loss reduction
in that arm is **domain adaptation**, not item-level leakage, and the R = 0 baseline saw no
mathematics at all so it does not separate the two.

A clean separation needs a fourth arm contaminated with *disjoint* math problems — same domain, no
item correspondence. That is not run and should not be claimed. The perturbed plateau bounds
*template + genre* learning together and remains an upper bound on what non-verbatim leakage buys.

## Reproduce

```bash
HF_HOME=/lfs/skampere1/0/shared_hf_cache PYTHONPATH=$PWD \
  ./mem_scoring_vs_sampling_env/bin/python -c "
import src.data; from datasets import load_dataset
o = src.data.load_dataset_hendrycks_math()['test']
for name in ['math_rephrased','math_perturbed']:
    d = load_dataset(f'RylanSchaeffer/{name}', split='test')
    same = sum(o[i]['solution'] == d[i]['solution'] for i in range(len(d)))
    print(name, f'{same}/{len(d)} identical solutions')
"
```
