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

## The caveat that survives even the perturbed arm

`math_perturbed` is still MATH-domain text with MATH-style solutions. Part of any loss reduction
in that arm is **domain adaptation**, not item-level leakage. The R = 0 baseline saw no
mathematics at all, so it does not separate the two.

A clean separation needs a fourth arm contaminated with *disjoint* math problems — same domain,
no item correspondence. That is not run and should not be claimed. State the perturbed number as
an upper bound on realistic-leakage transfer.

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
