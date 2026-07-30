# Contaminated Fraction of the Pretraining Budget

One copy of the MATH test set, tokenized as injected, is **1,441,312 tokens**.
Training budget is `20 x overtrain_multiplier x num_parameters` (Chinchilla-optimal at
`ot = 1`), so the same replica count occupies a very different share of a 34M budget
than of a 344M budget.

> ⚠️ **These percentages use the NOMINAL budget and are therefore understated (added
> 2026-07-30).** Runs do not consume the full `20 x N` target: the corpus-trimming step keeps
> documents while the cumulative token count stays *below* the target, so the actual budget is
> about **75%** of nominal. Measured on the 34M `ot = 1` runs, whose
> `train/num_input_tokens_seen` survives in the notebook-11 cache:
>
> | R | nominal % | **actual %** |
> |---|---|---|
> | 1 | 0.21 | **0.30** |
> | 3 | 0.64 | **0.89** |
> | 10 | 2.12 | **2.94** |
> | 32 | 6.77 | **9.24** |
> | 100 | 21.16 | **27.33** |
> | 316 | 66.86 | **73.80** |
>
> Multiply the table below by roughly **1.33** for actual shares. The qualitative claim is
> unchanged — R = 1 is still at or below published real-world leakage estimates, and the top of
> the ladder is still deliberately extreme — but quote the corrected figures.

Only configurations that exist as checkpoints on the Hub are shown. Combinations that
would exceed the token budget were rejected at dataset-construction time and were never
trained, which is why the replica ladders are ragged and why a given model reaches higher
replica counts only at higher overtraining multipliers.

## Percent of training tokens that are MATH test set, at `ot = 1`

|   Num. Replicas |    34M |    62M |    93M |   153M |   344M |
|----------------:|-------:|-------:|-------:|-------:|-------:|
|               0 |   0.00 |   0.00 |   0.00 |   0.00 |   0.00 |
|               1 |   0.21 |   0.12 |   0.08 |   0.05 |   0.02 |
|               3 |   0.64 |   0.35 |   0.23 |   0.14 |   0.06 |
|              10 |   2.12 |   1.16 |   0.77 |   0.47 |   0.21 |
|              32 |   6.78 |   3.72 |   2.48 |   1.51 |   0.67 |
|             100 |  21.20 |  11.62 |   7.75 |   4.71 |   2.09 |
|             316 |  66.98 |  36.73 |  24.49 |  14.88 |   6.62 |
|            1000 | nan    | nan    |  77.49 |  47.10 |  20.95 |
|            3162 | nan    | nan    | nan    | nan    |  66.24 |

## Largest contaminated fraction reached at each (size, overtrain multiplier)

|   Overtrain Multiplier |   34M |   62M |   93M |   153M |   344M |
|-----------------------:|------:|------:|------:|-------:|-------:|
|                   1.00 | 66.98 | 36.73 | 77.49 |  47.10 |  66.24 |
|                   2.00 | 33.49 | 58.12 | 38.74 |   2.36 |  33.12 |
|                   4.00 | 52.99 | 91.88 | 61.26 |   1.18 |  16.56 |
|                   8.00 | 83.78 | 45.94 | 30.63 | nan    |   8.28 |
|                  16.00 | 41.89 | 22.97 | 15.31 | nan    |   0.04 |

## How to use this in the rebuttal

- **Low replica counts bracket realistic leakage from below.** At `R = 1` the test set is
  a fraction of a percent of the budget for every model size — comparable to or below
  published estimates of real-world benchmark leakage, and the paper measures effects there.
- **High replica counts are deliberately extreme, and should be described that way.** At
  the top of the ladder the injected replicas are a large share of the smaller models'
  budgets. That is a feature — it upper-bounds the effect — but it is not 'realistic
  leakage' and claiming so invites exactly aPBL's objection.
- The honest framing is a **dose-response curve spanning from below-realistic to
  saturating**, with the interesting science in where the transition happens.

## Caveat on `Num. Tokens` for overtrained checkpoints

`src/analyze.py:75` computes pretraining `Num. Tokens` as
`20 * overtrain_multiplier * num_parameters`, which is right. But the **eval-side**
computation in `notebooks/11_*` (`Num. Tokens = 20 * Num. Parameters`) omits the
overtrain multiplier. That is harmless for the `ot = 1` runs it was written for, and
**wrong for the overtrained checkpoints** — it would understate their compute by up to
16x and misplace every point on a FLOP axis. Fix before plotting the overtraining results.
