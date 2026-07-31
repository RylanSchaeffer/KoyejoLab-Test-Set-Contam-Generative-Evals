# Bootstrap Confidence Intervals on Math Verify

> ⛔ **SUPERSEDED 2026-07-30. Do not quote this file.** These intervals are computed from the
> **leniently** scored logs, while every number the rebuttal reports is strict-scored, so the point
> estimates here (R=0 reading 0.38-1.26%) contradict the measured 0.00% floor. A percentile
> bootstrap is also degenerate at zero counts.
>
> Use `strict_score_binomial_cis.csv` instead: exact binomial 95% intervals on the strict scores,
> median half-width **0.123 pp**, max **1.350 pp**, zero-scoring conditions bounded above by
> **0.074%**. For a proportion this is equivalent to bootstrapping the per-problem mean, and it is
> well defined when a condition scores zero.

95% percentile bootstrap over the 5001 MATH test problems, 10,000 resamples, greedy decoding, **0-shot** (the protocol behind the manuscript's Finding #1 figure).

> **Scoring caveat (added 2026-07-30).** These intervals are computed from the scores the 0-shot
> sweeps *logged*, which used the lenient scorer (~1.4% false positives) rather than the
> boxed-required scorer used everywhere else — see `PROTOCOL_SENSITIVITY_RESCORED.md`. That shifts
> the *point estimates* at near-zero conditions (a lenient 0.4-1.3% is truly 0.00%) but barely
> moves the *half-widths*, which is what this file is quoted for: a binomial half-width over 5,001
> problems is insensitive to a shift of that size except at the extremes. The claim the rebuttal
> makes from this file — that effect sizes dwarf test-set sampling error — is unaffected. Do not
> quote the per-condition point estimates from here; use the rescored grid.

## What this does and does not cover

These intervals quantify **sampling error over the test set** — how much the score would
move given a different draw of 5,000 problems. They are **not** multi-seed error bars:
they say nothing about variance across pretraining seeds or decoding seeds. State that
explicitly in the rebuttal and commit to seeds for camera-ready rather than letting these
be read as covering that concern.

What they do establish: the intervals are on the order of a percentage point, while the
effects claimed span roughly 1% to 100%. The contamination effect is orders of magnitude
larger than test-set sampling noise.

## Math Verify %, [95% CI]

|   Num. Replicas | 34M               | 62M                  | 93M                   | 153M                    | 344M                    |
|----------------:|:------------------|:---------------------|:----------------------|:------------------------|:------------------------|
|               0 | 0.38 [0.22, 0.56] | 1.26 [0.96, 1.58]    | 0.74 [0.52, 0.98]     | 1.18 [0.88, 1.48]       | nan                     |
|               1 | 0.22 [0.10, 0.36] | 0.88 [0.62, 1.14]    | 1.32 [1.02, 1.64]     | 1.74 [1.38, 2.10]       | 1.30 [1.00, 1.62]       |
|               3 | 0.26 [0.12, 0.40] | 1.48 [1.16, 1.82]    | 1.42 [1.10, 1.76]     | 1.82 [1.46, 2.20]       | 1.54 [1.22, 1.88]       |
|              10 | 0.76 [0.52, 1.00] | 1.66 [1.32, 2.02]    | 1.76 [1.40, 2.14]     | 1.78 [1.42, 2.16]       | 2.18 [1.78, 2.60]       |
|              32 | 1.64 [1.30, 2.00] | 1.78 [1.42, 2.16]    | 2.54 [2.12, 2.98]     | 2.50 [2.08, 2.94]       | 12.72 [11.80, 13.64]    |
|             100 | 1.94 [1.58, 2.32] | 7.40 [6.70, 8.12]    | 37.29 [35.97, 38.63]  | 80.88 [79.78, 81.96]    | 99.12 [98.86, 99.38]    |
|             316 | 7.32 [6.62, 8.04] | 79.90 [78.78, 81.00] | 98.72 [98.40, 99.02]  | 100.00 [100.00, 100.00] | nan                     |
|            1000 | nan               | nan                  | 99.94 [99.86, 100.00] | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
|            3162 | nan               | nan                  | nan                   | nan                     | 100.00 [100.00, 100.00] |

Median CI half-width across all conditions: **0.33 percentage points**.
