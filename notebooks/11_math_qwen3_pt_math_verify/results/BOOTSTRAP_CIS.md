# Bootstrap Confidence Intervals on Math Verify

95% percentile bootstrap over the 5001 MATH test problems, 10,000 resamples, greedy decoding, **0-shot** (the protocol behind the manuscript's Finding #1 figure).

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
