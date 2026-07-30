# Finding #2 at 0-Shot: Table 1, With a Baseline

Greedy decoding, 0-shot — the protocol under which contamination actually produces gains, and the one behind the manuscript's Finding #1 figure.

The manuscript's Table 1 omits an Original column, so the reader supplies the baseline from Fig. 1. Including it here makes the collapse legible in one table, and gives the numbers provenance that the printed table currently lacks.

## Math Verify %, by condition

| Parameters   |   Num. Replicas |   Original |   Rephrased |   Perturbed |
|:-------------|----------------:|-----------:|------------:|------------:|
| 34M          |               0 |       0.38 |        0    |        0    |
| 34M          |               1 |       0.22 |        0    |        0    |
| 34M          |               3 |       0.26 |        0    |        0    |
| 34M          |              10 |       0.76 |        0.04 |        0.02 |
| 34M          |              32 |       1.64 |        0.66 |        0.36 |
| 34M          |             100 |       1.94 |        1.74 |        1.09 |
| 34M          |             316 |       7.32 |        1.86 |        1.45 |
| 62M          |               0 |       1.26 |        0    |        0    |
| 62M          |               1 |       0.88 |        0    |        0    |
| 62M          |               3 |       1.48 |        0.18 |        0.07 |
| 62M          |              10 |       1.66 |        0.02 |        0    |
| 62M          |              32 |       1.78 |        0.72 |        0.41 |
| 62M          |             100 |       7.4  |        1.6  |        1.43 |
| 62M          |             316 |      79.9  |        2.04 |        1.72 |
| 93M          |               0 |       0.74 |        0    |        0    |
| 93M          |               1 |       1.32 |        0    |        0    |
| 93M          |               3 |       1.42 |        0.1  |        0.02 |
| 93M          |              10 |       1.76 |        0.1  |        0.07 |
| 93M          |              32 |       2.54 |        1.46 |        1.13 |
| 93M          |             100 |      37.29 |        2.4  |        2.08 |
| 93M          |             316 |      98.72 |        3.18 |        2.1  |
| 93M          |            1000 |      99.94 |        3.44 |        1.63 |
| 153M         |               0 |       1.18 |        0    |        0    |
| 153M         |               1 |       1.74 |        0    |        0    |
| 153M         |               3 |       1.82 |        0.04 |        0    |
| 153M         |              10 |       1.78 |        0.04 |        0.02 |
| 153M         |              32 |       2.5  |        1.64 |        1.13 |
| 153M         |             100 |      80.88 |        2.06 |        2.06 |
| 153M         |             316 |     100    |        2.88 |        2.15 |
| 153M         |            1000 |     100    |        3.32 |        1.86 |
| 344M         |               0 |       0    |        0    |        0    |
| 344M         |               1 |       1.3  |        0.02 |        0.02 |
| 344M         |               3 |       1.54 |        0.04 |        0.07 |
| 344M         |              10 |       2.18 |        0.14 |        0.43 |
| 344M         |              32 |      12.72 |        2.12 |        1.74 |
| 344M         |             100 |      99.12 |        2.8  |        2.51 |
| 344M         |             316 |      99.84 |        3.26 |        2.2  |
| 344M         |            1000 |     100    |        3.1  |        2.38 |
| 344M         |            3162 |     100    |        5.24 |        2.1  |

## How complete is the collapse?

Uncontaminated floor (R = 0, Original): **0.71%** (n = 5 model sizes).

- **Rephrased**, at R >= 100: Original **72.31%** -> Rephrased **2.78%** (n = 14 checkpoints). That removes **97.1%** of the contamination advantage over the floor — but note the residual: Rephrased sits at 3.9x the uncontaminated floor, not at it.
- **Perturbed**, at R >= 100: Original **72.31%** -> Perturbed **1.91%** (n = 14 checkpoints). That removes **98.3%** of the contamination advantage over the floor — but note the residual: Perturbed sits at 2.7x the uncontaminated floor, not at it.

The residual is small but consistent and should be stated rather than rounded away. Describing the collapse as reaching 'baseline' overstates it; 'removes the large majority of the contamination advantage, leaving a small residual' is what the data support.
