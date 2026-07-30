# Does SFT improve generalization? Teacher-forced NLL on perturbed MATH

Perturbed MATH problems are novel: never seen during pretraining or SFT. If SFT lowers NLL on
them, the model has genuinely generalized rather than merely memorized. Teacher forcing is
0-shot by design (the prompt must match what was injected during pretraining).

Paired per-problem over 7,138 problems, so pre- and post-SFT are compared on
identical items.

## Correction to the manuscript

`04_further_training.tex` currently states **"SFT lowers NLL at 14/17 conditions, with
improvements up to -4.72 nats"**. Those numbers predate the token-weighting fix in commit
`342deb5` (2026-07-27), which corrected an aggregation that weighted problems by token count
rather than uniformly.

**Corrected: SFT lowers NLL at 17/17 conditions, with improvements up to 2.18 nats**
(at 153M, R = 1000).

Stronger on consistency — the effect is now universal rather than 82% of conditions — and weaker
on magnitude. Both halves of that should be stated; quoting only the first would be selective.

## Per condition

| Parameters   |   Num. MATH Test Set Replicas |   Pre-SFT Mean NLL |   Post-SFT Mean NLL |   Mean Delta NLL (Paired Per Problem) |   N Paired Problems |
|:-------------|------------------------------:|-------------------:|--------------------:|--------------------------------------:|--------------------:|
| 153M         |                             0 |             3.5958 |              3.2431 |                               -0.3527 |                7138 |
| 153M         |                             1 |             2.7377 |              2.6732 |                               -0.0645 |                7138 |
| 153M         |                             3 |             2.7385 |              2.7066 |                               -0.0319 |                7138 |
| 153M         |                            10 |             2.7341 |              2.6701 |                               -0.064  |                7138 |
| 153M         |                            32 |             3.2259 |              2.5996 |                               -0.6263 |                7138 |
| 153M         |                           100 |             3.8864 |              2.7892 |                               -1.0971 |                7138 |
| 153M         |                           316 |             4.606  |              3.103  |                               -1.503  |                7138 |
| 153M         |                          1000 |             6.0111 |              3.8341 |                               -2.1769 |                7138 |
| 344M         |                             0 |             3.2695 |              2.9735 |                               -0.2959 |                7138 |
| 344M         |                             1 |             2.621  |              2.5569 |                               -0.0641 |                7138 |
| 344M         |                             3 |             2.5535 |              2.5336 |                               -0.0199 |                7138 |
| 344M         |                            10 |             2.4656 |              2.4017 |                               -0.0639 |                7138 |
| 344M         |                            32 |             3.6259 |              2.3244 |                               -1.3015 |                7138 |
| 344M         |                           100 |             4.4277 |              2.463  |                               -1.9647 |                7138 |
| 344M         |                           316 |             3.2983 |              2.5149 |                               -0.7834 |                7138 |
| 344M         |                          1000 |             4.3047 |              2.6896 |                               -1.6151 |                7138 |
| 344M         |                          3162 |             5.6902 |              3.6314 |                               -2.0588 |                7138 |

Negative delta = SFT lowered NLL = improved generalization to unseen problems.
