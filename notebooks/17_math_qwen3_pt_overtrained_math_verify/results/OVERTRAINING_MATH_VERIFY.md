# Finding #4 in Accuracy Space

Math Verify on the overtrained checkpoints, 0-shot greedy decoding — the same protocol as `notebooks/11_*` and the teacher-forced notebooks.

**Accuracy tracks loss, so 8RFz's loss-vs-correctness objection is answered on its own terms.** The 'stealth contamination' alternative — accuracy persisting while loss rises — does not occur.

**But dilution is threshold-dependent, and that changes what the finding means.** Near the memorization threshold, overtraining suppresses contamination by more than an order of magnitude. Above it, 16x more training does essentially nothing: a heavily leaked benchmark stays memorized. Stating Finding #4 as 'overtraining dilutes contamination' invites the reading that training longer mitigates leakage, which is false exactly where it would matter most. The mechanism is dilution of the *contaminated token fraction* (see `reviews/2026_neurips/data/CONTAMINATED_TOKEN_FRACTION.md`), so it only helps when it pushes that fraction back below threshold.

## Retained performance, lowest vs highest overtrain multiplier

Restricted to conditions scoring above 5% at their lowest multiplier — elsewhere there is nothing to dilute.

**Compare `ot_low`/`ot_high` before comparing retained fractions across rows.** The replica ladders are ragged (a configuration only exists where the replicas fit inside the token budget), so conditions span different multiplier ranges and their retained fractions are not all measured over the same interval. The cleanest like-for-like comparison is within a single model size over the full 1x-16x span: at 93M, R=100 retains 0.019 while R=1000 retains 0.995 — same range, a ~50x difference in how much overtraining helps.

| Parameters   |   Num. Replicas |   ot_low |   ot_high |   score_low |   score_high |   retained_fraction |
|:-------------|----------------:|---------:|----------:|------------:|-------------:|--------------------:|
| 153M         |             100 |        1 |         4 |      0.8088 |       0.1798 |              0.2222 |
| 344M         |              32 |        1 |         4 |      0.1272 |       0.0162 |              0.1274 |
| 344M         |             100 |        1 |         8 |      0.9912 |       0.2156 |              0.2175 |
| 344M         |             316 |        2 |         8 |      0.9984 |       0.961  |              0.9625 |
| 344M         |            1000 |        1 |         4 |      1      |       0.9984 |              0.9984 |
| 344M         |            3162 |        1 |         4 |      1      |       0.9984 |              0.9984 |
| 34M          |             316 |        1 |        16 |      0.0732 |       0.0134 |              0.1831 |
| 34M          |            1000 |        4 |        16 |      0.3867 |       0.0764 |              0.1975 |
| 34M          |            3162 |        8 |        16 |      0.7984 |       0.6825 |              0.8547 |
| 62M          |             100 |        1 |        16 |      0.074  |       0.0038 |              0.0514 |
| 62M          |             316 |        1 |        16 |      0.799  |       0.0618 |              0.0773 |
| 62M          |            1000 |        2 |        16 |      0.9676 |       0.9214 |              0.9523 |
| 62M          |            3162 |        4 |        16 |      0.9942 |       0.9884 |              0.9942 |
| 93M          |             100 |        1 |        16 |      0.3729 |       0.007  |              0.0188 |
| 93M          |             316 |        1 |        16 |      0.9872 |       0.2861 |              0.2899 |
| 93M          |            1000 |        1 |        16 |      0.9994 |       0.9944 |              0.995  |
| 93M          |            3162 |        4 |        16 |      0.9978 |       0.9982 |              1.0004 |

## Full grid

Columns are the overtrain multiplier; rows are model size and contamination.

| Parameters   |   Num. MATH Test Set Replicas |      1.0 |      2.0 |      4.0 |      8.0 |     16.0 |
|:-------------|------------------------------:|---------:|---------:|---------:|---------:|---------:|
| 153M         |                             0 |   0.0118 |   0      |   0      | nan      | nan      |
| 153M         |                             1 |   0.0174 | nan      | nan      | nan      | nan      |
| 153M         |                             3 |   0.0182 | nan      | nan      | nan      | nan      |
| 153M         |                            10 |   0.0178 | nan      | nan      | nan      | nan      |
| 153M         |                            32 |   0.025  | nan      | nan      | nan      | nan      |
| 153M         |                           100 |   0.8088 |   0.5083 |   0.1798 | nan      | nan      |
| 153M         |                           316 |   1      | nan      | nan      | nan      | nan      |
| 153M         |                          1000 |   1      | nan      | nan      | nan      | nan      |
| 344M         |                             0 | nan      |   0      |   0      | nan      |   0      |
| 344M         |                             1 |   0.013  |   0      |   0.0006 |   0      | nan      |
| 344M         |                             3 |   0.0154 |   0.0004 |   0      |   0.0002 | nan      |
| 344M         |                            10 |   0.0218 |   0.0012 |   0.0026 |   0.001  | nan      |
| 344M         |                            32 |   0.1272 |   0.0258 |   0.0162 | nan      | nan      |
| 344M         |                           100 |   0.9912 |   0.8364 |   0.827  |   0.2156 | nan      |
| 344M         |                           316 | nan      |   0.9984 |   0.9984 |   0.961  | nan      |
| 344M         |                          1000 |   1      |   0.9984 |   0.9984 | nan      | nan      |
| 344M         |                          3162 |   1      |   0.9984 |   0.9984 | nan      | nan      |
| 34M          |                             0 |   0.0038 |   0      |   0      |   0      |   0      |
| 34M          |                             1 |   0.0022 |   0      |   0      |   0      |   0      |
| 34M          |                             3 |   0.0026 |   0      |   0      |   0      |   0      |
| 34M          |                            10 |   0.0076 |   0.0002 |   0      |   0.0002 |   0.0008 |
| 34M          |                            32 |   0.0164 |   0.0008 |   0.0004 |   0      |   0      |
| 34M          |                           100 |   0.0194 |   0.0138 |   0.0114 |   0.0022 |   0.001  |
| 34M          |                           316 |   0.0732 |   0.0416 |   0.0254 |   0.017  |   0.0134 |
| 34M          |                          1000 | nan      | nan      |   0.3867 |   0.1952 |   0.0764 |
| 34M          |                          3162 | nan      | nan      | nan      |   0.7984 |   0.6825 |
| 62M          |                             0 |   0.0126 |   0      |   0      |   0      |   0      |
| 62M          |                             1 |   0.0088 |   0.0002 |   0.0012 |   0      |   0      |
| 62M          |                             3 |   0.0148 |   0      |   0.0002 |   0.001  |   0      |
| 62M          |                            10 |   0.0166 |   0      |   0.0014 |   0.0006 |   0.0006 |
| 62M          |                            32 |   0.0178 |   0.0034 |   0.0012 |   0.0004 |   0.0002 |
| 62M          |                           100 |   0.074  |   0.044  |   0.016  |   0.0102 |   0.0038 |
| 62M          |                           316 |   0.799  |   0.7385 |   0.6067 |   0.3241 |   0.0618 |
| 62M          |                          1000 | nan      |   0.9676 |   0.9674 |   0.9766 |   0.9214 |
| 62M          |                          3162 | nan      | nan      |   0.9942 |   0.9894 |   0.9884 |
| 93M          |                             0 |   0.0074 |   0      |   0      |   0      |   0      |
| 93M          |                             1 |   0.0132 |   0.002  |   0      |   0      |   0      |
| 93M          |                             3 |   0.0142 |   0      |   0.0002 |   0.0002 |   0.0002 |
| 93M          |                            10 |   0.0176 |   0.0002 |   0.0002 |   0      |   0.0002 |
| 93M          |                            32 |   0.0254 |   0.0086 |   0.002  |   0.0006 |   0.0004 |
| 93M          |                           100 |   0.3729 |   0.2182 |   0.0462 |   0.0162 |   0.007  |
| 93M          |                           316 |   0.9872 |   0.98   |   0.9532 |   0.8462 |   0.2861 |
| 93M          |                          1000 |   0.9994 |   0.9956 |   0.9964 |   0.9956 |   0.9944 |
| 93M          |                          3162 | nan      | nan      |   0.9978 |   0.998  |   0.9982 |
