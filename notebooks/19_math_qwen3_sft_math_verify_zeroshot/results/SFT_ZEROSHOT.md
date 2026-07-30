# Finding #5 at Matched Protocol (0-Shot)

Both stages measured at 0-shot greedy decoding, so the comparison isolates SFT rather than confounding it with a protocol change.

**The ~60x collapse quoted in `REBUTTAL_PLAN.md` P0.1 is an artifact** of comparing 0-shot pretrained against 4-shot SFT. Use the numbers below instead.

## What SFT removes, where there was something to remove

Restricted to the 13 conditions scoring >= 5% before SFT.

- Mean pretrained: **70.89%**
- Mean after SFT: **3.00%**
- Median retained fraction: **0.028** (range 0.001-0.302)

The collapse is real and large, but it varies by more than an order of magnitude across conditions — quote the range, not a single multiplier.

## How much of the drop is format loss?

Raw accuracy scores a response incorrect both when the model emits no `\boxed{}` and when it emits one that is wrong. Those are different failures, and post-SFT boxed rates differ enormously by model size, so the distinction changes the interpretation:

| Parameters   |   Num. Replicas |   pretrained_score |   sft_score |   sft_boxed_rate |   sft_score_given_boxed |
|:-------------|----------------:|-------------------:|------------:|-----------------:|------------------------:|
| 34M          |             316 |               7.22 |        2.18 |            0.917 |                    2.38 |
| 62M          |             100 |               7.3  |        1.12 |            0.426 |                    2.63 |
| 62M          |             316 |              79.78 |        3.64 |            0.792 |                    4.6  |
| 93M          |             100 |              37.25 |        0.58 |            0.196 |                    2.97 |
| 93M          |             316 |              98.56 |        1.36 |            0.286 |                    4.76 |
| 93M          |            1000 |              99.78 |       21.6  |            0.879 |                   24.56 |
| 153M         |             100 |              80.74 |        0.44 |            0.099 |                    4.43 |
| 153M         |             316 |              99.84 |        0.36 |            0.071 |                    5.08 |
| 153M         |            1000 |              99.84 |        3.78 |            0.288 |                   13.12 |
| 344M         |              32 |              12.56 |        0.6  |            0.244 |                    2.46 |
| 344M         |             100 |              98.96 |        0.46 |            0.111 |                    4.15 |
| 344M         |            1000 |              99.84 |        0.12 |            0.019 |                    6.38 |
| 344M         |            3162 |              99.84 |        2.8  |            0.078 |                   35.71 |

- **3 conditions keep the format** (boxed rate >= 0.5). There the accuracy drop is genuine loss of memorized content: the model still answers in the expected form and is simply wrong.
- **6 conditions lose the format** (boxed rate < 0.2), and these are concentrated in the larger models. There the raw drop mostly measures that the model stopped emitting `\boxed{}` at all, and attributing it entirely to forgetting would overstate the result.

Report both columns. The defensible claim is that SFT removes the contamination advantage; the mechanism differs by scale, and `sft_score_given_boxed` is the column that isolates capability from formatting. Note it has its own selection effect — it conditions on a subset the model chose — so it is a diagnostic, not a drop-in replacement for the headline number.

## Per-condition

| Parameters   |   Num. Replicas |   pretrained_score |   sft_score |   retained_fraction |   sft_boxed_rate |
|:-------------|----------------:|-------------------:|------------:|--------------------:|-----------------:|
| 34M          |               0 |               0    |        0    |             nan     |            0     |
| 34M          |               1 |               0    |        0    |             nan     |            0     |
| 34M          |               3 |               0    |        0    |             nan     |            0     |
| 34M          |              10 |               0.06 |        0.04 |               0.667 |            0.013 |
| 34M          |              32 |               0.56 |        0.28 |               0.5   |            0.071 |
| 34M          |             100 |               1.7  |        1.48 |               0.871 |            0.575 |
| 34M          |             316 |               7.22 |        2.18 |               0.302 |            0.917 |
| 62M          |               0 |               0    |        0    |             nan     |            0     |
| 62M          |               1 |               0.02 |        0    |               0     |            0.002 |
| 62M          |               3 |               0.08 |        0.04 |               0.5   |            0.028 |
| 62M          |              10 |               0    |        0    |             nan     |            0.01  |
| 62M          |              32 |               0.66 |        0.2  |               0.303 |            0.106 |
| 62M          |             100 |               7.3  |        1.12 |               0.153 |            0.426 |
| 62M          |             316 |              79.78 |        3.64 |               0.046 |            0.792 |
| 93M          |               0 |               0    |        0    |             nan     |            0     |
| 93M          |               1 |               0    |        0.02 |             nan     |            0.013 |
| 93M          |               3 |               0.02 |        0.02 |               1     |            0.01  |
| 93M          |              10 |               0.02 |        0.04 |               2     |            0.011 |
| 93M          |              32 |               1.42 |        0.46 |               0.324 |            0.157 |
| 93M          |             100 |              37.25 |        0.58 |               0.016 |            0.196 |
| 93M          |             316 |              98.56 |        1.36 |               0.014 |            0.286 |
| 93M          |            1000 |              99.78 |       21.6  |               0.216 |            0.879 |
| 153M         |               0 |               0    |        0    |             nan     |            0     |
| 153M         |               1 |               0.02 |        0.02 |               1     |            0.015 |
| 153M         |               3 |               0.04 |        0.02 |               0.5   |            0.015 |
| 153M         |              10 |               0.08 |        0.06 |               0.75  |            0.027 |
| 153M         |              32 |               2.02 |        0.7  |               0.347 |            0.201 |
| 153M         |             100 |              80.74 |        0.44 |               0.005 |            0.099 |
| 153M         |             316 |              99.84 |        0.36 |               0.004 |            0.071 |
| 153M         |            1000 |              99.84 |        3.78 |               0.038 |            0.288 |
| 344M         |               0 |             nan    |        0    |             nan     |            0     |
| 344M         |               1 |               0.04 |        0.02 |               0.5   |            0.027 |
| 344M         |               3 |               0.06 |        0.02 |               0.333 |            0.022 |
| 344M         |              10 |               0.26 |        0.18 |               0.692 |            0.059 |
| 344M         |              32 |              12.56 |        0.6  |               0.048 |            0.244 |
| 344M         |             100 |              98.96 |        0.46 |               0.005 |            0.111 |
| 344M         |             316 |             nan    |        0.14 |             nan     |            0.055 |
| 344M         |            1000 |              99.84 |        0.12 |               0.001 |            0.019 |
| 344M         |            3162 |              99.84 |        2.8  |               0.028 |            0.078 |

`sft_boxed_rate` is the fraction of responses containing a `\boxed{}` at all. If it collapses, the accuracy drop is partly a formatting artifact rather than lost capability — check it before attributing the drop entirely to forgetting.
