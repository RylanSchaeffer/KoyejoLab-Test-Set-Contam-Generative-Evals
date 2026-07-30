# Finding #5 at Matched Protocol (0-Shot)

Both stages measured at 0-shot greedy decoding, so the comparison isolates SFT rather than confounding it with a protocol change.

**The ~60x collapse quoted in `REBUTTAL_PLAN.md` P0.1 is an artifact** of comparing 0-shot pretrained against 4-shot SFT. Use the numbers below instead.

## What SFT removes, where there was something to remove

Restricted to the 13 conditions scoring >= 5% before SFT.

- Mean pretrained: **71.02%**
- Mean after SFT: **3.00%**
- Median retained fraction: **0.028** (range 0.001-0.298)

The collapse is real and large, but it varies by more than an order of magnitude across conditions — quote the range, not a single multiplier.

## How much of the drop is format loss?

Raw accuracy scores a response incorrect both when the model emits no `\boxed{}` and when it emits one that is wrong. Those are different failures, and post-SFT boxed rates differ enormously by model size, so the distinction changes the interpretation:

| Parameters   |   Num. Replicas |   pretrained_score |   sft_score |   sft_boxed_rate |   sft_score_given_boxed |
|:-------------|----------------:|-------------------:|------------:|-----------------:|------------------------:|
| 34M          |             316 |               7.32 |        2.18 |            0.917 |                    2.38 |
| 62M          |             100 |               7.4  |        1.12 |            0.426 |                    2.63 |
| 62M          |             316 |              79.9  |        3.64 |            0.792 |                    4.6  |
| 93M          |             100 |              37.29 |        0.58 |            0.196 |                    2.97 |
| 93M          |             316 |              98.72 |        1.36 |            0.286 |                    4.76 |
| 93M          |            1000 |              99.94 |       21.6  |            0.879 |                   24.56 |
| 153M         |             100 |              80.88 |        0.44 |            0.099 |                    4.43 |
| 153M         |             316 |             100    |        0.36 |            0.071 |                    5.08 |
| 153M         |            1000 |             100    |        3.78 |            0.288 |                   13.12 |
| 344M         |              32 |              12.72 |        0.6  |            0.244 |                    2.46 |
| 344M         |             100 |              99.12 |        0.46 |            0.111 |                    4.15 |
| 344M         |            1000 |             100    |        0.12 |            0.019 |                    6.38 |
| 344M         |            3162 |             100    |        2.8  |            0.078 |                   35.71 |

- **3 conditions keep the format** (boxed rate >= 0.5). There the accuracy drop is genuine loss of memorized content: the model still answers in the expected form and is simply wrong.
- **6 conditions lose the format** (boxed rate < 0.2), and these are concentrated in the larger models. There the raw drop mostly measures that the model stopped emitting `\boxed{}` at all, and attributing it entirely to forgetting would overstate the result.

Report both columns. The defensible claim is that SFT removes the contamination advantage; the mechanism differs by scale, and `sft_score_given_boxed` is the column that isolates capability from formatting. Note it has its own selection effect — it conditions on a subset the model chose — so it is a diagnostic, not a drop-in replacement for the headline number.

## Per-condition

| Parameters   |   Num. Replicas |   pretrained_score |   sft_score |   retained_fraction |   sft_boxed_rate |
|:-------------|----------------:|-------------------:|------------:|--------------------:|-----------------:|
| 34M          |               0 |               0.38 |        0    |               0     |            0     |
| 34M          |               1 |               0.22 |        0    |               0     |            0     |
| 34M          |               3 |               0.26 |        0    |               0     |            0     |
| 34M          |              10 |               0.76 |        0.04 |               0.053 |            0.013 |
| 34M          |              32 |               1.64 |        0.28 |               0.171 |            0.071 |
| 34M          |             100 |               1.94 |        1.48 |               0.763 |            0.575 |
| 34M          |             316 |               7.32 |        2.18 |               0.298 |            0.917 |
| 62M          |               0 |               1.26 |        0    |               0     |            0     |
| 62M          |               1 |               0.88 |        0    |               0     |            0.002 |
| 62M          |               3 |               1.48 |        0.04 |               0.027 |            0.028 |
| 62M          |              10 |               1.66 |        0    |               0     |            0.01  |
| 62M          |              32 |               1.78 |        0.2  |               0.112 |            0.106 |
| 62M          |             100 |               7.4  |        1.12 |               0.151 |            0.426 |
| 62M          |             316 |              79.9  |        3.64 |               0.046 |            0.792 |
| 93M          |               0 |               0.74 |        0    |               0     |            0     |
| 93M          |               1 |               1.32 |        0.02 |               0.015 |            0.013 |
| 93M          |               3 |               1.42 |        0.02 |               0.014 |            0.01  |
| 93M          |              10 |               1.76 |        0.04 |               0.023 |            0.011 |
| 93M          |              32 |               2.54 |        0.46 |               0.181 |            0.157 |
| 93M          |             100 |              37.29 |        0.58 |               0.016 |            0.196 |
| 93M          |             316 |              98.72 |        1.36 |               0.014 |            0.286 |
| 93M          |            1000 |              99.94 |       21.6  |               0.216 |            0.879 |
| 153M         |               0 |               1.18 |        0    |               0     |            0     |
| 153M         |               1 |               1.74 |        0.02 |               0.011 |            0.015 |
| 153M         |               3 |               1.82 |        0.02 |               0.011 |            0.015 |
| 153M         |              10 |               1.78 |        0.06 |               0.034 |            0.027 |
| 153M         |              32 |               2.5  |        0.7  |               0.28  |            0.201 |
| 153M         |             100 |              80.88 |        0.44 |               0.005 |            0.099 |
| 153M         |             316 |             100    |        0.36 |               0.004 |            0.071 |
| 153M         |            1000 |             100    |        3.78 |               0.038 |            0.288 |
| 344M         |               0 |             nan    |        0    |             nan     |            0     |
| 344M         |               1 |               1.3  |        0.02 |               0.015 |            0.027 |
| 344M         |               3 |               1.54 |        0.02 |               0.013 |            0.022 |
| 344M         |              10 |               2.18 |        0.18 |               0.083 |            0.059 |
| 344M         |              32 |              12.72 |        0.6  |               0.047 |            0.244 |
| 344M         |             100 |              99.12 |        0.46 |               0.005 |            0.111 |
| 344M         |             316 |             nan    |        0.14 |             nan     |            0.055 |
| 344M         |            1000 |             100    |        0.12 |               0.001 |            0.019 |
| 344M         |            3162 |             100    |        2.8  |               0.028 |            0.078 |

`sft_boxed_rate` is the fraction of responses containing a `\boxed{}` at all. If it collapses, the accuracy drop is partly a formatting artifact rather than lost capability — check it before attributing the drop entirely to forgetting.
