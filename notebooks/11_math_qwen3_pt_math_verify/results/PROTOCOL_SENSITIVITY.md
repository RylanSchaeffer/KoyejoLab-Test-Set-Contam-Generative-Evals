# Evaluation Protocol Determines the Measured Contamination Effect

Greedy decoding (temperature = 0.0), pretrained (`ot=1`) checkpoints only.

## Headline

| Protocol | Peak Math Verify across the whole grid |
|---|---|
| 0-shot | 1.0000 |
| 4-shot | 0.0112 |

Under 0-shot prompting, heavily contaminated checkpoints reproduce the memorized
solution verbatim and saturate. Prepending four worked examples moves the prompt off
the memorized context, and the same checkpoints score near the uncontaminated floor.

## Why this matters for the manuscript

- `notebooks/11_*` declares the 4-shot sweep IDs but its cached data was built from
  the 0-shot list (confirmed by reproducing the cache's md5 filename). The figure the
  manuscript uses for Finding #1 is therefore 0-shot.
- `notebooks/13_*` (SFT) reads 4-shot sweeps. Comparing the pretrained figure against
  the SFT figure compares protocols as well as training stages, so the apparent
  'SFT collapses accuracy' effect is confounded with the protocol change.
- Any claim about contamination magnitude must state the protocol it was measured under.

## Per-condition comparison

| Parameters   |   Num. Replicas |   0-shot |   4-shot |   ratio (0-shot / 4-shot) |
|:-------------|----------------:|---------:|---------:|--------------------------:|
| 153M         |               0 |   0.0118 |   0.0000 |                  nan      |
| 153M         |               1 |   0.0174 |   0.0010 |                   17.4000 |
| 153M         |               3 |   0.0182 |   0.0020 |                    9.1000 |
| 153M         |              10 |   0.0178 |   0.0090 |                    1.9778 |
| 153M         |              32 |   0.0250 |   0.0092 |                    2.7174 |
| 153M         |             100 |   0.8088 |   0.0112 |                   72.2321 |
| 153M         |             316 |   1.0000 |   0.0078 |                  128.2308 |
| 153M         |            1000 |   1.0000 |   0.0088 |                  113.6591 |
| 344M         |               0 | nan      |   0.0000 |                  nan      |
| 344M         |               1 |   0.0130 |   0.0012 |                   10.8333 |
| 344M         |               3 |   0.0154 |   0.0026 |                    5.9231 |
| 344M         |              10 |   0.0218 |   0.0030 |                    7.2667 |
| 344M         |              32 |   0.1272 |   0.0058 |                   21.9310 |
| 344M         |             100 |   0.9912 |   0.0074 |                  133.9730 |
| 344M         |             316 | nan      |   0.0066 |                  nan      |
| 344M         |            1000 |   1.0000 |   0.0104 |                   96.1731 |
| 344M         |            3162 |   1.0000 |   0.0052 |                  192.3462 |
| 34M          |               0 |   0.0038 |   0.0000 |                  nan      |
| 34M          |               1 |   0.0022 |   0.0000 |                  nan      |
| 34M          |               3 |   0.0026 |   0.0000 |                  nan      |
| 34M          |              10 |   0.0076 |   0.0004 |                   19.0000 |
| 34M          |              32 |   0.0164 |   0.0070 |                    2.3429 |
| 34M          |             100 |   0.0194 |   0.0050 |                    3.8800 |
| 34M          |             316 |   0.0732 |   0.0060 |                   12.2000 |
| 62M          |               0 |   0.0126 |   0.0000 |                  nan      |
| 62M          |               1 |   0.0088 |   0.0000 |                  nan      |
| 62M          |               3 |   0.0148 |   0.0016 |                    9.2500 |
| 62M          |              10 |   0.0166 |   0.0008 |                   20.7500 |
| 62M          |              32 |   0.0178 |   0.0010 |                   17.8000 |
| 62M          |             100 |   0.0740 |   0.0062 |                   11.9355 |
| 62M          |             316 |   0.7990 |   0.0060 |                  133.2000 |
| 93M          |               0 |   0.0074 |   0.0000 |                  nan      |
| 93M          |               1 |   0.0132 |   0.0002 |                   66.0000 |
| 93M          |               3 |   0.0142 |   0.0054 |                    2.6296 |
| 93M          |              10 |   0.0176 |   0.0054 |                    3.2593 |
| 93M          |              32 |   0.0254 |   0.0052 |                    4.8846 |
| 93M          |             100 |   0.3729 |   0.0036 |                  103.6111 |
| 93M          |             316 |   0.9872 |   0.0054 |                  182.8519 |
| 93M          |            1000 |   0.9994 |   0.0052 |                  192.2308 |
