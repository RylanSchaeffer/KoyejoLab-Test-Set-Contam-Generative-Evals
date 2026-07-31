# `\boxed{}` Format Sanity Check

Greedy decoding (temperature = 0.0). 78 runs, 390078 scored responses.

**Both stages here are 4-shot**, which is the point: comparing the SFT figures in
`notebooks/13_*` against the pretrained figure in `notebooks/11_*` compares 4-shot
against 0-shot, because notebook 11's cache was built from the superseded 0-shot
sweeps. See `reviews/2026_neurips/PROTOCOL_CONFOUND.md`.

## Verdict

**Genuine task failure.** SFT models still emit `\boxed{}` at 19.5% versus 36.3% for pretrained models (0.54x). The post-SFT accuracy floor is not a formatting artifact.

## The matched-protocol numbers

At 4-shot, mean Math Verify is **0.0040 pretrained** and **0.0020 after SFT**. Both sit at the uncontaminated floor, so there is
no large post-SFT collapse to explain in this protocol. The ~60x collapse quoted in
`REBUTTAL_PLAN.md` (P0.1) compares 0-shot pretrained (~100%) against 4-shot SFT
(~1-2%) and is an artifact of the protocol mismatch. Do not use it.

## Why this check exists

`src.scoring.score_response` scores a response incorrect unless it contains a
`\boxed{...}`. A model that forgot the output format is therefore indistinguishable
from a model that cannot solve the problems, unless the format rate is measured
separately. The post-SFT Math Verify floor is only a contamination result under the
first reading.

## By stage

| stage      |   n_runs |   mean_boxed_rate |   min_boxed_rate |   max_boxed_rate |   mean_accuracy |   mean_response_chars |
|:-----------|---------:|------------------:|-----------------:|-----------------:|----------------:|----------------------:|
| pretrained |       39 |          0.362717 |                0 |         0.886623 |      0.00398895 |               1846.86 |
| sft        |       39 |          0.195151 |                0 |         0.676065 |      0.00202011 |               2494.44 |

## By stage and model size

| stage      | Parameters   |   n_runs |   mean_boxed_rate |   mean_accuracy |
|:-----------|:-------------|---------:|------------------:|----------------:|
| pretrained | 153M         |        8 |         0.486203  |      0.00612378 |
| pretrained | 344M         |        9 |         0.418494  |      0.00468795 |
| pretrained | 34M          |        7 |         0.187248  |      0.00262805 |
| pretrained | 62M          |        7 |         0.255663  |      0.00222813 |
| pretrained | 93M          |        8 |         0.42369   |      0.00379924 |
| sft        | 153M         |        8 |         0.288567  |      0.00359928 |
| sft        | 344M         |        9 |         0.267258  |      0.00239952 |
| sft        | 34M          |        7 |         0.0988945 |      0.00137115 |
| sft        | 62M          |        7 |         0.155169  |      0.00102837 |
| sft        | 93M          |        8 |         0.139822  |      0.00144971 |

## Per-run detail

See `boxed_format_rates.csv`.

---

Regenerate: `./mem_scoring_vs_sampling_env/bin/python scripts/check_boxed_format_rate.py`
