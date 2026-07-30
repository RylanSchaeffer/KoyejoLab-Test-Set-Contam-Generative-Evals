# Temperature response, rescored with boxed-required scoring

Rescores every 0-shot temperature run from its raw responses under the boxed-required scorer, then recomputes the contamination advantage at matched temperature, `score(R) - score(R=0)`, which is the control for reviewer 8RFz's W2/Q2.

## Headline: rescoring does not change the answer

The concern that motivated this rescoring was real but small. The lenient scorer does inflate the uncontaminated arm more than the contaminated arm, so subtracting a lenient `R=0` over-subtracts and understates the advantage -- measured on 62M at tau=1.0, the advantage is 0.0066 lenient against 0.0100 strict. But that condition is one of the smallest contributors. Averaged over the contributing conditions the effect is **+0.3 pp**: retention at tau=1.0 goes from 0.2495 (lenient) to 0.2528 (strict). The table below therefore agrees with `TEMPERATURE_RESPONSE.md` to two significant figures at every temperature.

> An earlier version of this file reported 0.0961 at tau=1.0 and attributed the
> change to the scoring rule. That was wrong. The script had silently dropped 344M
> (which has no finished 0-shot R=0 run, so its advantage was NaN and
> `groupby().mean()` skipped it) and had also switched the estimator from a ratio
> of means to a mean of ratios. Coverage accounted for -12.8 pp and the estimator
> for -2.9 pp; the scoring rule accounted for +0.3 pp. See
> `reviews/2026_neurips/verification/TEMPERATURE_VERIFICATION.md`.

## Contamination advantage at matched temperature (strict scoring)

Averaged over conditions with greedy (strict) score >= 5%. `fraction_of_greedy_advantage` is the ratio of means, matching `analyze_temperature_response.py`; `mean_of_ratios` is the per-condition average, shown so the estimator choice is visible rather than load-bearing and invisible.

|      T |   advantage |   fraction_of_greedy_advantage |   mean_of_ratios |   n_conditions |
|-------:|------------:|-------------------------------:|-----------------:|---------------:|
| 0      |      0.7087 |                         1      |           1      |             13 |
| 0.1    |      0.7084 |                         0.9996 |           0.9972 |             13 |
| 0.1778 |      0.7057 |                         0.9958 |           0.9807 |             13 |
| 0.3162 |      0.6949 |                         0.9805 |           0.9321 |             13 |
| 0.5623 |      0.6379 |                         0.9001 |           0.7919 |             13 |
| 0.75   |      0.5132 |                         0.724  |           0.6022 |             13 |
| 0.938  |      0.2779 |                         0.3921 |           0.3068 |             13 |
| 1      |      0.1792 |                         0.2528 |           0.1928 |             13 |
| 1.2915 |      0.0026 |                         0.0036 |           0.0026 |             13 |
| 1.5    |      0      |                         0      |           0      |             13 |

Conditions contributing to each mean (model size -> replica levels):

```
Parameters
153M         [100, 316, 1000]
344M    [32, 100, 1000, 3162]
34M                     [316]
62M                [100, 316]
93M          [100, 316, 1000]
```

Clean reference: R=0 where it exists; 344M -> R=1 because the ten 0-shot 344M R=0 runs from
2025-09-25 all failed. **Validated 2026-07-30:** finished 344M R=0 0-shot runs do exist in
sweeps `woygzpil` (2025-12-19) and `oj6o8idv` (2025-12-31) and score 0.000-0.140% strict across
tau in {0, 0.316, 1.0} -- on the floor, like the R=1 stand-in, so the fallback moves nothing and
the retention row below is unaffected. See `reviews/2026_neurips/data/LENIENT_SCORER_AUDIT.md`. 344M R=1 scores 0.0004 strict at greedy, i.e. it sits on the uncontaminated floor, so it is a sound stand-in. Without the fallback 344M drops out of every mean silently and tau=1.0 reads 0.1251 (ratio of means) or 0.0961 (mean of ratios) instead of 0.2528.

One run, 344M R=100 at tau=1.0, was absent from the first version of the per-run CSV (a worker stalled and the table was regex-parsed out of the log). It is present now and scores 0.1758 strict.

Per-run detail: `temperature_response_rescored.csv`.
