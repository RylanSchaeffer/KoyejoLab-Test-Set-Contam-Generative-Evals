# Robustness of the Irreducible-Error Claim (aPBL Q3)

## The logical structure, which the paper should state explicitly

Contaminated losses are **measured**. Only the uncontaminated asymptote E(0) is
**extrapolated**. So the claim does not require the functional form to be correct — it
requires only that a conservative lower bound on E(0) still exceed the losses measured
at R >= 1. Framing it that way converts a modelling assumption into a much weaker and
checkable one.

## Fitted irreducible error per contamination level

`L = E + C_0 * C^(-alpha)`, refit on the measured points; 95% bootstrap interval over those points, 300 resamples.

|   Num. Replicas |   n_points |   E_point_estimate |   n_bootstrap_ok |   E_ci_lower |   E_ci_upper |   min_measured_loss | reliable   |
|----------------:|-----------:|-------------------:|-----------------:|-------------:|-------------:|--------------------:|:-----------|
|               0 |          6 |             3.5942 |              275 |       3.5359 |       3.6639 |              3.9015 | True       |
|               1 |          5 |             1.8377 |              274 |       1.7601 |       1.873  |              1.9834 | True       |
|               3 |          5 |             1.6759 |              276 |       1.5965 |       1.7478 |              1.7729 | True       |
|              10 |          5 |             1.2284 |              275 |       1.2157 |       1.3251 |              1.3485 | True       |
|              32 |          5 |             0      |              276 |       0      |       0      |              0.2733 | False      |
|             100 |          5 |             0.0474 |              266 |       0.0075 |       0.0595 |              0.0562 | True       |
|             316 |          6 |             0.0347 |              282 |       0.0341 |       0.0356 |              0.0354 | True       |
|            1000 |          3 |             0.0298 |               79 |       0.0298 |       0.0298 |              0.0325 | False      |

## Does the claim survive the uncertainty?

- E(0) point estimate: **3.5942**
- E(0) lower end of the interval: **3.5359**
- Contaminated runs (R >= 1) whose *measured* loss falls below that lower bound: **33 of 35** (94.3%)

The claim survives: a majority of contaminated runs beat even the conservative end of the uncontaminated asymptote's interval, so it does not rest on the point estimate.

## Caveats — read before quoting an interval

**The intervals are optimistically narrow.** Each bootstrap resample is refit by local
optimization seeded at the full-data solution, because the repo's grid search over 5,760
starting points is far too slow to bootstrap. Seeding every resample at the same solution
biases each refit toward it, so the spread understates true parameter uncertainty. Treat
these as a lower bound on the uncertainty, not a calibrated interval. The headline
conclusion is robust to this because it clears the bound by a wide margin (E(0) ~ 3.5 vs
contaminated losses ~1-2), but do not quote an interval width as if it were calibrated.

**Each level is fit from 3-6 model sizes.** Resamples that collapse the compute range are
discarded, since an asymptote is not identifiable without spread in the covariate;
`n_bootstrap_ok` records how many survived. A level with few survivors, or with `E`
driven to 0, is flagged unreliable in the table and should not be reported as a measured
irreducible error — `E = 0` means the optimizer pushed `e_0` toward negative infinity,
i.e. the data are consistent with *no* asymptote, not that the asymptote is zero.
