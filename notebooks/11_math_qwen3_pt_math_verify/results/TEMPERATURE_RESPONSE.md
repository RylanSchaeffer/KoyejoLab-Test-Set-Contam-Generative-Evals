# Temperature Response Is Contamination-Specific, Not Generic Degradation

0-shot. Each condition normalized by its own greedy (tau = 0) score, so any effect acting uniformly on all populations cancels. Conditions whose greedy score is below 5% are excluded from the normalized view — dividing a floor-level score by itself manufactures ratios out of noise.

## The answer to 8RFz's W2/Q2

The clean control is the **contamination advantage at matched temperature**:
`score(R) - score(R = 0)`, with both terms measured at the *same* tau. Any degradation
that acts on all generation hits both terms and cancels in the difference, so whatever
shrinkage remains is specific to contamination.

Averaged over conditions that show real contamination at greedy decoding (greedy score >= 5%):

|   Temp. |   advantage |   fraction_of_greedy_advantage |
|--------:|------------:|-------------------------------:|
|  0      |      0.6996 |                         1      |
|  0.1    |      0.698  |                         0.9978 |
|  0.1778 |      0.6946 |                         0.9929 |
|  0.3162 |      0.6837 |                         0.9774 |
|  0.5623 |      0.6288 |                         0.8989 |
|  0.75   |      0.5066 |                         0.7242 |
|  0.938  |      0.2731 |                         0.3903 |
|  1      |      0.1746 |                         0.2495 |
|  1.2915 |      0.0028 |                         0.004  |
|  1.5    |      0.0003 |                         0.0004 |

Conditions contributing to that mean (model size -> replica levels):

```
Parameters
153M         [100, 316, 1000]
344M    [32, 100, 1000, 3162]
34M                     [316]
62M                [100, 316]
93M          [100, 316, 1000]
```

Clean reference used per size: 153M -> R=0, 344M -> R=1, 34M -> R=0, 62M -> R=0, 93M -> R=0. 344M has no 0-shot R=0 run, so its lowest available replica level stands in; that
checkpoint scores ~1.3% at greedy, i.e. it is at the uncontaminated floor, so it is a
sound reference. Without this fallback 344M would drop out of the mean silently.

The advantage is not merely reduced by sampling — it is reduced *while the
uncontaminated baseline it is measured against is itself unaffected*, which is exactly
the asymmetry generic degradation cannot produce.

A note on what could not be computed, since it bears on how to phrase the claim: the
greedy-normalized ratio is only meaningful for conditions with a greedy score above the
floor. Uncontaminated models are at the floor by definition, so 'normalized contaminated
vs normalized uncontaminated' is not a computable comparison and should not be asserted.
The matched-tau difference above is the defensible version.

For contaminated conditions specifically, the retained fraction of greedy performance at tau = 1.0 is **0.283** (n = 8). Memorized regurgitation is
a narrow, high-probability path through the output distribution, and sampling knocks
the model off it.

## Scope of the claim

Restricted to tau <= 1.0. Note that tau = 1.0 is **not** a hot
setting — it is the model's own distribution — yet contaminated models fall toward the
uncontaminated floor there while uncontaminated models barely move. Above tau = 1
everything degrades; concede that rather than claiming it.

## Per-condition data

`temperature_response.csv`.
