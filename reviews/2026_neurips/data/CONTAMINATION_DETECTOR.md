# Temperature Response as a Black-Box Contamination Detector

Not requested by any reviewer, which is why it is worth including: it adds a
contribution rather than only patching holes.

## The idea

Detecting contamination in a released model normally needs the training corpus, or a
known-clean reference model to compare accuracy against. The temperature response needs
neither — it compares a model **to itself** at two decoding temperatures. Contaminated
models lose most of their advantage between tau = 0 and tau = 1 because verbatim
regurgitation is a narrow high-probability path; uncontaminated models are flat because
they had nothing to lose.

## Separability

32 pretrained checkpoints, 13 contaminated (R >= 100) vs 19 clean (R <= 10), 0-shot greedy vs tau = 1.

| feature       |    auc |   permutation_p |
|:--------------|-------:|----------------:|
| greedy_only   | 0.996  |               0 |
| absolute_drop | 0.9919 |               0 |
| relative_drop | 0.9393 |               0 |

`greedy_only` is the naive baseline. Its advantage is that it is simple; its problem is
that thresholding raw accuracy requires already knowing what accuracy a clean model of
that size achieves, which is exactly what an auditor lacks. The drop features need no
such reference.

## The comparison above cannot establish which feature is better

This is the most important caveat and it should not be buried. In this grid **every
uncontaminated checkpoint scores near zero**, because these are tiny models trained from
scratch with no mathematical capability (see the pass@k result: 0 correct out of
5,000,000 samples). When all negatives sit at the floor, *any* feature that keys on high
greedy accuracy separates the classes almost perfectly — which is why `greedy_only`
scores 0.996 here.

So these AUCs establish that **the temperature signal exists and is strong**. They do
**not** establish that the drop features beat raw accuracy, because the regime where the
drop features should win — a genuinely capable clean model that scores well at tau = 0
and stays there at tau = 1 — contains no checkpoints in this study. Claiming superiority
from this table would be an overclaim of exactly the kind the paper is already being
criticized for.

The honest framing: the mechanism is demonstrated, the deployment advantage is argued
from first principles (no reference model required), and validating it needs capable
clean models as negatives. That is a concrete, checkable camera-ready commitment.

## What this does not show

- A few dozen checkpoints, one architecture family, one benchmark, one contamination
  mechanism (verbatim replicas). AUC on this sample size is noisy; the permutation test
  is reported instead of an asymptotic interval for that reason.
- These are small models trained from scratch. A large pretrained model with genuine
  competence may degrade differently under sampling, and that is the case that matters
  for real audits.
- Untested against paraphrased or partial contamination, which is where a detector would
  most plausibly fail.

Present it as a proof of concept with a clear path to validation, not as a working
detector. Overclaiming here would repeat the framing error the paper is already being
criticized for.
