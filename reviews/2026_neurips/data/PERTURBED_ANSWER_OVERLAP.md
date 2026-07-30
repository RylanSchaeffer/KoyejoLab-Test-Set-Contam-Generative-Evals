# Do Modified Problems Keep the Original Answer?

At 0-shot, the Perturbed column shows a *larger* residual than Rephrased (4.78% vs 2.74% at R >= 100). That ordering is backwards: rephrasing preserves the answer, so regurgitation can still score correct, whereas perturbing changes the answer and regurgitation should score wrong.

A problem whose perturbation leaves the ground-truth answer unchanged hands a free point to a purely memorizing model.

| condition   |   n_compared |   n_identical_answer |   fraction_identical |
|:------------|-------------:|---------------------:|---------------------:|
| Perturbed   |         5000 |                  582 |               0.1164 |
| Rephrased   |         5000 |                 4998 |               0.9996 |

**Material.** 11.64% of perturbed problems keep the original answer, which is on the same order as the entire Perturbed residual. The residual should be reported net of these problems, or they should be excluded from the Perturbed column, before the number goes in the paper.
