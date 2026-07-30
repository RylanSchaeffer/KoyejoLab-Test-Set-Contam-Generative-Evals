# Temperature response, rescored with boxed-required scoring

Supersedes the `fraction_of_greedy_advantage` column of `TEMPERATURE_RESPONSE.md`, which was
computed from leniently scored runs.

The matched-temperature *difference* was supposed to cancel scoring artifacts. It does not:
the uncontaminated arm's lenient score is almost entirely false positives while the
contaminated arm's is mostly real, so subtracting a lenient R=0 over-subtracts and
**understates** the advantage — most severely in the high-temperature tail where the true
values are smallest. Measured on 62M at tau=1.0: advantage 0.0066 lenient vs 0.0100 strict.

Averaged over conditions with greedy (strict) score >= 5%. 369 of 370 runs; one run
hung in the scorer and is excluded (its omission cannot move a mean over ~100 conditions).

|      T |   advantage |   fraction_of_greedy_advantage |
|-------:|------------:|-------------------------------:|
| 0      |      0.6781 |                         1      |
| 0.1    |      0.6779 |                         0.9976 |
| 0.1778 |      0.6741 |                         0.9749 |
| 0.3162 |      0.6618 |                         0.9241 |
| 0.5623 |      0.5987 |                         0.7718 |
| 0.75   |      0.4518 |                         0.5495 |
| 0.938  |      0.1741 |                         0.2014 |
| 1      |      0.0849 |                         0.0961 |
| 1.2915 |      0.0002 |                         0.0002 |
| 1.5    |      0      |                         0      |

Per-run detail: `temperature_response_rescored.csv`.
