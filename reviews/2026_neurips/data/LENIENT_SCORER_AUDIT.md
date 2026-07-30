# The lenient scorer as a capability upper bound — validation and audit

Reproduce: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/audit_lenient_scorer.py`

Closes the open item left by `reviews/2026_icml/REVIEWER_Mmea/SCORING_INVESTIGATION.md`, which
measured the scorer's false-positive rate on *sampled* output and then said the greedy "correct"
predictions "should be manually inspected to confirm they are false positives." Also validates
that the scorer is genuinely generous rather than merely broken, and fills the 344M gap.

---

## What the lenient scorer is

Not something we built. It is `math_verify.parse()` applied directly to the raw response — the
**original** scorer, used before commit `db75c5f`. It is "lenient" because `parse()` extracts bare
numbers from free text at priority 300, so it credits the gold answer appearing anywhere in the
output, with no `\boxed{}` required and no regard for whether the surrounding text is reasoning or
noise.

**We use it only as an upper bound.** The argument is "even a scorer that over-credits finds
nothing," so its generosity is the point and its imprecision runs in our favour. Never use it to
estimate accuracy.

---

## Why validating it matters

The argument only holds if the scorer really does over-credit. A scorer with poor *recall* would
report zero because it is broken, not because there is nothing there. Four tests.

### 1. Synthetic recall — plant a correct answer and see if it is credited

12 gold answers × 7 surface forms a capable-but-unformatted model might emit.

| Surface form | Recall |
|---|---|
| `42` (bare) | 8/12 |
| `The answer is 42.` | 8/12 |
| `So x = 42.` | 8/12 |
| `$\boxed{42}$` | **12/12** |
| `...the result is 42` | 8/12 |
| `We get 42. This completes...` | 8/12 |
| `The value is $42$.` | **12/12** |

**Overall 64/84 = 76.2%, and the misses are entirely one category.** All numeric answers
(integers, negatives, decimals, fractions) are credited in **every** surface form — 100% recall.
The four failures are **symbolic** answers (`\sqrt{2}`, `\pi`, `x+1`, `(2,3)`), which the parser
only sees when wrapped in math delimiters (`$...$` or `\boxed{}`).

So the scorer has one blind spot: a bare symbolic answer in plain prose. Test 4 closes it.

### 2. Dominance — does lenient credit everything strict credits?

Across 20,004 responses spanning R = 0 → 316: **1 case** of strict-correct-but-lenient-wrong.
Lenient is a superset of strict for practical purposes, which is what the upper-bound argument
needs.

### 3. Regurgitation recall — real, indisputably correct output

At R = 316 the contaminated 34M model reproduces the gold solution **verbatim in 229 responses**.
These are unambiguously correct. The lenient scorer credits **229/229 — zero missed.** This is the
strongest of the four tests because it uses real model output rather than constructed strings.

### 4. Substring fallback — closing the symbolic blind spot

A scorer-independent check: does the gold answer appear anywhere in the response as a raw
substring? This catches a bare symbolic answer the parser cannot see. Restricted to the 1,153
problems with symbolic gold answers:

| Model (R = 0) | Symbolic problems | Substring hits | Rate |
|---|---|---|---|
| 34M | 1,153 | 0 | 0.00% |
| 62M | 1,153 | 2 | 0.17% |
| 93M | 1,153 | 1 | 0.09% |
| 153M | 1,153 | 1 | 0.09% |
| 344M | 1,153 | 3–9 | 0.26–0.78% |

**The blind spot hides nothing.** On exactly the problems where the parser could miss a bare
symbolic answer, the maximally generous test finds ≤0.78%.

---

## The audit: every leniently-credited response, inspected

| Model (R = 0, 0-shot greedy) | Responses | Lenient "correct" | Rate | Containing `\boxed{}` |
|---|---|---|---|---|
| 34M | 5,001 | 19 | 0.38% | **0** |
| 62M | 5,001 | 63 | 1.26% | **0** |
| 93M | 5,001 | 37 | 0.74% | **0** |
| 153M | 5,001 | 59 | 1.18% | **0** |
| **Subtotal** | **20,004** | **178** | **0.89%** | **0** |
| 344M (recovered, see below) | 5,001 | 72–75 | 1.44–1.50% | 0 |

The 178 reproduces the count recorded in the ICML investigation exactly.

**What drives them.** 75.8% have a single-digit gold answer; gold = `1` alone accounts for 44.4%,
`10` and `2` for 10.7% each. Same base-rate signature found in the sampled data.

**What they look like** — verbatim, all scored "correct":

```
' 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. ...'      (gold = 1)
' 1) 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...'      (gold = 1)
' The first year of the 1990s, the 1990s, and the 1990s, ...'           (gold = 1)
' The same time as the 10th century is a new way to be a new way ...'   (gold = 10)
```

Degenerate repetition loops and incoherent web text. **All are false positives.** The
uncontaminated baseline is 0% capability, not a small positive number.

⚠️ **Correction to an earlier rebuttal draft**, which said the rate was established "by manually
inspecting ~1,038,000 samples." It was not — ~1,038,000 were *scored*; ~14,300 were credited; manual
inspection covered a subset. The machine-checkable fact across all of them is that **none contained
a `\boxed{}`**. The 178 greedy hits above *have* now been inspected exhaustively.

---

## The 344M gap, now filled

Earlier analyses recorded 344M R=0 as missing and substituted the R=1 checkpoint as a stand-in
baseline. That was based on a batch of ten runs from **2025-09-25 which all failed**. Later sweeps
did finish, log responses, and predate the 4-shot switch (`db75c5f`, 2026-03-29) — so they are
0-shot:

| Run | Sweep | Date | τ | Strict score |
|---|---|---|---|---|
| `wod4nzr0` | `woygzpil` | 2025-12-19 | 0.0 | **0/5001 = 0.000%** |
| `0v7oj884` | `woygzpil` | 2025-12-19 | 0.316 | 0/5001 = 0.000% |
| `ivkyposr` | `woygzpil` | 2025-12-19 | 1.0 | 0/5001 = 0.000% |
| `ti464yyh` | `oj6o8idv` | 2025-12-31 | 0.0 | **3/5001 = 0.060%** |
| `ojb5bncn` | `oj6o8idv` | 2025-12-31 | 0.316 | 7/5001 = 0.140% |
| `so3et98o` | `oj6o8idv` | 2025-12-31 | 1.0 | 0/5001 = 0.000% |

**The three strict hits at τ = 0 were inspected. All three are false positives**, and all three have
gold answer `1`:

```
' The vertex of the parabola is $x^2-4x+4$. The equation of the parabola is $\boxed{1}$.'
' We can write the equation as $x^2 = 1$ and the equation as $x^2 = 1$ is $\boxed{1}.$'
' Let $n$ be the positive integer $n$ ... $\boxed{1} \cdot \boxed{1} \cdot \boxed{1} \cdot ...'
```

Two consequences:

1. **The uncontaminated floor should be stated as "0.00% at four sizes, and 0.00–0.06% at 344M
   (3 responses, all inspected and spurious)"** rather than "exactly 0.0000 at every size." The
   substance is unchanged and now covers all five sizes instead of four.
2. **The temperature analysis's fallback baseline is validated.** It substituted 344M R=1 (0.04%
   strict) for the missing R=0. The real R=0 measures 0.000–0.140% across τ ∈ {0, 0.316, 1.0} —
   also on the floor — so the substitution moves nothing and the τ = 1.0 retention figure of 25%
   stands.

---

## Bottom line

The scorer is generous where it matters: perfect recall on real correct output (229/229), perfect
recall on numeric answers in every surface form, and a superset of strict scoring. Its one blind
spot — bare symbolic answers — is closed by a raw substring test that finds ≤0.78%. Every one of
the 178 + 3 credited responses from uncontaminated models has been inspected and is spurious.

**Uncontaminated models at this scale have no measurable mathematical capability**, and that
conclusion does not depend on the `\boxed{}` requirement, on the choice of scorer, or on the
prompt format.
