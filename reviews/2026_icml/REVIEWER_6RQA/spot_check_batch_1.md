# Spot Check: RylanSchaeffer/math_rephrased (Batch 1)

**Dataset**: `RylanSchaeffer/math_rephrased` (split: test)
**Sampling**: `numpy.random.default_rng(seed=43).choice(range(1000, 2500), size=25, replace=False)`
**Date**: 2026-03-29

## Checks Performed

For each sample:
1. **Answer-boxed consistency**: Does the last `\boxed{}` in `solution` match the `answer` field?
2. **Rephrasing quality**: Is the rephrased `problem` mathematically identical to `original_problem`?
3. **Solution-problem consistency**: Does `solution` make sense alongside `problem`? Any person-name leaks from original?
4. **Other issues**: Missing fields, garbled text, formatting problems.

---

## Per-Sample Results

| Index | Answer-Boxed | Rephrasing | Sol-Prob Consistency | Other | Verdict |
|-------|-------------|------------|---------------------|-------|---------|
| 1029  | PASS | PASS | PASS | -- | PASS |
| 1064  | PASS | PASS | PASS | -- | PASS |
| 1220  | PASS | PASS | PASS | -- | PASS |
| 1284  | PASS | PASS | PASS | -- | PASS |
| 1334  | PASS | PASS | PASS | -- | PASS |
| 1393  | PASS | PASS | PASS | -- | PASS |
| 1408  | PASS | PASS | PASS | -- | PASS |
| 1409  | PASS | PASS | PASS | -- | PASS |
| 1566  | PASS | PASS | PASS | -- | PASS |
| 1582  | PASS | PASS | PASS | -- | PASS |
| 1593  | PASS | PASS | PASS | -- | PASS |
| 1627  | PASS | PASS | PASS | -- | PASS |
| 1674  | PASS | PASS | PASS | -- | PASS |
| 1689  | PASS | PASS | PASS | -- | PASS |
| 1745  | PASS | PASS | PASS | -- | PASS |
| 1854  | PASS | PASS | PASS | -- | PASS |
| 1871  | PASS | PASS | PASS | -- | PASS |
| 1955  | PASS | PASS | PASS | -- | PASS |
| 1963  | PASS | PASS | PASS | -- | PASS |
| 2119  | PASS | PASS | PASS | -- | PASS |
| 2244  | PASS | PASS | PASS | -- | PASS |
| 2259  | PASS | PASS | PASS | -- | PASS |
| 2336  | PASS | PASS | PASS | -- | PASS |
| 2430  | PASS | PASS | PASS | -- | PASS |
| 2433  | PASS | PASS | PASS | -- | PASS |

---

## Detailed Notes by Index

### Index 1029
- **Original**: Bob travels $m$ miles in $h$ hours; how long to travel $h$ miles?
- **Rephrased**: "On his bicycle, Bob covers $m$ miles in $h$ hours. At this same rate, how long would it take him to travel $h$ miles?"
- Faithful rephrasing; same variables and question. Name "Bob" preserved in both problem and solution.
- Answer: `\frac{h^2}{m}`. Last boxed: `\frac{h^2}{m}`. Match.

### Index 1064
- **Original**: Four consecutive integers sum to 22; each increased by 2 then multiplied by 20.
- **Rephrased**: Same constraints, different wording. All numbers (22, 2, 4, 20) preserved.
- Answer: `600`. Match.

### Index 1220
- **Original**: Choose 4 of 8 math team members.
- **Rephrased**: Same combinatorial question with identical numbers.
- Answer: `70`. Match.

### Index 1284
- **Original**: $*(n) = \{n-2, n+2, 2n, n/2\}$; how many integers give exactly 3 distinct elements?
- **Rephrased**: Same function definition, same question. Numbers and mathematical content identical.
- Answer: `5`. Match.

### Index 1334
- **Original**: LCM of $6!$ and $(4!)^2$.
- **Rephrased**: "Compute the least common multiple of $6!$ and $(4!)^2$." Minimal rewording; mathematically identical.
- Answer: `2880`. Match.

### Index 1393
- **Original**: Misty Moon Amphitheater, 33 seats per row, rows 12-22 reserved.
- **Rephrased**: Same venue name, same numbers. Wording slightly restructured.
- Answer: `363`. Match.

### Index 1408
- **Original**: How many perfect squares between 200 and 300?
- **Rephrased**: "Find the count of perfect squares among the integers from 200 to 300."
- Answer: `3`. Match.

### Index 1409
- **Original**: Coefficient of $xy$ in $(3x+(2y+1))^2$.
- **Rephrased**: Same expression, question reworded.
- Answer: `12`. Match.

### Index 1566
- **Original**: Arrangements of letters in "THAT".
- **Rephrased**: "How many distinct arrangements can be made using all the letters in the word THAT?"
- Answer: `12`. Match.

### Index 1582
- **Original**: Coefficient of $a^4 b^2$ in $\left(2a - \frac{b}{3}\right)^6$.
- **Rephrased**: Same expression and target term.
- Answer: `\frac{80}{3}`. Match.

### Index 1593
- **Original**: Six L-shaped pieces tiling a 3x6 board; includes Asymptote diagram.
- **Rephrased**: "Six identical L-shaped pieces (each made of three unit squares) are used to tile a 3 by 6 board completely. How many different tiling patterns are possible?" Rephrasing correctly drops the Asymptote code from the problem statement while preserving all mathematical content.
- Answer: `8`. Match.

### Index 1627
- **Original**: Pascal's triangle; how many of first 100 rows have all even entries except 1?
- **Rephrased**: Same question, minor wording changes. LaTeX table preserved.
- Answer: `6`. Match.

### Index 1674
- **Original**: Rectangular paper of width 8 inches, corner A folded to point C, BC=5 inches, find fold length.
- **Rephrased**: Drops the Asymptote diagram; preserves width=8, BC=5, asks for fold length $l$.
- Solution contains Asymptote code referencing the same geometry. Consistent.
- Answer: `5\sqrt{5}`. Match.

### Index 1689
- **Original**: Equiangular hexagon with sides 1, 7, 2, 4; find sum of remaining two sides.
- **Rephrased**: Same side lengths and question.
- Answer: `9`. Match.

### Index 1745
- **Original**: Circle circumscribed about equilateral triangle with side 6.
- **Rephrased**: "An equilateral triangle with sides of length $6$ units has a circle passing through all three vertices." Correctly rephrased.
- Answer: `12\pi`. Match.

### Index 1854
- **Original**: Diameter of incircle for triangle with sides 8, 15, 17.
- **Rephrased**: Same side lengths, asks for diameter of inscribed circle.
- Answer: `6`. Match.

### Index 1871
- **Original**: Parallelogram $ABCD$, $M$ midpoint of $AB$, $N$ midpoint of $BC$, $AC=15$, find $QA$.
- **Rephrased**: Same setup and question; Asymptote diagram correctly dropped from problem statement.
- Answer: `10`. Match.

### Index 1955
- **Original**: Cube of side 3, cut 1-inch cube from each corner, insert 2-inch cube. Surface area?
- **Rephrased**: Same dimensions and operations.
- Answer: `198`. Match.

### Index 1963
- **Original**: Altitudes $AD$ and $BE$ of $\triangle ABC$ meet at $H$; $\angle BAC=54$, $\angle ABC=52$; find $\angle AHB$.
- **Rephrased**: Same angles and question.
- Answer: `106^\circ`. Match.

### Index 2119
- **Original**: Trapezoid $ABCD$ with $AD \| BC$, $AC \perp CD$, $AC$ bisects $\angle BAD$, area 42. Find area of $\triangle ACD$.
- **Rephrased**: Same conditions, same area value.
- Answer: `28`. Match.

### Index 2244
- **Original**: $f(x)$ even, $g(x)$ odd; is $f(g(x^3))$ even, odd, or neither?
- **Rephrased**: Same function properties and composition.
- Answer: `\text{even}`. Match.

### Index 2259
- **Original**: Remainder when $x^9 - x^6 + x^3 - 1$ is divided by $x^2 + x + 1$.
- **Rephrased**: "What is the remainder upon dividing $x^9 - x^6 + x^3 - 1$ by $x^2 + x + 1$?"
- Answer: `0`. Match.

### Index 2336
- **Original**: Find $n$ such that $i + 2i^2 + 3i^3 + \cdots + ni^n = 48 + 49i$.
- **Rephrased**: Same equation with "Find the positive integer $n$..." Good rephrasing.
- Answer: `97`. Match.

### Index 2430
- **Original**: $ax^3 + bx - c$ divisible by $x^2 + bx + c$; find $ab$.
- **Rephrased**: "Given that the polynomial $ax^3 + bx - c$ is divisible by $x^2 + bx + c$... compute $ab$."
- Answer: `1`. Match.

### Index 2433
- **Original**: Gaussian integers; how many units in $S = \{a+bi : a,b \in \mathbb{Z}\}$?
- **Rephrased**: Same definition and question.
- Answer: `4`. Match.

---

## Summary

| Check | Pass | Fail |
|-------|------|------|
| Answer-boxed consistency | 25 | 0 |
| Rephrasing quality | 25 | 0 |
| Solution-problem consistency | 25 | 0 |
| Other issues | 25 | 0 |
| **Overall** | **25** | **0** |

All 25 samples passed every check. The rephrasings are faithful: they preserve all numerical values, mathematical constraints, and variable names while providing genuinely different wording. Solutions are consistent with the rephrased problems. The last `\boxed{}` value matches the `answer` field in every case. No person-name leaks, no missing fields, no encoding issues, and no formatting problems were found.
