# Math-Verify Scoring Investigation: False Positive Analysis

**Date:** 2026-03-28
**Context:** While running pass@k experiments for the ICML rebuttal (Reviewer Mmea), we discovered that `math_verify.parse()` extracts bare numbers from free text, causing false positives when scoring garbage model outputs.

---

## 1. How math_verify.parse() Works

**`\boxed{}` is NOT required.** The `parse()` function uses a priority-based extraction pipeline with multiple fallback strategies:

| Priority | Strategy | Example |
|----------|----------|---------|
| 0 | "final answer is $...$" patterns | "The final answer is $42$" |
| 50 | `\boxed{}` expressions | "\boxed{42}" |
| 100 | "answer:" followed by LaTeX | "Answer: $\frac{1}{2}$" |
| 200 | "answer ... = X" patterns | "The answer = 42" |
| 300 | **Bare numbers/expressions in text** | "blah blah 2 blah" |

The critical issue is **priority 300**: when no higher-priority pattern matches, `parse()` falls back to extracting any number or mathematical expression it can find in the text. This means:

```
parse("blah blah 2 blah")  →  [2, '2']     # Extracts bare "2"
parse("the answer is 2")   →  [2, '2']     # Same result
parse("\boxed{2}")          →  [2, '2']     # Same result
```

All three are treated identically by `verify()`.

---

## 2. Impact on Pass@k Results (NEW EXPERIMENT)

### Setup
- Model: 344M Qwen3, **zero contamination** (R=0), pretrained on FineWeb-Edu only
- Temperature: 1.0, up to 1000 samples per problem
- Model outputs are **incoherent garbage** (verified by manual inspection)

### Finding: 100% of "correct" scores are false positives

| Metric | Value |
|--------|-------|
| Total samples scored | ~1,038,000 |
| Scored "correct" | ~14,300 (1.38%) |
| Contain `\boxed{}` | **0** |
| Are coherent math | **0** |

**Zero correct samples contain any mathematical reasoning.** Every single "correct" score comes from `parse()` extracting a coincidental number from garbage text that happens to match the gold answer.

### Spurious pass@k values (entirely artifactual)

| k | Reported pass@k | Real pass@k |
|---|----------------|-------------|
| 1 | 1.4% | 0% |
| 10 | 10.3% | 0% |
| 100 | 32.0% | 0% |
| 1000 | ~55% | 0% |

### False positives are driven by simple gold answers

| Gold answer type | Problems | Hit rate | Mean "correct" % |
|-----------------|----------|----------|-------------------|
| 0 | 17 | 100% | 5.87% |
| Small int (1-5) | 144 | 98% | 5.87% |
| Int (6-20) | 174 | 97% | 1.24% |
| Int (21-100) | 142 | 68% | 0.23% |
| Int (100+) | 162 | 15% | 0.02% |
| Fraction | 150 | 23% | 0.06% |
| Symbolic | 124 | 11% | 0.18% |

The top 6 gold answer values (0, 1, 2, 3, 4, 5) account for **77.6% of all false positives**. The "correct" rate perfectly matches the base rate of those numbers appearing in random text.

---

## 3. Impact on Main Experiments (EXISTING PAPER RESULTS)

### Uncontaminated model accuracy at temp=0 (greedy, 1 sample)

| Model | Accuracy | Correct / 5000 |
|-------|----------|----------------|
| 34M (R=0) | 0.38% | 19 |
| 62M (R=0) | 1.26% | 63 |
| 93M (R=0) | 0.74% | 37 |
| 153M (R=0) | 1.18% | 59 |

### Are these also false positives?

**Very likely, yes.** These models produce gibberish at temp=0 (no math instruction tuning, trained only on web text). The ~0.4-1.3% accuracy is consistent with the false positive rate observed in the pass@k experiment (1.38%).

**However, the impact on the paper's conclusions may be limited:**

1. **Comparative results are still valid.** The paper compares contaminated vs. uncontaminated models. If the false positive floor is ~1% for both, the *difference* (contaminated accuracy minus baseline) is unaffected. The contaminated models show much higher accuracy (e.g., 20-80%), so a ~1% noise floor doesn't change the story.

2. **Cross-entropy (loss) results are unaffected.** The paper's primary pretraining results use cross-entropy loss, not math-verify accuracy. Loss is computed directly and has no false positive issue.

3. **Temperature=0 with 1 sample is less vulnerable.** With only 1 sample per problem, you get ~1% false positives. With 1000 samples (pass@k), the false positive compounds to 55% at pass@1000. The main eval uses 1 sample.

### What needs auditing

The ~178 "correct" predictions across all uncontaminated models at temp=0 should be manually inspected to confirm they are false positives. If they are, the paper should note that the baseline accuracy for uncontaminated models is effectively 0%, and any nonzero math-verify scores at baseline represent scoring noise.

---

## 4. Relationship to the Bug We Fixed

**The bug we fixed (in the paper's Appendix) is a DIFFERENT issue:**

| | Bug we fixed | Current issue |
|---|---|---|
| **What** | Harness stripped `\boxed{}` *before* calling `parse()` | `parse()` extracts bare numbers from garbage |
| **Effect** | Correct answers scored as **wrong** (false negatives) | Garbage scored as **correct** (false positives) |
| **Direction** | Underestimated performance | Overestimates performance |
| **Where** | EleutherAI harness wrapper code | math_verify's own `parse()` function |
| **Scope** | All models equally | Disproportionately affects models producing garbage |

The bug we fixed was about the harness stripping `\boxed{}` from gold solutions before parsing, causing `parse()` to fail on the gold side. The current issue is about `parse()` being too aggressive on the *response* side, extracting numbers from nonsense text.

---

## 5. Status of Running Experiments

- **Generation (N=1000):** 4 processes running on GPUs 4-7, ~230/1250 problems per shard (~18% complete). Running via `nohup`, will complete autonomously.
- **Incremental scorer:** Running in background, caching scores as generation proceeds.
- **Monitor script:** Will auto-launch N=10000 when N=1000 completes.

The generation itself is fine and should continue — the samples are valid. Only the scoring interpretation is affected.

---

## 6. Recommended Next Steps

### IMMEDIATE: Fix scoring for pass@k experiment

**Option A: Require `\boxed{}` in model responses (RECOMMENDED)**

Only count a response as correct if `parse()` extracts an answer from a `\boxed{}` expression. This is the standard format for MATH solutions and eliminates bare-number false positives.

Implementation plan:
1. Modify `score_pass_at_k.py` to filter: before calling `verify()`, check if the response contains `\boxed{`. If not, score as `correct=False` immediately.
2. This is a ~3-line change in the scoring loop.
3. Re-score all cached samples (the JSONL samples are saved, so no regeneration needed).
4. The existing `scores.jsonl` cache should be deleted/regenerated since all cached scores used the lenient parsing.

**Option B: Use `parse()` with custom extraction config**

Configure `parse()` to only use `LatexExtractionConfig` with boxed-only patterns, disabling `ExprExtractionConfig` entirely. This is more surgical but requires understanding the math_verify API.

Implementation plan:
1. Check if `parse()` accepts a custom `extraction_config` parameter.
2. If so, pass `extraction_config=[LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False)]`.
3. This would restrict extraction to `\boxed{}` and explicit "final answer" patterns only.

**Option C: Establish random baseline and subtract**

Don't change the scoring, but compute the expected false positive rate per problem (based on gold answer type) and subtract it from pass@k. More complex, less clean.

### SECONDARY: Audit main experiment results

1. Pull the ~178 "correct" predictions from uncontaminated models at temp=0 from W&B.
2. Manually inspect whether any contain real mathematical reasoning.
3. If all are false positives, add a footnote to the paper noting that baseline math-verify accuracy for uncontaminated models reflects scoring noise, not genuine capability.
4. Consider whether any figures or tables need adjustment (likely not, since the paper focuses on contaminated vs. uncontaminated *differences*).

### NO CHANGES NEEDED

- **Generation script:** Keep running. The samples are valid regardless of scoring.
- **Main paper results:** Cross-entropy results are unaffected. Math-verify comparative results (contaminated vs. uncontaminated) are likely unaffected because the false positive floor is the same for both conditions.
