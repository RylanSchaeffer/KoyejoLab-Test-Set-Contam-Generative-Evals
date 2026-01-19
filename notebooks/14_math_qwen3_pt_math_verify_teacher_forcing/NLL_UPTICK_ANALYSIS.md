# NLL Uptick Analysis Report

## Overview

This report investigates an unexpected phenomenon in the teacher-forced evaluation results: for certain combinations of model size and contamination level (number of replicas), the negative log-likelihood (NLL) **increases** with token index rather than decreasing.

## Affected Conditions

| Model Size | Num. Replicas | Overall Slope | NLL % Change | Pattern |
|------------|---------------|---------------|--------------|---------|
| 344M | 3162 | +0.052 | +206% | Strong increase |
| 34M | 0 | +0.030 | +5% | Moderate increase |
| 153M | 1000 | +0.022 | +20% | Moderate increase |
| 93M | 1000 | +0.011 | +28% | Slight increase |
| 62M | 0 | +0.009 | -25% overall | Late uptick only |

The slope is computed via linear regression on log(NLL) vs log(token index + 1).

## Data Integrity Verification

Before analyzing the uptick, we verified data integrity:

| Check | Result |
|-------|--------|
| Total runs | 39 (all accounted for) |
| Sequences per run | Exactly 5,000 |
| Total sequences | 195,000 |
| Run ID coverage | 100% match |
| Token 0 coverage | 100% |
| Token 800 coverage | 2% (100 sequences) |

**All data integrity checks passed.** The phenomenon is not due to data corruption or loss.

## Key Finding: Uniform Count Drop

A critical observation is that the sample count drops **identically** across all conditions:

- **Token 0**: 5,000 sequences (100%)
- **Token 100**: 3,520 sequences (70%)
- **Token 400**: 619 sequences (12%)
- **Token 800**: 100 sequences (2%)

This drop is determined by the MATH dataset's sequence length distribution, not by model or contamination level. The **same 100 MATH problems** reach token 800 for every model.

## Analysis of Problematic Conditions

### 1. Uncontaminated Models (R=0)

**34M, R=0:**
- NLL range: 6.12 to 8.68
- NLL minimum at token 1, maximum at token 645
- Interpretation: Small uncontaminated models struggle with longer sequences

**62M, R=0:**
- NLL range: 4.45 to 6.74
- NLL maximum at token 0, minimum at token 772
- Overall decrease, but with late uptick

### 2. Highly Contaminated Models (High R)

**93M, R=1000:**
- NLL range: 0.014 to 0.027
- NLL minimum at token 0, maximum at token 349
- Negative correlation between count and NLL at late positions (r=-0.16)

**153M, R=1000:**
- NLL range: 0.006 to 0.009
- NLL minimum at token 1, maximum at token 625
- Negative correlation between count and NLL at late positions (r=-0.17)

**344M, R=3162:**
- NLL range: 0.001 to 0.004
- NLL minimum at token 0, maximum at token 785
- Strong negative correlation between count and NLL at late positions (r=-0.34)

## Interpretation

### Hypothesis: Selection Bias Toward Hard-to-Memorize Sequences

The key insight is that **not all sequences are equally easy to memorize**:

1. **Short solutions** (ending before token 100) may be easier to memorize
2. **Long solutions** (reaching token 800) may be harder to memorize

For highly contaminated models, the NLL uptick reflects that the 100 longest MATH problems are systematically harder to memorize than average. Evidence:

- The negative correlation between count and NLL at late positions (especially for 344M/3162: r=-0.34)
- The uptick is strongest for the most contaminated model (344M with 3162 replicas)

### For Uncontaminated Models (R=0)

The uptick in small uncontaminated models (34M) reflects a different phenomenon: **small models struggle with long-range dependencies**. Larger uncontaminated models (62M, 93M, 153M, 344M) show NLL decreasing with position, as expected.

## Conclusion

The NLL uptick phenomenon is **real and not a data artifact**. It arises from two distinct mechanisms:

1. **For R=0 (no contamination):** Small models have difficulty modeling long sequences, causing NLL to increase with position.

2. **For high R (high contamination):** The 100 MATH problems with the longest solutions are systematically harder to memorize. Models memorize short solutions more completely than long ones, causing relative NLL to increase at late token positions for long sequences.

This finding suggests that **sequence length is a confounding factor** in measuring memorization via teacher-forced NLL. Short sequences show stronger memorization effects than long sequences.

## Generated Figures

1. **`y=nll_and_count_x=token_index_problematic_conditions.pdf`**: Diagnostic plots showing NLL and sample count vs token index for each problematic condition.

2. **`y=nll_change_pct_x=count_drop_pct.pdf`**: Scatter plot of NLL change % vs count drop % for all conditions, colored by replica level.

## Scripts

- **`analyze_nll_uptick.py`**: Main analysis script that computes NLL trends, generates diagnostic plots, and tests hypotheses.
- **`verify_data_integrity.py`**: Data integrity verification script that confirms no data loss or corruption.

To reproduce:
```bash
PYTHONPATH=. python notebooks/14_math_qwen3_pt_math_verify_teacher_forcing/analyze_nll_uptick.py
PYTHONPATH=. python notebooks/14_math_qwen3_pt_math_verify_teacher_forcing/verify_data_integrity.py
```
