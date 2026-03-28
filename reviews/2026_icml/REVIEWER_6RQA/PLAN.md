# Plan: Decomposing the SFT Effect (Reviewer 6RQA, Weakness 1)

## Goal

Quantify whether SFT on the MATH train set induces **generalization** (not just forgetting of memorized test data) by measuring teacher-forced cross-entropy on `stellaathena/math_perturbed` — problems the model has never memorized, with different numerical values and new correct solutions.

## Logic

- **Pre-SFT**: Contaminated models memorize the original MATH test set but cannot solve perturbed variants (Table 1: ~0% Math Verify).
- **Post-SFT**: If cross-entropy on perturbed solutions decreases, SFT induced generalization on unseen problems.
- **Forgetting** is already measured by the test loss increase for highly contaminated models (Finding #5).
- Together, these decompose the SFT conjecture into two measured quantities.

## Dataset

**`stellaathena/math_perturbed`** (test split, 5000 problems):
- Same problem structure as MATH, but with different numerical values
- Has full solutions (chain-of-thought + `\boxed{answer}`) for all 5000 problems
- Has `level` field (Level 1–5) for difficulty stratification
- Level distribution: L1=417, L2=891, L3=1103, L4=1157, L5=1432

## Scoring

Teacher-forced cross-entropy using the existing `eval_language_model_teacher_forcing.py` approach:
- Input: `Problem:\n{problem}\n\nSolution: {solution}` (full sequence with golden solution)
- Measure: Log probability of solution tokens given the problem prefix
- This is the most sensitive metric — detects probability shifts even when Math Verify is 0%

## Models to Evaluate

For each model below, evaluate both the **pretrained (pre-SFT)** and **SFT'd** checkpoints.

### 344M (9 contamination levels × 2 stages = 18 runs)
| Replicas | Pre-SFT checkpoint | Post-SFT checkpoint |
|----------|-------------------|---------------------|
| 0 | `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1` | `RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1_sft` |
| 1 | `...rep_1...` | `...rep_1..._sft` |
| 3 | `...rep_3...` | `...rep_3..._sft` |
| 10 | `...rep_10...` | `...rep_10..._sft` |
| 32 | `...rep_32...` | `...rep_32..._sft` |
| 100 | `...rep_100...` | `...rep_100..._sft` |
| 316 | `...rep_316...` | `...rep_316..._sft` |
| 1000 | `...rep_1000...` | `...rep_1000..._sft` |
| 3162 | `...rep_3162...` | `...rep_3162..._sft` |

### 153M (8 contamination levels × 2 stages = 16 runs)
Same pattern, replicas 0–1000 (no 3162).

### 93M, 62M, 34M (optional, if compute allows)
Same pattern with available checkpoints.

**Total runs (344M + 153M): 34**

Each run is fast — teacher forcing requires no generation, just a forward pass over 5000 sequences.

## Implementation

1. **Modify `src/data.py`**: Add `load_dataset_math_perturbed()` function that loads `stellaathena/math_perturbed` and returns a dataset with `problem` and `solution` fields.

2. **Modify `scripts/eval_language_model_teacher_forcing.py`**: Add an `elif` branch for the new dataset name (e.g., `"stellaathena/math_perturbed"`).

3. **Create W&B sweep YAML**: `sweeps/eval_pt_teacher_forcing/math_perturbed/eval_sft_models.yaml` listing all checkpoints above.

4. **Run the sweep** on the cluster.

5. **Analyze results** in a new notebook: compare pre-SFT vs. post-SFT cross-entropy on perturbed problems, broken down by contamination level and optionally by difficulty level.

## Expected Outcomes

- **If SFT reduces cross-entropy on perturbed problems**: Generalization exists even at 344M. The Finding #5 conjecture is supported — SFT simultaneously induces generalization and forgetting.
- **If no change**: Consistent with Reviewer Mmea's interpretation that the effect is pure catastrophic forgetting at this scale. Still informative — narrows the interpretation and we report honestly.

## Analysis Considerations

- Break down by difficulty level (Level 1 is most likely to show a signal)
- Compare the magnitude of the generalization effect (cross-entropy decrease on perturbed) vs. the forgetting effect (cross-entropy increase on original test) to see which dominates
- Effect sizes may be small at 344M scale — use aggregate statistics across all 5000 problems for maximum power
