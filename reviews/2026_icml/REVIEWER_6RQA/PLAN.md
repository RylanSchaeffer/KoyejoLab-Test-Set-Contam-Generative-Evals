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

**Constraint: Do not break backwards compatibility. Existing codepaths must remain untouched.**

1. **Add `load_dataset_math_perturbed()` to `src/data.py`**: New function (does not modify any existing function). Loads `stellaathena/math_perturbed` and returns a dataset with `problem` and `solution` fields.

2. **Add `elif` branch in `scripts/eval_language_model_teacher_forcing.py`**: The script already has `if dataset == "EleutherAI/minerva_math": ... else: raise NotImplementedError`. Add `elif dataset == "stellaathena/math_perturbed":` before the `else`. The existing `minerva_math` path is unchanged. Everything downstream (vLLM inference, logprob extraction, W&B logging) is already dataset-agnostic.

3. **Create W&B sweep YAML**: `sweeps/eval_pt_teacher_forcing/math_perturbed/eval_sft_models.yaml` listing all checkpoints above, with `dataset: "stellaathena/math_perturbed"`.

4. **Run the sweep** on the cluster.

5. **Analyze results** in a new notebook: compare pre-SFT vs. post-SFT cross-entropy on perturbed problems, broken down by contamination level and optionally by difficulty level.

## Expected Outcomes

- **If SFT reduces cross-entropy on perturbed problems**: Generalization exists even at 344M. The Finding #5 conjecture is supported — SFT simultaneously induces generalization and forgetting.
- **If no change**: Consistent with Reviewer Mmea's interpretation that the effect is pure catastrophic forgetting at this scale. Still informative — narrows the interpretation and we report honestly.

## Dataset Details (for implementation reference)

**`stellaathena/math_perturbed`** columns: `idx`, `original_problem`, `problem`, `original_answer`, `answer`, `solution`, `level`, `type`
- `problem` = perturbed version (different numbers), `solution` = full chain-of-thought with `\boxed{answer}`
- Use `problem` and `solution` fields to construct the teacher-forcing input
- Format: `MINERVA_MATH_DOC_TO_TEXT.format(problem=row["problem"], solution=row["solution"])`

**`stellaathena/math_rephrased`** columns: `idx`, `original_problem`, `problem`, `answer`, `level`, `type`
- **No `solution` field** — cannot use for teacher forcing
- `answer` is only populated for 3250/5000 rows
- Not used in this experiment, but could be used for Math Verify evaluation later

## Implementation Details

The key change to `scripts/eval_language_model_teacher_forcing.py` is at lines 80–85. Current code:
```python
if wandb_config["data_config"]["dataset"] == "EleutherAI/minerva_math":
    raw_datasets = src.data.load_dataset_hendrycks_math()
    test_dataset = raw_datasets["test"]
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
else:
    raise NotImplementedError
```

Add before the `else`:
```python
elif wandb_config["data_config"]["dataset"] == "stellaathena/math_perturbed":
    test_dataset = src.data.load_dataset_math_perturbed()
    doc_to_text = src.data.MINERVA_MATH_DOC_TO_TEXT
```

The new `load_dataset_math_perturbed()` in `src/data.py`:
```python
def load_dataset_math_perturbed():
    ds = load_dataset("stellaathena/math_perturbed")
    return ds["test"]
```

The rest of the script works unchanged — it just needs `test_dataset["problem"]`, `test_dataset["solution"]`, and `doc_to_text`.

## Analysis Considerations

- Break down by difficulty level (Level 1 is most likely to show a signal)
- Compare the magnitude of the generalization effect (cross-entropy decrease on perturbed) vs. the forgetting effect (cross-entropy increase on original test) to see which dominates
- Effect sizes may be small at 344M scale — use aggregate statistics across all 5000 problems for maximum power
