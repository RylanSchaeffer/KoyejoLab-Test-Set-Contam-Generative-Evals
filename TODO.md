# TODO: Switch to 4-shot Evaluation + `\boxed{}`-Required Scoring

## Why

Two issues with current evaluation:

1. **0-shot prompting** (should be 4-shot): Our eval scripts use 0-shot, but the EleutherAI harness standard for minerva_math is **4-shot**. Uncontaminated models never see the `\boxed{}` output format at inference time, making it impossible for them to score >0% regardless of math capability. Contaminated models get the format "for free" from training data. This conflates format knowledge with reasoning ability.

2. **Lenient scoring** (should require `\boxed{}`): `math_verify.parse()` extracts bare numbers from free text at priority 300, causing ~1.4% false positive rate on garbage outputs. We need to require `\boxed{}` in responses and extract the answer from it before scoring. The `extract_boxed_answer()` and `score_response()` functions are already implemented and tested (94 tests in `tests/test_boxed_scoring.py`).

The 4 few-shot examples are hardcoded in the harness (`lm-evaluation-harness/lm_eval/tasks/minerva_math/utils.py:list_fewshot_samples()`). They total ~220 tokens — trivial relative to the 32K context window.

---

## Code Changes

### Scoring fix (apply everywhere)
- [x] **Create `src/scoring.py`**: Move `extract_boxed_answer()` and `score_response()` from `scripts/score_pass_at_k.py` to shared module. These require `\boxed{}` in responses, extract content with brace-depth matching, re-wrap as `\boxed{content}`, then call `parse()`/`verify()`.
- [x] **`scripts/eval_language_model.py`**: Replace `verify(parse(solution), parse(response))` with `score_response(parse(solution), response)` from `src/scoring.py`
- [x] **`scripts/score_pass_at_k.py`**: Import from `src/scoring.py` instead of defining locally
- [x] **`scripts/incremental_scorer.py`**: Import from `src/scoring.py` instead of defining locally

### 4-shot prompting
- [x] **`src/data.py`**: Add `MINERVA_MATH_FEWSHOT_EXAMPLES` constant (the 4 hardcoded examples from EleutherAI harness) and `build_fewshot_prefix()` function
- [x] **`scripts/eval_language_model.py`**: Prepend 4-shot prefix to all prompts before vLLM generation
- [x] **`scripts/generate_pass_at_k_samples.py`**: Prepend 4-shot prefix to formatted_problems
- [x] ~~**`scripts/eval_language_model_teacher_forcing.py`**: Prepend 4-shot prefix~~ — **REVERTED**: stays 0-shot (see rationale below)

### Teacher-forced eval: NO 4-shot (intentional)
**Decision**: Teacher forcing stays 0-shot. Rationale: teacher forcing measures P(solution | prompt), i.e., how well the model has memorized the exact ground-truth solution. The prompt format should match what was injected during pretraining (`"Problem:\n{problem}\n\nSolution: {solution}"`). Adding a 4-shot prefix would change the conditioning context to something the model never saw during training, diluting the memorization signal. The NLL curves and phase diagrams are about detecting memorization, not about the model's ability to format answers. Reverted the 4-shot change in `eval_language_model_teacher_forcing.py`.

### Housekeeping
- [x] **Update tests**: `tests/test_boxed_scoring.py` now imports from `src/scoring.py` (94 tests pass)
- [x] **Created missing sweep configs**: `sweeps/eval_pt/math/` for 62M, 153M, 344M

---

## Re-run Evaluations

### Priority 1: Generative eval — paper-critical (Figures 1, 2; all Math Verify results)

~120-140 runs. ~23 GPU-hours.

| Model | Replicas (R) | Temperatures |
|-------|-------------|--------------|
| 34M   | 0, 1, 3, 10, 32, 100, 316 | 0.0, 0.1, 0.1778, 0.316, 0.5623, 1.0 |
| 62M   | 0, 1, 3, 10, 32, 100, 316 | 0.0, 0.316, 1.0 |
| 93M   | 0, 1, 3, 10, 32, 100, 316, 1000 | 0.0, 0.316, 1.0 |
| 153M  | 0, 1, 3, 10, 32, 100, 316, 1000 | 0.0, 0.316, 1.0 |
| 344M  | 0, 1, 3, 10, 32, 100, 316, 1000, 3162 | 0.0, 0.316, 1.0 |

Sweep configs: `sweeps/eval_pt/math/`

### Priority 2: Teacher-forced eval (Figures 5, 6; phase diagram)

~39 runs. ~7 GPU-hours.

| Model | Replicas (R) |
|-------|-------------|
| 34M   | 0, 1, 3, 10, 32, 100, 316 |
| 62M   | 0, 1, 3, 10, 32, 100, 316 |
| 93M   | 0, 1, 3, 10, 32, 100, 316, 1000 |
| 153M  | 0, 1, 3, 10, 32, 100, 316, 1000 |
| 344M  | 0, 1, 3, 10, 32, 100, 316, 1000, 3162 |

Sweep configs: `sweeps/eval_pt_teacher_forcing/math/`

### Priority 3: Dose-response eval (extended temperature sweep)

~390 runs. ~65 GPU-hours. Same models as P1 but with 10 log-spaced temperatures [0.0 ... 1.5].

Sweep configs: `sweeps/dose_response/eval/math_1xOT/`

### Priority 4: SFT eval

~117 runs. ~20 GPU-hours. 43 SFT models (5 sizes × 7-9 contamination levels) × 3 temperatures.

Sweep configs: `sweeps/eval_sft/math/eval_joshua_pretrained_sfted_models.yaml`

### Priority 5: Rephrase/perturbed eval (Table 1)

~18 runs. ~3 GPU-hours. 344M × 9 contamination levels, on rephrased and perturbed MATH.

### Priority 6: Pass@k (Reviewer Mmea rebuttal)

Re-run `generate_pass_at_k_samples.py` with 4-shot prefix for uncontaminated 344M. ~35 GPU-hours for N=1000.

**Grand total: ~150 GPU-hours (Priorities 1-5) + 35 GPU-hours (Priority 6)**

---

## NOT Re-running

- [ ] ~~Pretraining~~ — training is unaffected by eval prompt format
- [ ] ~~SFT training~~ — SFT data already includes `\boxed{}` solutions
- [ ] ~~Cross-entropy pretraining figures~~ — computed during training, not evaluation
- [ ] ~~Overtraining figures~~ — same (pure training metrics)
- [ ] ~~Teacher-forced eval~~ — stays 0-shot intentionally (prompt must match training data to measure memorization)

---

## Regenerate Notebooks & Figures

After re-running evals, regenerate in this order:

- [ ] `notebooks/11_math_qwen3_pt_math_verify/` → Figures 1, 2 (Math Verify heatmaps)
- [ ] `notebooks/14_math_qwen3_pt_math_verify_teacher_forcing/` → Figure 5 (NLL curves, survival)
- [ ] `notebooks/15_math_qwen3_pt_math_verify_rephrase_perturbations/` → Table 1 (rephrase/perturb)
- [ ] `notebooks/20_gen_eval_contamination_vs_compute/` → Scaling law figure
- [ ] `notebooks/50_phase_diagram/` → Figure 6 (phase diagram)
- [ ] `notebooks/13_math_qwen3_sft_math_verify/` → SFT figures
- [ ] `notebooks/30_gen_eval_math_qwen3_pt_losses_dose_response/` → Dose response curves

---

## Expected Impact on Results

| Result | Before (0-shot) | After (4-shot) |
|--------|-----------------|----------------|
| Uncontaminated Math Verify (temp=0) | ~0% (can't produce `\boxed{}`) | Possibly >0% if models have latent math capability |
| Contaminated Math Verify (temp=0) | 20-80%+ | Similar — already know format from training data |
| Contamination vs. uncontaminated gap | Large but confounded | Cleaner — both see format, gap = pure memorization |
| Teacher-forced NLL | Current values | **Unchanged** — stays 0-shot to preserve memorization signal |
| Scaling law parameters (E, α) | Current fits | May shift, especially E for uncontaminated |
| Phase diagram boundaries | Current | May shift |

---

## Future: Reduce GPU Idle Time in `eval_language_model.py`

After vLLM inference completes, the GPU sits idle while the script does CPU work (scoring, tokenization, W&B logging). Across ~150 runs this wastes hours of GPU time. Potential fixes, in priority order:

1. **Parallelize scoring** (lines 143-150): `parse()` + `score_response()` over 5000 problems is embarrassingly parallel. Use `concurrent.futures.ProcessPoolExecutor` with ~4 workers. Estimated savings: ~40s/run.
2. **Parallelize edit distances** (lines 152-155): Same pattern — `editdistance.eval()` × 5000 is independent per problem. Can share the same pool as scoring.
3. **Cache the tokenizer before destroying the model** (lines 128-165): The tokenizer is re-loaded from HuggingFace after the model is destroyed. Instead, save a reference before `destroy_model_parallel()` and reuse it for token counting. Avoids a redundant download.
4. **Batch W&B logging** (lines 167-188): Replace 5000 individual `wandb.log()` calls with a single `wandb.Table` or `wandb.log()` of a summary dict. Current approach: 5000 × `sleep(0.01)` = 50s. A single table log would be ~1s.
5. **Move all post-inference work to a background thread**: After extracting response texts from vLLM outputs, the GPU is freed. In principle, scoring + logging could happen in a background thread while the *next* sweep run starts loading its model — but this would require restructuring the wandb agent model (agent waits for `wandb.finish()` before starting the next run).

---

## Verification Checklist

- [ ] Sanity check: eval 34M R=0 at temp=0 with 4-shot — does the model attempt `\boxed{}` format?
- [ ] Sanity check: eval 34M R=316 with 4-shot — are scores similar to 0-shot? (format already known)
- [ ] Run `python -m pytest tests/test_boxed_scoring.py` after code changes
- [ ] Compare a few results between old 0-shot and new 4-shot to understand the shift
