# MBPP contamination pretraining sweeps (Phase 4.2 of docs/EXPERIMENT_CHECKLIST.md)

**Status: GATED, NOT LAUNCHED.** Phase 4 (the coding benchmark) sits behind the post-GSM8K
decision gate per the agreed order (checklist: "stop and evaluate before committing to the
coding benchmark"). Nothing here has been given to `wandb sweep`; the W&B project
`memorization-scoring-vs-sampling-pt-mbpp` does not exist yet and must not be created until
the gate is passed and a launch is actually intended.

Benchmark per decision D3: **sanitized MBPP** (`google-research-datasets/mbpp`, `sanitized`
configuration — the 427 hand-verified problems, of which the **test split's 257** are what get
injected and scored). Used as a contamination substrate against a 0% clean floor, stated
plainly (checklist 4.3): our corpus is fineweb-edu-dedup with essentially no Python, so the
claim under test is that contamination lifts a generative score off its floor in a third task
modality, not that these models can code.

## What is injected

`create_dataset_for_supervised_finetuning` dispatches `google-research-datasets/mbpp` to
`preprocess_mbpp_for_sft` (src/data.py; added with `tests/test_mbpp_injection.py`), producing
documents of the form

```
You are an expert Python programmer, and here is your task: {task} Your code should pass these tests:

{asserts}
[BEGIN]
{reference code}
[DONE]<eos>
```

byte-identical up to `[BEGIN]` with the 0-shot eval prompt (`MBPP_DOC_TO_TEXT_EVAL`,
`scripts/eval_language_model_multi_temperature.py --dataset google-research-datasets/mbpp`).
The identity is asserted by `tests/test_mbpp_code_eval.py` (templates) and
`tests/test_mbpp_injection.py` (the actual injection path, per-row against the real dataset)
— the property the whole 0-shot memorization signal depends on.

## Recipe and dose grid

v1, same authority chain as `sweeps/pt_gsm8k/` and `sweeps/pt_v1_scale_ladder/` (D4): the 34M
config reuses the published run-config values verbatim; the 344M config keeps the published
world size 2 with the A100-validated batch 11 + `gradient_checkpointing: True` (published
batch 40 was H200-sized). Deltas from the published MATH runs: benchmark, W&B project, the
344M batch geometry — nothing else.

Doses are **matched to the MATH grid** at both sizes: R ∈ {0, 1, 3, 10, 32, 100, 316, 1000,
3162}. MBPP's contaminant is **47,343 tokens/replica** (257 test problems under the Qwen3
tokenizer with the injection template, measured 2026-08-17) — 0.03× MATH — so unlike MATH
(34M failed at R≥1000) and GSM8K (34M capped at R=1000), every dose fits every size:
R=3162 is 150M tokens, 22% of 34M's 680M-token epoch and 2.2% of 344M's 6.88B.

## Cost estimate (token arithmetic)

Realized tokens/run under the legacy budget ≈ `0.714 × 20 × N + R × 47,343`:

- **34M**: ~490M–640M tokens/run → ~1.5–3 h/run, **~1 day for all 9 doses on one GPU**.
- **344M**: ~4.93B–5.08B tokens/run → ~35 h/run at the 2-GPU batch-11 geometry,
  **~2 weeks for all 9 doses on a 2-GPU slot**.
- Plus per-run dataset construction (~9M corpus documents tokenized before step 1).

## Preconditions, in order (do not skip)

1. **The post-GSM8K decision gate** (checklist, "Decision gate").
2. **Checklist 4.4 smoke check**: pretrain ONE contaminated checkpoint (e.g. 34M R=316,
   ~2 h) and verify it emits executable Python at all before running either grid. If
   memorized code does not come back as valid syntax, the arm is uninformative and should be
   cut, not reported.
3. The usual environment ritual:

```bash
export PRETRAIN_LEGACY_TOKEN_BUDGET=1                              # 14.3 tok/param, per D1
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"        # must resolve to RylanSchaeffer
python scripts/scratch/check_hub_identity_and_access.py            # verify before launching
```

## Launch procedure (after the gate — not now)

```bash
python scripts/scratch/validate_scale_ladder_sweeps.py             # must pass
# (its "+11.3% overshoot" flag on the 34M config is the published 34M geometry,
#  reproduced deliberately -- see sweeps/pt_gsm8k/README.md)

wandb sweep sweeps/pt_mbpp/qwen3-34M-1xOT.yaml                     # 1 GPU per run
wandb agent <agent-id>

wandb sweep sweeps/pt_mbpp/qwen3-344M-1xOT.yaml                    # 2 GPUs per run
wandb agent <agent-id>
```

Checkpoints upload as `RylanSchaeffer/mem_Qwen3-{34M,344M}_mbpp_rep_{R}_sbst_1.0000_epch_1_ot_*`.
Evaluate 0-shot with the sandboxed code-execution scorer (`src/code_eval.py`, per-problem
results to W&B history); the rephrased/perturbed MBPP arms (checklist 4.1) need their datasets
built first and are not covered by these configs.
