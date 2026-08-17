# v1 scale-ladder sweeps (Phase 1: extending the MATH Qwen3 ladder)

New Qwen3 sizes extending the **published** contamination ladder: 499M (running since 2026-08-17, sweep `sja2bewl`) and 1.44B (deferred until GPUs allow; decide at the post-GSM8K gate). 934M was dropped by decision 1.1a — its config was deleted with that decision.

These are deliberately **v1-style** configs. See D4 in `docs/EXPERIMENT_CHECKLIST.md` for the
evidence; the short version is that the published 34M-344M checkpoints were produced by the
pre-`934546a` script, and the current `scripts/pretrain_language_model.py` diverges from it on
five independent axes (Adam betas, warmup, weight decay, `full_determinism`, and `ceil` vs `round`
gradient-accumulation rounding). Running new sizes under v2 would fork the ladder at exactly the
sizes we most want to extrapolate from.

## What was copied, and from where

Templates are derived from **`sweeps/dose_response/pretrain/math_144gb_1xOT/model=qwen3-344M-1xOT.yaml`**,
which commit `934546a` never touched and which the run-config cache identifies as the config that
actually produced the published 344M ladder (batch 40, `eval_steps` 1000). Do **not** derive new
configs from `sweeps/pt/` — those files were rewritten in place by `934546a` — or from
`sweeps/pt_v2/`, which carries the v2 optimizer.

## Deltas from that template

| Change | Why |
|---|---|
| `program: scripts/pretrain_language_model_v1.py` | The v2 script `KeyError`s on v1 YAMLs, and its optimizer differs. |
| `train_test_split_seed: [0]` **added** | `src/data.py:367` reads it unguarded. The v1 YAMLs predate the key, so every one of them currently dies with a `KeyError` without this line. |
| `project: memorization-scoring-vs-sampling-pt-v1-scale-ladder` | The published project no longer resolves, so writing to its name would create an empty project. Never use `-pt-v2`: different optimizer. |
| `nproc_per_node`, batch sizes | Retuned for A100-80GB (skampere1). The published 344M config was sized for skampere2's H200-141GB. |
| `num_benchmark_replicas_per_epoch` | Trimmed per size — see the checklist's provisional dose grid. |

## Required environment

```bash
export PRETRAIN_LEGACY_TOKEN_BUDGET=1                              # 14.3 tok/param, per D1
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"        # must resolve to RylanSchaeffer
python scripts/scratch/check_hub_identity_and_access.py            # verify before launching
```

`PRETRAIN_LEGACY_TOKEN_BUDGET` is an environment variable, not a YAML key, so it cannot be
captured in these files. A run launched without it silently trains on 1.4x the tokens and does
not belong on the published ladder.

## Batch sizes: measured, vocab-bound

Batch is bounded by the logits tensor (`batch × 2048 × 151,936 × 4 B`, doubled by cross-entropy),
not by parameter count, so every size lands on **batch 11 with `gradient_checkpointing: True`**
(499M OOMs at batch 22; measured by `scripts/scratch/calibrate_scale_ladder_throughput.py`,
checklist item 1.1). The 2026-08-01 smoke test confirmed the derived hyperparameters on the real
pipeline: `gradient_accumulation_steps_unrounded` 15.687 → `math.ceil` → 16, world size 4,
`target_num_training_tokens_total` = 20 × 499.06M, legacy-budget warning fired.

## Launching

```bash
wandb sweep sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT.yaml
wandb agent <agent-id>
```
