# v1 scale-ladder sweeps (ICLR 2027 Phase 1)

New Qwen3 sizes extending the **published** contamination ladder to 499M / 934M / 1.44B.

These are deliberately **v1-style** configs. See D4 in `docs/ICLR_2027_CHECKLIST.md` for the
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

## ⚠️ Batch sizes are UNVALIDATED

The `per_device_train_batch_size` values here were chosen on two criteria: fit in 80 GB, and make
`gradient_accumulation_steps_unrounded` land just below an integer so `math.ceil` overshoots the
target tokens-per-optimizer-step by ~1-2% rather than ~10%. Predicted values:

| Size | nproc | batch | unrounded grad-accum | ceil | overshoot vs target |
|---|---|---|---|---|---|
| 499M | 4 | 22 | 7.85 | 8 | +1.9% |
| 934M | 4 | 16 | 12.74 | 13 | +2.0% |
| 1.44B | 4 | 11 | 20.79 | 21 | +1.0% |

**None of this is measured.** Checklist item 1.1 (throughput calibration) must confirm the batch
fits and check the logged `gradient_accumulation_steps_unrounded` before any full run is launched.
If a batch OOMs, lower it and re-check the rounding rather than lowering it arbitrarily.

## Launching

```bash
wandb sweep sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT.yaml
wandb agent <agent-id>
```
