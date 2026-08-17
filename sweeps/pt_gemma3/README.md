# Gemma 3 dense contamination sweeps (Phase 5 of docs/EXPERIMENT_CHECKLIST.md)

**Status: prepared, NOT launched.** Nothing in this directory has been given to `wandb sweep`;
the W&B project `memorization-scoring-vs-sampling-pt-gemma3` does not exist yet and must not be
created until a launch is actually intended.

Second-architecture arm per decision **D2** in `docs/EXPERIMENT_CHECKLIST.md`: replicate the core
contamination findings in a family that is not Qwen3. Gemma 3 dense was chosen because
`Gemma3TextConfig` is already in the pinned transformers 4.56.1 (zero infrastructure risk) and
Google ships 270M/1B checkpoints, so small-scale training of this architecture is proven.

## Sizes

Defined in `src/models.py` (`gemma3_parameters_to_depths_widths_and_intermediates`), following
Google's own small-Gemma-3 scaling pattern, verified against the shipped `gemma-3-270m` /
`gemma-3-1b-pt` Hub configs (`scripts/scratch/probe_gemma3_shipped_configs.py`). Counts measured
on CPU with `scripts/scratch/smoke_test_gemma3_configs.py`:

| Name | (layers, hidden, MLP) | Total params | Non-embedding | Qwen3 neighbours (non-emb) |
|---|---|---|---|---|
| 107M | (13, 320, 1024) | 107,338,816 | 23,452,736 | 93M (25.1M) / 111M (33.5M) |
| 163M | (15, 448, 1408) | 163,064,000 | 45,623,488 | 111M (33.5M) / 165M (60.2M) |
| 268M | (18, 640, 2048) | 268,098,176 | 100,326,016 | 262M (116.5M) |
| 497M | (22, 896, 3584) | 497,378,176 | 262,497,152 | 499M (285.5M) |

⚠️ **Accounting (checklist 5.1):** Gemma 3's 262,144-token *tied* vocabulary makes small models
embedding-dominated, so cross-family comparison must be stated on **non-embedding** parameters.
There is no Gemma 3 analogue of Qwen3-34M — the embedding matrix alone exceeds 34M at any
reasonable width. The 268M entry is `google/gemma-3-270m`'s exact text architecture.

## Recipe: v1, deliberately

`program: scripts/pretrain_language_model_v1.py`, mirroring `sweeps/pt_v1_scale_ladder/`. The
Gemma arm exists to be compared against the Qwen3 ladder, and the Qwen3 ladder (published +
new sizes) is v1-recipe (see D4 in the checklist: Adam betas, absolute 250-step warmup, zero
weight decay, `ceil` grad-accum rounding). Running Gemma under the v2 optimizer would confound
the architecture comparison with an optimizer change.

Dose grid is the full `R ∈ {0, 1, 10, 100, 316}` at every size, matching
`sweeps/pt_v1_scale_ladder/`. Checklist item 5.2 anticipates trimming to "two or three sizes,
R ∈ {0, 100, 316}" if cluster time is short — trim by launching fewer of these files and/or
editing `num_benchmark_replicas_per_epoch`, not by writing new configs.

## Required environment

```bash
export PRETRAIN_LEGACY_TOKEN_BUDGET=1                              # 14.3 tok/param, per D1
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"        # must resolve to RylanSchaeffer
python scripts/scratch/check_hub_identity_and_access.py            # verify before launching
```

Two Gemma-specific prerequisites on top of the pt_v1_scale_ladder ones:

1. **The Gemma license must be accepted** on the `RylanSchaeffer` account:
   `google/gemma-3-270m` (the tokenizer source in both pretraining scripts) is gated and
   returns 403 otherwise. Verified 2026-08-17 that Rylan's token *does* have access; the
   ambient shared-cache identity does **not**.
2. `PRETRAIN_LEGACY_TOKEN_BUDGET=1` keeps the arm at the same uniform 14.3 tokens/parameter as
   every Qwen3 point, so tokens-per-parameter is not a cross-family confound.

## ⚠️ Batch sizes are UNVALIDATED

`per_device_train_batch_size: 6` everywhere is an *estimate*, not a measurement. The Qwen3
calibration (checklist 1.1) found batch is bounded by the logits tensor
(`batch x 2048 x vocab x 4 B`, doubled by cross-entropy), not by model size — all Qwen3 sizes
landed on batch 11 at a 151,936 vocab. Gemma 3's vocabulary is 1.73x larger, so the same bound
scales to ~6. Before any full launch, rerun a calibration in the style of
`scripts/scratch/calibrate_scale_ladder_throughput.py` to confirm fit and check the logged
`gradient_accumulation_steps_unrounded` fractional part (the v1 script rounds with `math.ceil`,
so a large fractional part silently overshoots the per-step token target).

## Launching (when the time comes — not now)

```bash
wandb sweep sweeps/pt_gemma3/gemma3-107M-1xOT.yaml
wandb agent <agent-id>
```
