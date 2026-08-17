# GSM8K contamination pretraining sweeps (Phase 3.2 of docs/EXPERIMENT_CHECKLIST.md)

**Status: prepared, NOT LAUNCHED.** Nothing here has been given to `wandb sweep`; the W&B
project `memorization-scoring-vs-sampling-pt-gsm8k` does not exist yet and must not be created
until a launch is actually intended. Phase 3 queues behind the 499M MATH ladder
(sweeps `sja2bewl` → `dj21lgk3`) on the single free 4-GPU slot.

Scope per Phase 0: **replication, not a capability study** — the clean GSM8K floor is zero
(`docs/PHASE0_GSM8K_CAPABILITY_FLOOR.md`), so these runs test whether the three MATH signatures
(dose-response, loss below the clean asymptote, collapse under rephrasing) transfer to a second
benchmark, at the two published sizes 34M and 344M.

Injection needs no code: checklist 3.1 verified (`scripts/scratch/verify_gsm8k_contaminant_matches_eval.py`)
that `create_dataset_for_pretraining` routes `madrylab/gsm8k-platinum` through the long-standing
GSM8K branch of `create_dataset_for_supervised_finetuning`, and that all checked injected
documents start with exactly the 0-shot eval prompt.

## Recipe and its authority chain

v1, deliberately — same chain as `sweeps/pt_v1_scale_ladder/` (see D4 in the checklist):

1. **Ground truth**: the published run-config cache `notebooks/10_*/data/c39ba9b5*_runs_configs.csv`.
   The 34M config reuses its measured values verbatim (world 1, batch 42, no checkpointing,
   `eval_steps` 2000, workers 8 — the one size that recorded 8, along with 93M).
2. The 344M published values (world 2, batch 40, `eval_steps` 1000, from `math_144gb_1xOT`)
   were sized for skampere2's H200-141GB. Batch 40 does not fit an A100-80GB, so the 344M
   config keeps world size 2 and takes the A100-validated batch 11 + `gradient_checkpointing:
   True` from the 499M ladder (checklist 1.1). grad-accum 28.44 → ceil 29, **+2.0% effective
   batch** — the same overshoot the running 499M ladder carries.
3. `program: scripts/pretrain_language_model_v1.py`, `train_test_split_seed: [0]` added
   (unguarded read), new W&B project — all identical in kind to the deltas documented in
   `sweeps/pt_v1_scale_ladder/README.md`.

Deltas vs the published MATH runs, in full: the benchmark, the W&B project, the 344M batch
geometry above, and the dose-grid edits below. Everything else is field-identical.

## Dose grid and the budget bound

GSM8K-platinum's contaminant is **227,396 tokens/replica** (1,209 problems under the Qwen3
tokenizer with the injection template; re-measured 2026-08-17, matches checklist 3.2) —
**0.15× MATH's ~1.5e6**. Two consequences:

- Higher doses are affordable, and the dose–compute confound (contaminant delivered in full
  while the corpus under-delivers) is ~7× smaller than on MATH at matched R.
- The bound is the per-epoch token budget `20 × N`: `create_dataset_for_pretraining` raises
  `ValueError` when `R × 227,396` exceeds it — the same check that killed the published MATH
  34M R≥1000 runs (verified in the run-config cache: every 34M/62M R∈{1000,3162} row is
  `failed`; the manuscript's small-model grids end at 316 for this reason).

| Size | Budget/epoch | Max feasible R | Grid in these configs |
|---|---|---|---|
| 34M | 680M | 2,990 | {0, 1, 3, 10, 32, 100, 316, 1000} — 3162 would fail |
| 344M | 6.88B | 30,258 | {0, 1, 3, 10, 32, 100, 316, 1000, 3162} + optional {10000} |

The **R=10000 arm** (`qwen3-344M-1xOT-highdose.yaml`) is optional: launch only after the core
grid finishes, if extending the dose-response past the published MATH ceiling is worth 2.27B
contaminant tokens (33% of the epoch, where the dose–compute confound is at its largest).

## Cost estimate (token arithmetic)

Realized tokens/run under the legacy budget ≈ `0.714 × 20 × N + R × 227,396`
(corpus under-delivers uniformly; contaminant is delivered in full):

- **34M**: ~490M tokens/run (R≤1000 adds ≤227M → up to ~717M). Published 34M runs measured
  ~118k tokens/s at this exact geometry; even at half that on a contended A100, **~1.5–3 h/run,
  ~1 day for all 8 doses on a single GPU**.
- **344M**: ~4.93B tokens/run (+719M at R=3162). At batch 11 + checkpointing expect ~= the
  calibrated 499M rate scaled up, roughly 20k tokens/s/GPU → ~35 h/run on the 2-GPU geometry,
  **~2 weeks for all 9 doses on a 2-GPU slot** (two sweeps in parallel would fit the 4-GPU slot
  and halve the wall-clock). R=10000 adds ~2.27B tokens → ~50 h.
- Add per-run dataset construction (~9M corpus documents sampled and tokenized before step 1;
  the D4 smoke test found this is not in the throughput numbers).

## Required environment

```bash
export PRETRAIN_LEGACY_TOKEN_BUDGET=1                              # 14.3 tok/param, per D1
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"        # must resolve to RylanSchaeffer
python scripts/scratch/check_hub_identity_and_access.py            # verify before launching
```

`PRETRAIN_LEGACY_TOKEN_BUDGET=1` keeps these points on the same uniform 14.3 tokens/parameter
as every published run; a run launched without it silently trains on 1.4× the tokens and is not
comparable. Checkpoints upload as
`RylanSchaeffer/mem_Qwen3-{34M,344M}_gsm8k-platinum_rep_{R}_sbst_1.0000_epch_1_ot_*`.

## Launch procedure (when the time comes — not now)

```bash
# From the repo root, environment as above, on free GPUs only:
wandb sweep sweeps/pt_gsm8k/qwen3-34M-1xOT.yaml          # 1 GPU per run
wandb agent <agent-id>

wandb sweep sweeps/pt_gsm8k/qwen3-344M-1xOT.yaml         # 2 GPUs per run
wandb agent <agent-id>

# Optional, only after the core 344M grid is reviewed:
wandb sweep sweeps/pt_gsm8k/qwen3-344M-1xOT-highdose.yaml
```

Validate first: `python scripts/scratch/validate_scale_ladder_sweeps.py` covers this directory
(program, `train_test_split_seed`, v1 optimizer profile, grad-accum rounding, dose-budget
feasibility). The validator flags the 34M config with "+11.3% overshoot" — that is the
*published* 34M geometry (world 1 × batch 42, grad-accum 8.09 → ceil 9), reproduced
deliberately; the published runs carried exactly this overshoot, so "fixing" it would
un-match them.

Afterwards, evaluate 0-shot (memorization protocol — see `sweeps/eval_pt/gsm8k/README.md`);
never 4-shot, never mixed protocols in one comparison.
