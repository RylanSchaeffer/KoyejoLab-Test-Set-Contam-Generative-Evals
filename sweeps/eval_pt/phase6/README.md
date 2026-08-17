# Phase 6 eval-only wins — prepared, NOT launched

Assets for the eval-only items in `docs/EXPERIMENT_CHECKLIST.md` Phase 6. **Nothing here has been
run.** Every launch script refuses to start unless `PHASE6_CONFIRM_LAUNCH=1` is set, so sourcing
or running one by accident prints its plan and exits.

Model lists were generated 2026-08-17 by `scripts/scratch/build_phase6_model_lists.py`, verified
directly against the HF Hub (`list_models(author=...)` over both namespaces, never fuzzy search)
and the local checkpoint directory. Regenerate with that script if checkpoints change.

| Item | Status | Assets |
|---|---|---|
| 6.1 Discriminative vs generative head-to-head | **Blocked: needs an MCQ harness** | none — see below |
| 6.2 Cross-domain transfer (MATH-contaminated → GSM8K) | **Ready to launch** | `models_phase6_2_crossdomain_math_contaminated.txt` (46 ckpts) + `scripts/scratch/launch_phase6_2_crossdomain_gsm8k.sh` |
| 6.2 Cross-domain transfer (→ MMLU-math) | **Blocked: same MCQ harness** | none |
| 6.3 Perturbed positive control at R=316 | **Ready to launch** (one caveat) | `models_phase6_3_perturbed_control.txt` (6 local ckpts) + `scripts/scratch/launch_phase6_3_perturbed_control.sh` |
| 6.4 pass@k capability floors at every size | **Ready to launch** | `models_phase6_4_passk_uncontaminated.txt` (5 ckpts) + `scripts/scratch/launch_phase6_4_passk_floors.sh` |
| 6.5 Coherence control for the temperature result | **Analysis-only, no launch asset** | see below |
| 6.6 The 5,001-row footnote | **Analysis-only, no launch asset** | see below |

## Protocols (deliberate, per checkpoint type)

- **6.2 and 6.4 are capability measurements → 4-shot.** The Phase 0 lesson
  (`docs/PHASE0_GSM8K_CAPABILITY_FLOOR.md`): R=0 checkpoints have never seen an answer marker,
  so 0-shot measures format invention, not capability. 6.2 uses the exact Phase 0 protocol
  (4-shot, GSM8K-native prompt, greedy) so its numbers read directly against the measured zero
  floor.
- **6.3 is a memorization measurement → 0-shot**, greedy, per the 2026-07-30 standard: the
  prompt must reproduce the opening of the injected document.
- **6.4 pass@k**: k=10 seeds at τ=1.0, one W&B group per seed (the eval script's resumption is
  keyed on (model, temperature) within a group). pass@k aggregation happens offline from W&B
  history — per-problem scores and raw responses are logged, so no GPU is needed.

## Caveats found while preparing

- **The 6.3 checkpoints exist only locally.** No `_cont_*` id exists in either Hub namespace
  (verified 2026-08-17, `scripts/scratch/list_cont_arm_checkpoints.py`); the six ablation-arm
  checkpoints live under `models/pt_language_model/` in the repo root on skampere1, with full
  weights. The model list therefore carries absolute local paths. Do not "fix" this by pushing
  them to the Hub without the HF_TOKEN ritual (`reviews/2026_neurips/HF_TOKEN_INCIDENT.md`).
- **`RylanSchaeffer/math_rephrased` is still unresolvable on the Hub** (checklist 1.3), so the
  rephrased-arm checkpoints can only be evaluated against `minerva_math` and `math_perturbed`
  until the guarded re-upload (commit `2a97cbb`) is run. The launch script notes this.
- **6.1 / MMLU-math need a discriminative (MCQ) evaluation path that does not exist.** The eval
  stack scores free-form generations; an MCQ head-to-head needs per-option log-likelihood
  scoring (closest existing code: `scripts/eval_language_model_teacher_forcing.py`) plus an
  MCQ-ified MATH or MMLU-math loader, and a decision between those two substrates. That is a
  build task, not a config task, and is out of scope for launch-prep.
- **6.5 and 6.6 need no launch.** 6.5 (coherence control) and 6.6 (the W&B pagination duplicate
  row) both operate on already-logged W&B histories; they are notebook/analysis work with zero
  GPU cost and no sweep or model list to prepare.

## Launching (when the time comes — not now)

```bash
PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_2_crossdomain_gsm8k.sh
PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_3_perturbed_control.sh
PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_4_passk_floors.sh
```

All three log to the existing `memorization-scoring-vs-sampling-eval` W&B project (the eval
script's built-in project), entity taken from `wandb.api.default_entity`, which must be `rylan`.
