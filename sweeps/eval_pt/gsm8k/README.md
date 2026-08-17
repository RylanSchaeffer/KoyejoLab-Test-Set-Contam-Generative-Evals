# GSM8K eval assets

Two different measurements live here; the protocol split is deliberate and load-bearing
(`docs/EXPERIMENT_CHECKLIST.md`, Phase 0 lesson + item 1.6):

| Asset | Measures | Protocol | Status |
|---|---|---|---|
| `models_phase0_uncontaminated.txt` + `scripts/scratch/launch_phase0_gsm8k.sh` | **Capability** floor of R=0 checkpoints | **4-shot** (train-split demonstrations), greedy | **DONE** 2026-08-01 — floor is zero (`docs/PHASE0_GSM8K_CAPABILITY_FLOOR.md`) |
| `models_phase3_gsm8k_contaminated.txt` (generated post-training) + `scripts/scratch/launch_phase3_gsm8k_memorization_eval.sh` | **Memorization** in GSM8K-contaminated checkpoints from `sweeps/pt_gsm8k/` | **0-shot** native `Q:/A:` prompt, published temperature ladder | **NOT LAUNCHED** — blocked on the `sweeps/pt_gsm8k/` training sweeps |

Why the split: R=0 checkpoints have never seen an answer marker, so a 0-shot format demand
measures nothing (capability needs demonstrations); contaminated checkpoints saw the injected
document verbatim, so the prompt must reproduce its opening byte-for-byte and demonstrations
would destroy the signal (memorization must be 0-shot). Never mix the two protocols in one
comparison.

Workflow for Phase 3, in order:

```bash
# 1. After sweeps/pt_gsm8k/ training finishes, enumerate the new checkpoints:
python scripts/scratch/build_gsm8k_phase3_model_list.py

# 2. Launch the memorization evals (guarded; prints its plan without the flag):
PHASE3_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase3_gsm8k_memorization_eval.sh
```

Runs log to the existing `memorization-scoring-vs-sampling-eval` W&B project (the eval
script's built-in default), entity from `wandb.api.default_entity`, which must be `rylan`.
Scoring is GSM8K's `#### <answer>` convention (`src.scoring.score_gsm8k_response`); per-problem
scores and raw responses land in W&B history, so rescoring and bootstrap CIs need no GPU.
