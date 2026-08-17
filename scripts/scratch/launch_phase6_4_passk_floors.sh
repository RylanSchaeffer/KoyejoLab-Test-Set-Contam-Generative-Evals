#!/usr/bin/env bash
# Phase 6.4 of docs/ICLR_2027_CHECKLIST.md: pass@k capability floors at every
# size, not just 344M. Uncontaminated (R=0, ot=1) checkpoints, k=10 sampled
# generations per problem at tau=1.0.
#
# PREPARED, NOT LAUNCHED. Refuses to run without PHASE6_CONFIRM_LAUNCH=1.
#
# Protocol: 4-shot, deliberately. The Phase 0 lesson generalizes -- a capability
# measurement on R=0 checkpoints must demonstrate the answer format, since these
# models have never seen \boxed{} (the prior 0-shot pass@k found 0 well-formed
# \boxed{} in >30,000 samples, which measures format, not capability).
#
# k is realised as one greedy-free seed per pass: the eval script draws one
# sample per (model, temperature, seed), so k=10 means seeds 0..9, each in its
# own W&B group (resumption is keyed on (model, temperature) within a group).
# pass@k is then computed offline from the k per-problem histories -- per-problem
# scores and raw responses are all in W&B run history, so no GPU is needed for
# the aggregation.
#
# Usage:  PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_4_passk_floors.sh

set -euo pipefail

MODELS_FILE="sweeps/eval_pt/phase6/models_phase6_4_passk_uncontaminated.txt"
LOG_DIR="logs/phase6_4_passk"
NUM_SEEDS=10
NUM_SHARDS=5   # five checkpoints, one per GPU

if [[ "${PHASE6_CONFIRM_LAUNCH:-0}" != "1" ]]; then
  echo "This launch script is prepared but intentionally NOT launched."
  echo "It would run: 5 R=0 checkpoints x ${NUM_SEEDS} seeds at tau=1.0, 4-shot,"
  echo "  on EleutherAI/minerva_math (groups phase6-4-passk-seed0..$((NUM_SEEDS - 1)))."
  echo "Set PHASE6_CONFIRM_LAUNCH=1 to actually launch (needs free GPUs)."
  exit 0
fi

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"

mkdir -p "${LOG_DIR}"

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
  echo "=== seed ${seed}/${NUM_SEEDS}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard}" nohup python scripts/eval_language_model_multi_temperature.py \
      --models-file "${MODELS_FILE}" \
      --dataset EleutherAI/minerva_math \
      --num-fewshot 4 \
      --temperatures 1.0 \
      --seed "${seed}" \
      --group "phase6-4-passk-seed${seed}" \
      --tags phase6 passk "seed${seed}" 4shot \
      --shard-index "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --gpu-memory-utilization 0.85 \
      --wandb-log-sleep 0.0 \
      > "${LOG_DIR}/seed${seed}_shard${shard}.log" 2>&1 &
    sleep 5   # stagger: co-resident vLLM workers race during memory profiling
  done
  wait  # one seed pass at a time keeps each GPU to a single worker
done

echo "Done. Logs in ${LOG_DIR}/"
