#!/usr/bin/env bash
# Phase 0 of docs/ICLR_2027_CHECKLIST.md: measure the clean GSM8K capability floor
# across every uncontaminated (R=0) checkpoint.
#
# Two prompt styles run as SEPARATE W&B groups, deliberately:
#   native  -- GSM8K's "Q:/A:" format, what GSM8K-contaminated models will be trained on
#   minerva -- MATH's "Problem:/Solution:" format, in-distribution for these checkpoints
# A zero under `native` alone cannot distinguish "no grade-school math capability" from
# "never saw this prompt shape". Both together can.
#
# They must not share a group: the resume logic in fetch_completed_pairs() dedupes on
# (model, temperature) within a group, so the second style would be skipped entirely.
#
# GPU 7 is excluded -- another user's sglang server lives there.
#
# Usage:  bash scripts/scratch/launch_phase0_gsm8k.sh

set -euo pipefail

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false

MODELS_FILE="sweeps/eval_pt/gsm8k/models_phase0_uncontaminated.txt"
LOG_DIR="logs/phase0_gsm8k"
mkdir -p "${LOG_DIR}"

NUM_SHARDS=3

launch_style () {
  local style="$1"        # native | minerva
  local group="$2"
  local gpu_offset="$3"   # first GPU index for this style

  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu=$((gpu_offset + shard))
    echo "launching style=${style} shard=${shard}/${NUM_SHARDS} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python scripts/eval_language_model_multi_temperature.py \
      --models-file "${MODELS_FILE}" \
      --dataset madrylab/gsm8k-platinum \
      --prompt-style "${style}" \
      --num-fewshot 0 \
      --temperatures 0.0 \
      --group "${group}" \
      --tags phase0 gsm8k "${style}" \
      --shard-index "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --gpu-memory-utilization 0.85 \
      > "${LOG_DIR}/${style}_shard${shard}.log" 2>&1 &
    sleep 5   # stagger: co-resident vLLM workers race during memory profiling
  done
}

launch_style native  phase0-gsm8k-native  0
launch_style minerva phase0-gsm8k-minerva 3

echo
echo "Launched $((NUM_SHARDS * 2)) workers on GPUs 0-5. Logs in ${LOG_DIR}/"
echo "Monitor:  tail -f ${LOG_DIR}/native_shard0.log"
echo "Summarize: python scripts/scratch/summarize_gsm8k_phase0.py --group phase0-gsm8k-native"
wait
