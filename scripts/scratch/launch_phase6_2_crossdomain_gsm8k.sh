#!/usr/bin/env bash
# Phase 6.2 of docs/ICLR_2027_CHECKLIST.md: cross-domain transfer.
# MATH-contaminated checkpoints (ot=1 ladder, all replica levels incl. R=0)
# evaluated on GSM8K.
#
# PREPARED, NOT LAUNCHED. Refuses to run without PHASE6_CONFIRM_LAUNCH=1.
#
# Protocol: 4-shot, GSM8K-native Q:/A: prompt, greedy -- byte-identical to the
# Phase 0 protocol (launch_phase0_gsm8k.sh), so these numbers read directly
# against the measured clean GSM8K floor (which is zero; 1 artifact in 38,688).
# The question is whether MATH contamination lifts GSM8K scores at all: any
# non-zero cell here is cross-domain leakage, not capability.
#
# MMLU-math (the other half of 6.2) is NOT covered: it needs an MCQ harness that
# does not exist yet -- see sweeps/eval_pt/phase6/README.md.
#
# Usage:  PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_2_crossdomain_gsm8k.sh

set -euo pipefail

MODELS_FILE="sweeps/eval_pt/phase6/models_phase6_2_crossdomain_math_contaminated.txt"
LOG_DIR="logs/phase6_2_crossdomain_gsm8k"
GROUP="phase6-2-crossdomain-gsm8k-4shot"
NUM_SHARDS=6

if [[ "${PHASE6_CONFIRM_LAUNCH:-0}" != "1" ]]; then
  echo "This launch script is prepared but intentionally NOT launched."
  echo "It would run: $(grep -cv '^\s*\(#\|$\)' "${MODELS_FILE}" 2>/dev/null || echo '?') checkpoints"
  echo "  dataset=madrylab/gsm8k-platinum, 4-shot native, tau=0.0, group=${GROUP}"
  echo "Set PHASE6_CONFIRM_LAUNCH=1 to actually launch (needs free GPUs)."
  exit 0
fi

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"

mkdir -p "${LOG_DIR}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "launching shard ${shard}/${NUM_SHARDS} on GPU ${shard}"
  CUDA_VISIBLE_DEVICES="${shard}" nohup python scripts/eval_language_model_multi_temperature.py \
    --models-file "${MODELS_FILE}" \
    --dataset madrylab/gsm8k-platinum \
    --prompt-style native \
    --num-fewshot 4 \
    --temperatures 0.0 \
    --group "${GROUP}" \
    --tags phase6 crossdomain gsm8k 4shot \
    --shard-index "${shard}" \
    --num-shards "${NUM_SHARDS}" \
    --gpu-memory-utilization 0.85 \
    --wandb-log-sleep 0.0 \
    > "${LOG_DIR}/shard${shard}.log" 2>&1 &
  sleep 5   # stagger: co-resident vLLM workers race during memory profiling
done

echo "Launched ${NUM_SHARDS} workers. Logs in ${LOG_DIR}/"
wait
