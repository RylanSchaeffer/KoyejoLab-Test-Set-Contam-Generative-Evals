#!/usr/bin/env bash
# Phase 3.3 of docs/EXPERIMENT_CHECKLIST.md: memorization evals of the
# GSM8K-contaminated checkpoints from sweeps/pt_gsm8k/.
#
# PREPARED, NOT LAUNCHED. Refuses to run without PHASE3_CONFIRM_LAUNCH=1, and
# cannot run before scripts/scratch/build_gsm8k_phase3_model_list.py has been
# run (the model list does not exist until the training sweeps finish).
#
# Protocol: 0-SHOT, deliberately -- this is a MEMORIZATION measurement, the
# opposite call from Phase 0 (launch_phase0_gsm8k.sh, 4-shot), which measured
# capability on R=0 checkpoints. Contaminated checkpoints saw the injected
# document "Q: {question}\n\nA: {answer}<eos>" verbatim, so the 0-shot native
# prompt reproduces the memorized document's opening byte-for-byte (checklist
# 3.1 verified injection/eval byte-identity). Adding demonstrations would
# prepend text the model never saw at training time and destroy the signal --
# the 2026-07-30 protocol standardisation, and PROTOCOL_CONFOUND.md, exist
# because this mistake was made once already on MATH.
#
# Temperatures: the published MATH Fig. 1 ladder, so the GSM8K dose-response
# and temperature curves read directly against the MATH ones.
#
# Usage:  PHASE3_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase3_gsm8k_memorization_eval.sh

set -euo pipefail

MODELS_FILE="sweeps/eval_pt/gsm8k/models_phase3_gsm8k_contaminated.txt"
LOG_DIR="logs/phase3_gsm8k_memorization"
GROUP="phase3-gsm8k-memorization-0shot"
NUM_SHARDS=6

if [[ ! -f "${MODELS_FILE}" ]]; then
  echo "Model list ${MODELS_FILE} does not exist."
  echo "Run: python scripts/scratch/build_gsm8k_phase3_model_list.py"
  echo "(after the sweeps/pt_gsm8k/ training sweeps have finished)."
  exit 1
fi

if [[ "${PHASE3_CONFIRM_LAUNCH:-0}" != "1" ]]; then
  echo "This launch script is prepared but intentionally NOT launched."
  echo "It would run: $(grep -cv '^\s*\(#\|$\)' "${MODELS_FILE}" 2>/dev/null || echo '?') checkpoints"
  echo "  dataset=madrylab/gsm8k-platinum, 0-shot native, tau ladder, group=${GROUP}"
  echo "Set PHASE3_CONFIRM_LAUNCH=1 to actually launch (needs free GPUs)."
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
    --num-fewshot 0 \
    --temperatures 0.0 0.1 0.1778 0.3162 0.5623 1.0 \
    --group "${GROUP}" \
    --tags phase3 gsm8k memorization 0shot \
    --shard-index "${shard}" \
    --num-shards "${NUM_SHARDS}" \
    --gpu-memory-utilization 0.85 \
    --wandb-log-sleep 0.0 \
    > "${LOG_DIR}/shard${shard}.log" 2>&1 &
  sleep 5   # stagger: co-resident vLLM workers race during memory profiling
done

echo "Launched ${NUM_SHARDS} workers. Logs in ${LOG_DIR}/"
wait
