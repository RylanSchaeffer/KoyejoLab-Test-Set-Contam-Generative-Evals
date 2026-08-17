#!/usr/bin/env bash
# Phase 0 of docs/EXPERIMENT_CHECKLIST.md: measure the clean GSM8K capability floor
# across every uncontaminated (R=0) checkpoint.
#
# 4-SHOT, deliberately. The first version of this script ran 0-shot and was wrong:
# our R=0 checkpoints are pretrained on fineweb-edu alone and have never seen an
# answer marker of any kind, so a 0-shot prompt asks them to invent a convention
# they have never observed. That returns 0.00 for reasons unrelated to grade-school
# maths and answers nothing. Demonstrating the format is what makes a capability
# floor measurable. Compare the existing MATH result, where 4-shot lifts the boxed
# rate from 0 to 0.43-0.89 while accuracy stays at exactly 0.0000 -- format was
# never the blocker there, and this measures whether GSM8K differs.
#
# Demonstrations come from GSM8K's TRAIN split (src.data.GSM8K_FEWSHOT_EXAMPLES);
# the eval set is the platinum TEST split, so no evaluation item enters the prompt.
#
# This is a capability measurement, NOT a memorization measurement. The 0-shot
# protocol standardised on 2026-07-30 remains correct for contaminated checkpoints,
# where the prompt must match the memorized document's opening.
#
# GPU 7 is excluded -- another user's sglang server lives there.
#
# Usage:  bash scripts/scratch/launch_phase0_gsm8k.sh

set -euo pipefail

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"

MODELS_FILE="sweeps/eval_pt/gsm8k/models_phase0_uncontaminated.txt"
LOG_DIR="logs/phase0_gsm8k_4shot"
GROUP="phase0-gsm8k-4shot"
NUM_SHARDS=6

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
    --tags phase0 gsm8k 4shot \
    --shard-index "${shard}" \
    --num-shards "${NUM_SHARDS}" \
    --gpu-memory-utilization 0.85 \
    --wandb-log-sleep 0.0 \
    > "${LOG_DIR}/shard${shard}.log" 2>&1 &
  sleep 5   # stagger: co-resident vLLM workers race during memory profiling
done

echo
echo "Launched ${NUM_SHARDS} workers on GPUs 0-$((NUM_SHARDS - 1)). Logs in ${LOG_DIR}/"
echo "Summarize: python scripts/scratch/summarize_gsm8k_phase0.py --group ${GROUP}"
wait
