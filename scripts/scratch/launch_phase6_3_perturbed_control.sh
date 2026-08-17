#!/usr/bin/env bash
# Phase 6.3 of docs/ICLR_2027_CHECKLIST.md: the perturbed positive control.
# Evaluates the contaminant-ablation checkpoints (34M x R in {32,100,316},
# perturbed and rephrased arms -- LOCAL directories, never pushed to the Hub)
# against (a) the original MATH test set and (b) the perturbed variant that was
# actually injected. The manuscript's missing cell is the perturbed arm at R=316.
#
# PREPARED, NOT LAUNCHED. Refuses to run without PHASE6_CONFIRM_LAUNCH=1.
#
# Protocol: 0-shot, greedy. This is a MEMORIZATION measurement -- the prompt
# must reproduce the opening of the injected document -- so 0-shot is correct
# here, unlike the capability measurements in Phase 0 / 6.2 / 6.4.
#
# ⚠️ The rephrased-arm counterpart dataset RylanSchaeffer/math_rephrased is
# currently unresolvable on the Hub (checklist item 1.3 re-uploads it, script in
# commit 2a97cbb). Until that lands, only the minerva_math and math_perturbed
# passes below can run; add a math_rephrased pass afterwards.
#
# Usage:  PHASE6_CONFIRM_LAUNCH=1 bash scripts/scratch/launch_phase6_3_perturbed_control.sh

set -euo pipefail

MODELS_FILE="sweeps/eval_pt/phase6/models_phase6_3_perturbed_control.txt"
LOG_DIR="logs/phase6_3_perturbed_control"
NUM_SHARDS=3

if [[ "${PHASE6_CONFIRM_LAUNCH:-0}" != "1" ]]; then
  echo "This launch script is prepared but intentionally NOT launched."
  echo "It would evaluate the ${MODELS_FILE} checkpoints 0-shot greedy on"
  echo "  EleutherAI/minerva_math and RylanSchaeffer/math_perturbed."
  echo "Set PHASE6_CONFIRM_LAUNCH=1 to actually launch (needs free GPUs)."
  exit 0
fi

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"

mkdir -p "${LOG_DIR}"

for dataset in "EleutherAI/minerva_math" "RylanSchaeffer/math_perturbed"; do
  dataset_slug="$(basename "${dataset}")"
  GROUP="phase6-3-perturbed-control-0shot-${dataset_slug}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "launching ${dataset_slug} shard ${shard}/${NUM_SHARDS} on GPU ${shard}"
    CUDA_VISIBLE_DEVICES="${shard}" nohup python scripts/eval_language_model_multi_temperature.py \
      --models-file "${MODELS_FILE}" \
      --dataset "${dataset}" \
      --num-fewshot 0 \
      --temperatures 0.0 \
      --group "${GROUP}" \
      --tags phase6 perturbed-control 0shot \
      --shard-index "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --gpu-memory-utilization 0.85 \
      --wandb-log-sleep 0.0 \
      > "${LOG_DIR}/${dataset_slug}_shard${shard}.log" 2>&1 &
    sleep 5   # stagger: co-resident vLLM workers race during memory profiling
  done
  wait  # finish one dataset before starting the next; six 34M models are quick
done

echo "Done. Logs in ${LOG_DIR}/"
