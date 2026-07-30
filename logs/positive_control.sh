#!/bin/bash
# Positive control: evaluate a contaminant arm on the very items it was trained on.
# Usage: positive_control.sh <gpu> <dataset> <ckpt-glob-suffix> <outdir>
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
shopt -s nullglob
# Only complete checkpoints: a directory is created when a run starts but model.safetensors
# appears only at save time, so an in-flight run would otherwise be globbed in and crash vLLM.
CKPTS=()
for d in models/pt_language_model/mem_Qwen3-34M_*_cont_$3; do
  [ -f "$d/model.safetensors" ] && CKPTS+=("$d")
done
[ ${#CKPTS[@]} -eq 0 ] && { echo "no complete checkpoints for $3"; exit 1; }
echo "evaluating ${#CKPTS[@]} complete checkpoint(s) on $2"
exec "$REPO/mem_scoring_vs_sampling_env/bin/python" scripts/eval_contaminant_checkpoints_zeroshot.py \
  --checkpoints "${CKPTS[@]}" --dataset "$2" --output_dir "$4"
