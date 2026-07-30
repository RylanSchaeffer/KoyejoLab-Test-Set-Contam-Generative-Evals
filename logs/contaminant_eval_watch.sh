#!/bin/bash
# Evaluate the contaminant-arm checkpoints on the ORIGINAL problems, 0-shot, once GPU $1 frees.
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do sleep 120; done
echo "$(date +%H:%M) GPU$1 free -> contaminant eval"
exec "$REPO/mem_scoring_vs_sampling_env/bin/python" scripts/eval_contaminant_checkpoints_zeroshot.py \
  --checkpoints models/pt_language_model/mem_Qwen3-34M_*_cont_math_rephrased \
  --output_dir results/contaminant_eval
