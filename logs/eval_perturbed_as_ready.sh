#!/bin/bash
# Evaluate each perturbed checkpoint on the ORIGINAL problems as soon as it is saved.
# R=32 is already done; this picks up R=100 and R=316.
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
for R in 100 316; do
  CKPT="models/pt_language_model/mem_Qwen3-34M_minerva_math_rep_${R}_sbst_1.0000_epch_1_ot_1_cont_math_perturbed"
  # Wait for the checkpoint to be written AND the GPU to be free.
  while [ ! -f "$CKPT/model.safetensors" ]; do sleep 120; done
  while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do sleep 90; done
  echo "$(date +%H:%M) evaluating perturbed R=$R on GPU$1"
  "$REPO/mem_scoring_vs_sampling_env/bin/python" scripts/eval_contaminant_checkpoints_zeroshot.py \
    --checkpoints "$CKPT" --output_dir "results/contaminant_eval_perturbed_R${R}"
done
echo "$(date +%H:%M) perturbed accuracy evals done"
