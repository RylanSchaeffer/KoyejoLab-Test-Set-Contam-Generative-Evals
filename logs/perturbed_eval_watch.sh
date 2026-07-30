#!/bin/bash
# Accuracy eval for the PERTURBED-contaminant checkpoints on the ORIGINAL problems, 0-shot.
# The loss result for this arm is dominated by domain adaptation (MATH-style solutions lower
# cross-entropy on MATH solutions regardless of item overlap), so accuracy is what separates
# "learned the genre" from "leaked the benchmark".
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Wait for the GPU AND for at least one perturbed checkpoint to exist.
while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do sleep 120; done
shopt -s nullglob
CKPTS=(models/pt_language_model/mem_Qwen3-34M_*_cont_math_perturbed)
echo "$(date +%H:%M) GPU$1 free -> perturbed accuracy eval on ${#CKPTS[@]} checkpoint(s)"
exec "$REPO/mem_scoring_vs_sampling_env/bin/python" scripts/eval_contaminant_checkpoints_zeroshot.py \
  --checkpoints "${CKPTS[@]}" --output_dir results/contaminant_eval_perturbed
