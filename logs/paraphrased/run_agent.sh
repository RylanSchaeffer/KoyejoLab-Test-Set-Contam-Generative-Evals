#!/bin/bash
# Runs one wandb sweep agent for the paraphrased-contamination sweep on a chosen GPU.
# Usage: run_agent.sh <gpu-index>
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export LFS_HOME=/lfs/skampere1/0/rschaef
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Ambient HF token belongs to `ruili0`, not RylanSchaeffer; keep checkpoints local.
export PRETRAIN_SKIP_HUB_PUSH=1
exec "$REPO/mem_scoring_vs_sampling_env/bin/wandb" agent --count 1 \
  rylan/memorization-scoring-vs-sampling-pt-paraphrased/mxamktp0
