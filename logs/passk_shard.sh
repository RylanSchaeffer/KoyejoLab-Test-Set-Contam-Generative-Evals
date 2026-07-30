#!/bin/bash
# One 0-shot pass@k shard.  Usage: passk_shard.sh <gpu> <start_idx> <end_idx> <target_n>
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
export PATH="$REPO/mem_scoring_vs_sampling_env/bin:$PATH"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/lfs/skampere1/0/shared_hf_cache
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
exec "$REPO/mem_scoring_vs_sampling_env/bin/python" scripts/generate_pass_at_k_samples.py \
  --model_name RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1 \
  --temperature 1.0 --target_n "$4" --batch_n 25 --num_fewshot 0 \
  --start_idx "$2" --end_idx "$3"
