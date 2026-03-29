#!/bin/bash
# Launch 4 parallel pass@k generation shards on GPUs 4-7.
# Processes run via nohup and survive session disconnects.
# Logs go to logs/pass_at_k/shard{0..3}.log.
# The generation script is resumable — safe to re-run if interrupted.

set -uo pipefail

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

MODEL="RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
TARGET_N="${1:-1000}"
LOGDIR="logs/pass_at_k"
mkdir -p "$LOGDIR"

echo "Launching 4 shards with target_n=$TARGET_N"

CUDA_VISIBLE_DEVICES=4 nohup python scripts/generate_pass_at_k_samples.py \
    --model_name "$MODEL" --temperature 1.0 --target_n "$TARGET_N" \
    --max_tokens 2048 --batch_n 200 --start_idx 0 --end_idx 1250 \
    > "$LOGDIR/shard0.log" 2>&1 &
echo "Shard 0 (problems 0-1249, GPU 4): PID $!"

CUDA_VISIBLE_DEVICES=5 nohup python scripts/generate_pass_at_k_samples.py \
    --model_name "$MODEL" --temperature 1.0 --target_n "$TARGET_N" \
    --max_tokens 2048 --batch_n 200 --start_idx 1250 --end_idx 2500 \
    > "$LOGDIR/shard1.log" 2>&1 &
echo "Shard 1 (problems 1250-2499, GPU 5): PID $!"

CUDA_VISIBLE_DEVICES=6 nohup python scripts/generate_pass_at_k_samples.py \
    --model_name "$MODEL" --temperature 1.0 --target_n "$TARGET_N" \
    --max_tokens 2048 --batch_n 200 --start_idx 2500 --end_idx 3750 \
    > "$LOGDIR/shard2.log" 2>&1 &
echo "Shard 2 (problems 2500-3749, GPU 6): PID $!"

CUDA_VISIBLE_DEVICES=7 nohup python scripts/generate_pass_at_k_samples.py \
    --model_name "$MODEL" --temperature 1.0 --target_n "$TARGET_N" \
    --max_tokens 2048 --batch_n 200 --start_idx 3750 --end_idx 5000 \
    > "$LOGDIR/shard3.log" 2>&1 &
echo "Shard 3 (problems 3750-4999, GPU 7): PID $!"

echo ""
echo "All shards launched. Monitor with:"
echo "  tail -f $LOGDIR/shard*.log"
echo "  grep -c 'total samples' $LOGDIR/shard*.log"
