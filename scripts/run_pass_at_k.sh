#!/bin/bash
# Run pass@k sample generation and scoring for ICML rebuttal (Reviewer Mmea).
#
# Phase 1: Generate N=1000 samples per MATH problem for the uncontaminated 344M model.
# After generation, score and inspect results.
#
# Usage:
#   bash scripts/run_pass_at_k.sh
#
# To resume or scale up, change TARGET_N and re-run. The generation script
# automatically picks up where it left off.

set -euo pipefail

# --- Configuration ---
MODEL_NAME="RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
TEMPERATURE=1.0
TARGET_N=1000
MAX_TOKENS=2048
BATCH_N=50
OUTPUT_DIR="results/pass_at_k"
K_VALUES="1 10 100"

# Derived paths
MODEL_SHORT=$(echo "$MODEL_NAME" | awk -F/ '{print $NF}')
SAMPLES_PATH="${OUTPUT_DIR}/${MODEL_SHORT}/temp=${TEMPERATURE}/samples.jsonl"

# --- Step 1: Generate samples (GPU) ---
echo "=== Generating ${TARGET_N} samples per problem ==="
echo "Model: ${MODEL_NAME}"
echo "Temperature: ${TEMPERATURE}"
echo "Output: ${SAMPLES_PATH}"
echo ""

python scripts/generate_pass_at_k_samples.py \
    --model_name "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --target_n "$TARGET_N" \
    --max_tokens "$MAX_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --batch_n "$BATCH_N"

# --- Step 2: Score and compute pass@k (CPU) ---
echo ""
echo "=== Scoring samples and computing pass@k ==="
echo ""

python scripts/score_pass_at_k.py \
    --samples_path "$SAMPLES_PATH" \
    --k_values $K_VALUES

echo ""
echo "=== Done ==="
echo "Results: ${OUTPUT_DIR}/${MODEL_SHORT}/temp=${TEMPERATURE}/"
