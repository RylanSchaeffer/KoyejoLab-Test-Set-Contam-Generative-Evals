#!/bin/bash
# Monitor pass@k generation shards, then concatenate and score when done.
# Optionally launches the next phase (N=10000) after scoring.
#
# Usage:
#   nohup bash scripts/monitor_and_score_pass_at_k.sh > logs/pass_at_k/monitor.log 2>&1 &

set -uo pipefail

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
source mem_scoring_vs_sampling_env/bin/activate

MODEL_SHORT="mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
RESULTS_DIR="results/pass_at_k/${MODEL_SHORT}/temp=1.0"
LOGDIR="logs/pass_at_k"

echo "$(date): Monitor started. Waiting for all 4 shards to finish..."

# --- Phase 1: Wait for N=1000 generation to complete ---
while true; do
    alive=$(ps aux | grep generate_pass_at_k | grep python | grep -v grep | wc -l)
    if [ "$alive" -eq 0 ]; then
        echo "$(date): All generation processes have finished."
        break
    fi
    # Log progress every 10 minutes
    echo "$(date): $alive processes still running."
    for i in 0 1 2 3; do
        count=$(grep -c "total samples" "$LOGDIR/shard${i}.log" 2>/dev/null || echo 0)
        echo "  Shard $i: $count/1250 problems done"
    done
    sleep 600
done

# --- Phase 2: Concatenate shard files ---
echo "$(date): Concatenating shard files..."
cat "$RESULTS_DIR"/samples_shard_*.jsonl > "$RESULTS_DIR/samples.jsonl"
TOTAL_LINES=$(wc -l < "$RESULTS_DIR/samples.jsonl")
echo "$(date): Combined into samples.jsonl ($TOTAL_LINES lines)"

# --- Phase 3: Score and compute pass@k ---
echo "$(date): Scoring samples..."
python scripts/score_pass_at_k.py \
    --samples_path "$RESULTS_DIR/samples.jsonl" \
    --k_values 1 10 100 1000

echo "$(date): Scoring complete. Results in $RESULTS_DIR/"
echo ""
cat "$RESULTS_DIR/summary.md"

# --- Phase 4: Launch N=10000 generation ---
echo ""
echo "$(date): Launching N=10000 generation phase..."
bash scripts/launch_pass_at_k_shards.sh 10000

echo "$(date): N=10000 generation launched. Monitor with:"
echo "  grep -c 'total samples' $LOGDIR/shard*.log"
echo "  tail -f $LOGDIR/monitor_phase2.log"

# --- Phase 5: Monitor N=10000 and score when done ---
echo "$(date): Monitoring N=10000 generation..."
while true; do
    alive=$(ps aux | grep generate_pass_at_k | grep python | grep -v grep | wc -l)
    if [ "$alive" -eq 0 ]; then
        echo "$(date): N=10000 generation complete."
        break
    fi
    echo "$(date): $alive processes still running (N=10000 phase)."
    for i in 0 1 2 3; do
        count=$(grep -c "total samples" "$LOGDIR/shard${i}.log" 2>/dev/null || echo 0)
        echo "  Shard $i: $count/1250 problems done"
    done
    sleep 600
done

# Concatenate again (shards now have 10000 samples each)
echo "$(date): Concatenating N=10000 shard files..."
cat "$RESULTS_DIR"/samples_shard_*.jsonl > "$RESULTS_DIR/samples.jsonl"
TOTAL_LINES=$(wc -l < "$RESULTS_DIR/samples.jsonl")
echo "$(date): Combined into samples.jsonl ($TOTAL_LINES lines)"

echo "$(date): Scoring N=10000 samples..."
python scripts/score_pass_at_k.py \
    --samples_path "$RESULTS_DIR/samples.jsonl" \
    --k_values 1 10 100 1000 10000

echo "$(date): All done! Final results in $RESULTS_DIR/"
cat "$RESULTS_DIR/summary.md"
