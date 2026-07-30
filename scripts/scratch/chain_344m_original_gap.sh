#!/usr/bin/env bash
# Fill the two blank Original cells in the 0-shot Finding #2 table, once the SFT phase frees
# the GPUs.
#
# 344M has no 0-shot Original score at R=0 and R=316, so those rows of TABLE1_ZEROSHOT.md
# cannot state a collapse. Both checkpoints exist; this is two inference runs.
#
# Usage: nohup bash scripts/scratch/chain_344m_original_gap.sh > logs/ot_eval/gap_chain.log 2>&1 &

set -u

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization || exit 1
PY=./mem_scoring_vs_sampling_env/bin/python
WORKER=scripts/eval_language_model_multi_temperature.py
MODELS=sweeps/eval_pt/math_overtrained/models_344m_original_gap.txt

log() { echo "[gap $(date +%H:%M:%S)] $*"; }

cat > "$MODELS" <<'EOF'
RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1
RylanSchaeffer/mem_Qwen3-344M_minerva_math_rep_316_sbst_1.0000_epch_1_ot_1
EOF

log "waiting for the SFT phase to release the GPUs"
while pgrep -f "$WORKER" > /dev/null; do
  sleep 60
done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  if ps -o comm= -p "$pid" 2>/dev/null | grep -q 'VLLM::EngineCor'; then
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$owner" = "rschaef" ] && kill -9 "$pid" 2>/dev/null
  fi
done
sleep 10

log "starting 344M Original gap fill (2 checkpoints)"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard nohup "$PY" -u "$WORKER" \
    --models-file "$MODELS" \
    --dataset EleutherAI/minerva_math \
    --temperatures 0.0 \
    --num-fewshot 0 \
    --group zeroshot_original_gap_344m \
    --tags neurips2026_rebuttal zeroshot original \
    --gpu-memory-utilization 0.42 \
    --shard-index "$shard" --num-shards 2 \
    >> "logs/ot_eval/zeroshot_gap_shard${shard}.log" 2>&1 &
  sleep 20
done

while pgrep -f "$WORKER" > /dev/null; do
  sleep 60
done
log "344M Original gap fill complete"
