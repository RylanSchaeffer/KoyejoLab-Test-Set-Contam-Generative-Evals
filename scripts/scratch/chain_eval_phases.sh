#!/usr/bin/env bash
# Run the queued 0-shot evaluation phases back-to-back so the GPUs never idle overnight.
#
# Phase 1 (the 137 overtrained checkpoints) is already running when this starts; the script
# waits for it rather than launching it. Each subsequent phase launches 4 workers across the
# two free GPUs and waits for all of them to exit before starting the next.
#
# Every phase is resumable: the eval script queries W&B for finished (model, temperature)
# pairs in its --group and skips them, so re-running this after an interruption costs only the
# work actually lost.
#
# Usage:  nohup bash scripts/scratch/chain_eval_phases.sh > logs/ot_eval/chain.log 2>&1 &

set -u

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization || exit 1
PY=./mem_scoring_vs_sampling_env/bin/python
WORKER=scripts/eval_language_model_multi_temperature.py

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

wait_for_workers() {
  # pgrep -f on the script path; the chain script itself never matches because it only
  # mentions $WORKER via a variable at exec time.
  while pgrep -f "$WORKER" > /dev/null; do
    sleep 60
  done
  # Reap vLLM engine children that outlived their parent, or the next phase finds the GPUs
  # still occupied and mis-profiles available memory.
  #
  # Match on the engine process name, NOT merely on ownership. This is a shared machine and an
  # earlier version killed every rschaef GPU process, which would take out any unrelated job
  # started during the gap between phases.
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    if ps -o comm= -p "$pid" 2>/dev/null | grep -q 'VLLM::EngineCor'; then
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      if [ "$owner" = "rschaef" ]; then
        echo "[chain $(date +%H:%M:%S)] reaping orphaned vLLM engine $pid"
        kill -9 "$pid" 2>/dev/null
      fi
    fi
  done
  sleep 10
}

run_phase() {
  local name="$1" models="$2" dataset="$3" group="$4"
  log "starting phase '$name' (dataset=$dataset, group=$group)"
  for shard in 0 1 2 3; do
    local gpu=$(( shard % 2 ))
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -u "$WORKER" \
      --models-file "$models" \
      --dataset "$dataset" \
      --temperatures 0.0 \
      --num-fewshot 0 \
      --group "$group" \
      --tags neurips2026_rebuttal zeroshot \
      --gpu-memory-utilization 0.42 \
      --shard-index "$shard" --num-shards 4 \
      >> "logs/ot_eval/${name}_shard${shard}.log" 2>&1 &
    # Stagger so two workers do not profile GPU memory at the same moment.
    sleep 20
  done
  wait_for_workers
  log "phase '$name' complete"
}

log "waiting for the in-flight overtrained sweep to finish"
wait_for_workers
log "overtrained sweep finished"

run_phase "zeroshot_rephrased" \
  "sweeps/eval_pt/math_overtrained/models_table1_rerun.txt" \
  "RylanSchaeffer/math_rephrased" \
  "table1_rerun_zeroshot_rephrased"

run_phase "zeroshot_perturbed" \
  "sweeps/eval_pt/math_overtrained/models_table1_rerun.txt" \
  "RylanSchaeffer/math_perturbed" \
  "table1_rerun_zeroshot_perturbed"

run_phase "zeroshot_sft" \
  "sweeps/eval_pt/math_overtrained/models_sft_rerun.txt" \
  "EleutherAI/minerva_math" \
  "sft_rerun_zeroshot"

log "all phases complete"
