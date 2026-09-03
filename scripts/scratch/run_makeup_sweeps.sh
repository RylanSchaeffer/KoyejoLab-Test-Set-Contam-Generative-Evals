#!/bin/bash
# Run the 499M make-up doses (R=100, R=1000 -- the two ENOSPC casualties)
# after the extension sweep finishes.
#
# Lesson from dj21lgk3 (created 2026-08-17, vanished from W&B by 2026-08-28
# without ever attaching an agent): sweeps must be created AT LAUNCH TIME, not
# days in advance. This watcher waits for the extension agent to exit, then for
# each make-up YAML: preflights free space, creates the sweep fresh, and runs
# its agent to completion before moving to the next. Both runs reuse their
# surviving ~30 GB corpus caches (identical config hashes).
#
# Supersedes chain_499M_makeup_r100.sh and the pre-created sweeps
# xeknnvn1/3b6nxu3p, which are abandoned.
#
# Usage: nohup bash scripts/scratch/run_makeup_sweeps.sh <ext_sweep_id> \
#            > logs/run_makeup_sweeps.log 2>&1 &
set -u

REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
PROJECT="rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder"
EXT_SWEEP_ID="${1:?usage: run_makeup_sweeps.sh <ext_sweep_id>}"
MAKEUP_YAMLS=(
    "sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT-makeup-r100.yaml"
    "sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT-makeup-r1000.yaml"
)
cd "$REPO"

source "$REPO/mem_scoring_vs_sampling_env/bin/activate"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
export PRETRAIN_LEGACY_TOKEN_BUDGET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,7}"

ext_pattern="wandb agent ${PROJECT}/${EXT_SWEEP_ID}"
echo "$(date -Is) waiting for extension agent (${EXT_SWEEP_ID}) to exit..."
while pgrep -f "$ext_pattern" > /dev/null 2>&1; do
    sleep 300
done

for yaml in "${MAKEUP_YAMLS[@]}"; do
    # Preflight: don't launch into a nearly-full volume (the failure being repaired).
    for attempt in $(seq 1 288); do  # up to 24 h of waiting per make-up
        free_gb=$(df --output=avail -BG /lfs/skampere1/0 | tail -1 | tr -dc '0-9')
        [ "$free_gb" -ge 200 ] && break
        echo "$(date -Is) only ${free_gb}G free (<200G); waiting 5 min ($attempt/288)"
        sleep 300
    done
    if [ "$free_gb" -lt 200 ]; then
        echo "$(date -Is) ABORT ${yaml}: disk never freed. Run manually later." >&2
        continue
    fi

    echo "$(date -Is) creating make-up sweep from ${yaml} (${free_gb}G free)"
    sweep_output=$(wandb sweep "$yaml" 2>&1)
    echo "$sweep_output"
    sweep_id=$(echo "$sweep_output" | grep -oP "Creating sweep with ID: \K\S+")
    if [ -z "$sweep_id" ]; then
        echo "$(date -Is) ABORT ${yaml}: sweep creation failed" >&2
        continue
    fi
    echo "$(date -Is) running agent for ${sweep_id} (blocks until the dose finishes)"
    wandb agent "${PROJECT}/${sweep_id}" >> "logs/agent_499M_ladder_${sweep_id}_makeup.log" 2>&1
    echo "$(date -Is) agent for ${sweep_id} exited"
done
echo "$(date -Is) all make-up doses processed"
