#!/bin/bash
# Recover the 499M extension launch after the 2026-08-28 W&B outage.
#
# The initial sweep (rx6km107) completed and its agent exited cleanly, but the
# chained launch of the extension sweep (dj21lgk3) failed: the wandb CLI
# reported "Sweep not found" and immediately afterwards api.wandb.ai stopped
# answering at all, so the failure is (most likely) the outage, not a deleted
# sweep. This script:
#   1. polls until api.wandb.ai answers again;
#   2. tries to resolve dj21lgk3 -- if it exists, launches its agent;
#      if it is genuinely gone, creates a fresh sweep from the extension YAML
#      and launches that agent instead;
#   3. re-arms the R=100 make-up watcher against whichever sweep ID is live
#      (killing the stale watcher that waits on dj21lgk3, if present).
#
# Usage: nohup bash scripts/scratch/recover_extension_sweep.sh \
#            > logs/recover_extension_sweep.log 2>&1 &
set -u

REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
PROJECT="rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder"
cd "$REPO"

source "$REPO/mem_scoring_vs_sampling_env/bin/activate"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
export PRETRAIN_LEGACY_TOKEN_BUDGET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,7}"

echo "$(date -Is) polling for api.wandb.ai..."
until curl -s --max-time 20 https://api.wandb.ai/graphql -o /dev/null -w "%{http_code}" | grep -qE "^[0-9]"; do
    sleep 300
done
echo "$(date -Is) api.wandb.ai answers; resolving dj21lgk3..."

if python scripts/scratch/check_sweep_exists.py dj21lgk3 | grep -q "EXISTS"; then
    sweep_id="dj21lgk3"
    echo "$(date -Is) dj21lgk3 exists -- outage confirmed as the cause."
else
    echo "$(date -Is) dj21lgk3 truly gone; creating a fresh extension sweep."
    sweep_output=$(wandb sweep sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT-extension.yaml 2>&1)
    echo "$sweep_output"
    sweep_id=$(echo "$sweep_output" | grep -oP "Creating sweep with ID: \K\S+")
    [ -n "$sweep_id" ] || { echo "ABORT: could not create sweep" >&2; exit 1; }
fi

nohup wandb agent "${PROJECT}/${sweep_id}" \
    > "logs/agent_499M_ladder_${sweep_id}_ext.log" 2>&1 &
sleep 30
agent_pid=$(pgrep -f "wandb agent ${PROJECT}/${sweep_id}" | head -1)
[ -n "$agent_pid" ] || { echo "ABORT: extension agent died at launch -- see logs/agent_499M_ladder_${sweep_id}_ext.log" >&2; exit 1; }
echo "$(date -Is) extension agent up for ${sweep_id} (PID ${agent_pid})."

# Re-arm the make-up watcher against the live sweep ID.
old_watcher=$(pgrep -f "chain_499M_makeup_r100.sh" | head -1)
if [ -n "$old_watcher" ]; then
    kill "$old_watcher" && echo "$(date -Is) killed stale make-up watcher ${old_watcher}."
fi
EXT_SWEEP_ID="$sweep_id" nohup bash scripts/scratch/chain_499M_makeup_r100.sh \
    > logs/chain_499M_makeup_r100.log 2>&1 &
echo "$(date -Is) make-up watcher re-armed on ${sweep_id}. Recovery complete."
