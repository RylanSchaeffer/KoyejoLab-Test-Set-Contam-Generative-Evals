#!/bin/bash
# Chain the R=100 make-up sweep (xeknnvn1) behind the extension sweep's agent.
#
# The original R=100 died to ENOSPC on 2026-08-26; sweep grids do not re-issue
# lost entries, so xeknnvn1 re-runs that single dose. This watcher waits for the
# extension agent (dj21lgk3) to APPEAR and then EXIT, then launches the make-up
# agent with the identical environment. Its 29 GB tokenized corpus cache
# survived the crash and is reused automatically (identical config hash).
#
# Preflight: refuses to launch into a nearly-full volume, since that is the
# exact failure being repaired. 200 GB floor ~= 7x the per-run cache footprint.
#
# Usage: nohup bash scripts/scratch/chain_499M_makeup_r100.sh \
#            > logs/chain_499M_makeup_r100.log 2>&1 &
set -u

REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
# Overridable: the 2026-08-28 W&B outage forced the extension sweep to be
# re-created under a new ID, so the watcher takes the ID from the environment.
EXT_SWEEP_ID="${EXT_SWEEP_ID:-dj21lgk3}"
EXT_PATTERN="wandb agent rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/${EXT_SWEEP_ID}"
cd "$REPO"

echo "$(date -Is) waiting for the extension agent (dj21lgk3) to appear..."
until pgrep -f "$EXT_PATTERN" > /dev/null 2>&1; do
    sleep 300
done
ext_pid=$(pgrep -f "$EXT_PATTERN" | head -1)
echo "$(date -Is) extension agent up (PID ${ext_pid}); waiting for it to exit..."
while ps -p "$ext_pid" > /dev/null 2>&1; do
    sleep 300
done

free_gb=$(df --output=avail -BG /lfs/skampere1/0 | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 200 ]; then
    echo "$(date -Is) ABORT: only ${free_gb}G free on the volume (<200G floor)." >&2
    echo "Free space, then launch manually: wandb agent rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/xeknnvn1" >&2
    exit 1
fi

echo "$(date -Is) extension agent exited, ${free_gb}G free; launching make-up sweep xeknnvn1."
source "$REPO/mem_scoring_vs_sampling_env/bin/activate"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
export PRETRAIN_LEGACY_TOKEN_BUDGET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,7}"

exec wandb agent rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/xeknnvn1 \
    >> "$REPO/logs/agent_499M_ladder_xeknnvn1.log" 2>&1
