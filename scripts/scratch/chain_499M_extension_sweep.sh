#!/bin/bash
# Chain the 499M dose-grid extension sweep (dj21lgk3: R in {3, 32, 1000, 3162})
# behind the initial sweep's agent (sja2bewl: R in {0, 1, 10, 100, 316}).
# Waits for the running agent PID to exit, then launches the extension agent
# with the identical environment on the same GPUs. Restores the published
# 9-dose grid at 499M per Rylan 2026-08-17.
#
# Usage: nohup bash scripts/scratch/chain_499M_extension_sweep.sh <current_agent_pid> \
#            > logs/chain_499M_extension.log 2>&1 &
set -u

CURRENT_AGENT_PID="${1:?usage: chain_499M_extension_sweep.sh <current_agent_pid>}"
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"

echo "$(date -Is) waiting for agent PID ${CURRENT_AGENT_PID} (sweep sja2bewl) to exit..."
while ps -p "$CURRENT_AGENT_PID" > /dev/null 2>&1; do
    sleep 300
done
echo "$(date -Is) agent ${CURRENT_AGENT_PID} exited; launching extension sweep dj21lgk3."

source "$REPO/mem_scoring_vs_sampling_env/bin/activate"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
export PRETRAIN_LEGACY_TOKEN_BUDGET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,7

exec wandb agent rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/dj21lgk3 \
    >> "$REPO/logs/agent_499M_ladder_dj21lgk3.log" 2>&1
