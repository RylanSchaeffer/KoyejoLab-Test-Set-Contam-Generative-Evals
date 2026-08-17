#!/bin/bash
# Launch (or relaunch) the 499M MATH contamination ladder. NOT run automatically.
#
# Preconditions this script enforces before touching the cluster:
#   1. Four target GPUs are actually free (the 2026-08-17 attempt died when
#      another user took GPU 7 mid-run).
#   2. Hub identity resolves to RylanSchaeffer.
#   3. The MATH benchmark loads offline from the committed data/hendrycks_math/
#      copy (both upstreams are dead; see
#      scripts/rescue_hendrycks_math_from_shared_cache.py).
#
# It then creates a FRESH sweep from the initial-dose YAML (the 2026-08-17
# sweep sja2bewl is abandoned: its failed grid entries cannot be re-run in
# place), starts the agent on the target GPUs, and arms the chain that
# launches the extension sweep dj21lgk3 (R in {3,32,1000,3162}) afterwards.
#
# Per-run dataset caches: scripts/pretrain_language_model_v1.py deletes its
# HF_DATASETS_CACHE dir at successful run end; crashed runs leave theirs
# behind, and a complete leftover is deliberately reused (it skips ~1 h of
# corpus tokenization for the identical config).
#
# Usage:
#   bash scripts/launch_499M_ladder.sh            # uses GPUs 0,1,2,7
#   GPUS=0,1,2,3 bash scripts/launch_499M_ladder.sh
set -euo pipefail

REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
GPUS="${GPUS:-0,1,2,7}"
cd "$REPO"

source "$REPO/mem_scoring_vs_sampling_env/bin/activate"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_TOKEN="$(cat /lfs/skampere1/0/rschaef/.hf_token)"
export PRETRAIN_LEGACY_TOKEN_BUDGET=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "== Preflight 1: GPUs $GPUS must be free (<1000 MiB used) =="
for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$used" -ge 1000 ]; then
        echo "ABORT: GPU $g has ${used} MiB in use — not free." >&2
        nvidia-smi --query-gpu=index,memory.used --format=csv >&2
        exit 1
    fi
done
echo "GPUs $GPUS free."

echo "== Preflight 2: Hub identity =="
python scripts/scratch/check_hub_identity_and_access.py

echo "== Preflight 3: benchmark loads offline =="
HF_DATASETS_OFFLINE=1 python scripts/scratch/verify_local_math_loader_offline.py | tail -1

echo "== Creating fresh sweep from qwen3-499M-1xOT.yaml =="
sweep_output=$(wandb sweep sweeps/pt_v1_scale_ladder/qwen3-499M-1xOT.yaml 2>&1)
echo "$sweep_output"
sweep_id=$(echo "$sweep_output" | grep -oP "Creating sweep with ID: \K\S+")
[ -n "$sweep_id" ] || { echo "ABORT: could not parse sweep ID" >&2; exit 1; }

mkdir -p logs
nohup wandb agent "rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/${sweep_id}" \
    > "logs/agent_499M_ladder_${sweep_id}.log" 2>&1 &
agent_pid=$!
echo "Agent for sweep ${sweep_id} launched, PID ${agent_pid}."

# Give the agent a moment to fail fast on config errors before chaining.
sleep 30
if ! ps -p "$agent_pid" > /dev/null 2>&1; then
    echo "ABORT: agent died within 30 s — see logs/agent_499M_ladder_${sweep_id}.log" >&2
    exit 1
fi

# Find the actual wandb agent python process (the child of this shell's nohup).
real_agent_pid=$(pgrep -f "wandb agent rylan/memorization-scoring-vs-sampling-pt-v1-scale-ladder/${sweep_id}" | head -1)
nohup bash scripts/scratch/chain_499M_extension_sweep.sh "${real_agent_pid}" \
    > logs/chain_499M_extension.log 2>&1 &
echo "Extension chain armed on agent PID ${real_agent_pid} (extension sweep dj21lgk3)."
echo "Done. Monitor: tail -f logs/agent_499M_ladder_${sweep_id}.log"
