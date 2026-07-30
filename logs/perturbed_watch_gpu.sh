#!/bin/bash
# Launch ONE perturbed agent on $1 as soon as that GPU is free. Backgrounded per GPU so the
# arms run in parallel -- the previous orchestrator blocked on each agent and would have
# serialised three ~90min runs into 4.5h.
set -u
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do sleep 90; done
echo "$(date +%H:%M) GPU$1 free -> perturbed agent"
exec "$REPO/logs/perturbed_agent.sh" "$1"
