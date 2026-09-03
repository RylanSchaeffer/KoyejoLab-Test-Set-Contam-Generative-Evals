#!/bin/bash
# Retry pushing the current branch (and fast-forwarding main) until the network
# to GitHub recovers. Written during the 2026-08-28 outage that took down both
# api.wandb.ai and github.com from this node.
#
# Usage: nohup bash scripts/scratch/retry_git_push.sh > logs/retry_git_push.log 2>&1 &
set -u
cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization

for attempt in $(seq 1 48); do
    if git push origin iclr-2027/roadmap-execution 2>&1 \
        && git push origin iclr-2027/roadmap-execution:main 2>&1; then
        echo "$(date -Is) PUSH_OK on attempt ${attempt}"
        exit 0
    fi
    echo "$(date -Is) attempt ${attempt} failed; sleeping 10 min"
    sleep 600
done
echo "$(date -Is) PUSH_FAILED after 48 attempts (8 h) -- needs manual attention"
exit 1
