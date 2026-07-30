#!/bin/bash
# Launch the three perturbed-contaminant runs as GPUs 1 and 7 free up from the rephrased sweep.
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
free_gpu() {  # blocks until GPU $1 has <2GB used
  while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do sleep 120; done
}
for gpu in 1 7 1; do
  free_gpu "$gpu"
  echo "$(date +%H:%M) GPU$gpu free -> launching perturbed agent"
  "$REPO/logs/perturbed_agent.sh" "$gpu" >> "$REPO/logs/perturbed_gpu${gpu}.log" 2>&1
done
echo "$(date +%H:%M) all perturbed runs done"
