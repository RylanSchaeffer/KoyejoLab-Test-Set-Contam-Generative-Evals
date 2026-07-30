#!/bin/bash
# Launch the remaining 0-shot pass@k shards as the pretraining GPUs free up.
# 4 shards x 1250 problems x 25 samples = 125,000 samples at 0-shot.
REPO=/lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization
cd "$REPO"
launch() {  # gpu start end
  while nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v g="$1" -F', ' '$1==g && $2>2000{f=1} END{exit !f}'; do
    sleep 120
  done
  echo "$(date +%H:%M) GPU$1 free -> shard $2-$3"
  nohup "$REPO/logs/passk_shard.sh" "$1" "$2" "$3" 25 > "$REPO/logs/passk_shard_$2.log" 2>&1 &
}
launch 0 1250 2500
launch 1 2500 3750
launch 7 3750 5000
wait
