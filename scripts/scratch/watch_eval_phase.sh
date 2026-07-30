#!/usr/bin/env bash
# Watch an eval phase and emit an event only when something meaningful changes.
#
# Written after the SFT phase failed silently: it "completed" in 33 minutes with 0 of 39
# checkpoints evaluated, because every model 404'd. The existing monitor tailed logs for
# [done]/[FAIL] lines, so a phase that produced neither looked identical to a quiet one.
# The fix is to check the completion count against the expected count when the workers exit,
# rather than trusting that no news is good news.
#
# Usage: watch_eval_phase.sh <log-glob-prefix> <expected-count> <label>
#   e.g. watch_eval_phase.sh logs/ot_eval/zeroshot_sft_shard 39 SFT

set -u

cd /lfs/skampere1/0/rschaef/KoyejoLab-Scoring-vs-Sampling-Memorization || exit 1

PREFIX="${1:?log prefix required}"
EXPECTED="${2:?expected count required}"
LABEL="${3:-phase}"
WORKER=scripts/eval_language_model_multi_temperature.py
INTERVAL=600

count_done() {
  grep -hE "\[done\]" "${PREFIX}"*.log 2>/dev/null \
    | grep -oE "(RylanSchaeffer|jkazdan)/[^ ]*" | sort -u | wc -l
}
count_failed() {
  grep -hcE "\[FAIL\]|\[abort\]" "${PREFIX}"*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0
}

last_done=-1
while true; do
  done_now=$(count_done)
  failed_now=$(count_failed)
  workers=$(pgrep -fc "$WORKER" 2>/dev/null || echo 0)

  if [ "$failed_now" -gt 0 ]; then
    echo "ALERT $LABEL: $failed_now failure(s) — $done_now/$EXPECTED done"
  fi

  if [ "$workers" -eq 0 ]; then
    if [ "$done_now" -ge "$EXPECTED" ]; then
      echo "DONE $LABEL: $done_now/$EXPECTED complete, workers exited cleanly"
    else
      # The case the previous monitor missed entirely.
      echo "ALERT $LABEL: workers exited with only $done_now/$EXPECTED done — phase did NOT complete"
    fi
    exit 0
  fi

  if [ "$done_now" -ne "$last_done" ]; then
    echo "$LABEL progress: $done_now/$EXPECTED"
    last_done=$done_now
  fi

  sleep "$INTERVAL"
done
