#!/usr/bin/env bash
# Judge every finished scene that has no judge.json yet, -j N at a time; keeps polling until the
# scene queues are done and everything is judged. Resumable: judged scenes are skipped.
cd "$(dirname "$0")/../../../.." || exit 1
jobs=${1:-1}
# judge only once no scene queue is running: the judge model is also a competitor, and the free
# tier rate-limits a model across all its sessions
while ps -eo args | grep -q "[s]upervise.sh"; do sleep 300; done
while true; do
  todo=$(find bench -mindepth 3 -maxdepth 3 -name metrics.json -not -path "bench/_invalid/*" -not -path "*.smoke/*" \
         -not -path "*.aborted*" | while read -r m; do d=$(dirname "$m"); [ -f "$d/judge.json" ] || echo "$d"; done)
  if [ -n "$todo" ]; then
    echo "$todo" | xargs -P "$jobs" -I{} bash -c 'd={}; a=$(basename $(dirname $d)); b=$(basename $d);
      echo "=== $(date +%H:%M:%S) judge $a $b: $(.venv/bin/python packages/ai-db-scene/eval/agent_bench/bench.py judge $a $b 2>&1 | tail -1 | cut -c1-200)"'
    sleep 300  # some judgements may have failed (rate limits): retry them after a pause
  else
    echo "=== $(date +%H:%M:%S) judging: all done"; break
  fi
done
