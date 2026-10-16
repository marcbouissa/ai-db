#!/usr/bin/env bash
# Keep running the queue until every scene of the given arms has a metrics.json.
# Survives model-server outages (Strata) and crashed runs: an unfinished scene is retried
# from scratch on the next pass; finished scenes are never redone.
# usage: supervise.sh [-j N] <arm> [<arm> ...]
cd "$(dirname "$0")/../../../.." || exit 1
here=packages/ai-db-scene/eval/agent_bench
args=("$@")
arms=("$@"); [ "$1" = "-j" ] && arms=("${@:3}")
missing() {
  n=0
  for a in "${arms[@]}"; do for b in living_room classroom cafe round_lobby park_corner warehouse beach_cove; do
    [ -f "bench/$a/$b/metrics.json" ] || n=$((n + 1)); done; done
  echo $n
}
pass=0
while [ "$(missing)" -gt 0 ]; do
  pass=$((pass + 1))
  if printf '%s\n' "${arms[@]}" | grep -q '^strata-'; then
    until curl -s -m 10 -o /dev/null http://10.160.30.12:20000/v1/models; do
      echo "=== $(date +%H:%M:%S) strata unreachable, waiting"; sleep 300
    done
  fi
  before=$(missing)
  echo "=== $(date +%H:%M:%S) pass $pass: $before scenes left"
  "$here/run_arms.sh" "${args[@]}"
  if [ "$(missing)" -ge "$before" ] && [ "$before" -gt 0 ]; then
    # nothing finished (provider quotas or outage): back off instead of hammering
    echo "=== $(date +%H:%M:%S) no progress in pass $pass, pausing 30 min"; sleep 1800
  else
    sleep 60
  fi
done
echo "=== $(date +%H:%M:%S) supervisor: all scenes done for ${arms[*]}"
