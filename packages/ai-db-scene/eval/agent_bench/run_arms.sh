#!/usr/bin/env bash
# Run every brief for the given arms; finished scenes (metrics.json present) are skipped.
# usage: run_arms.sh [-j N] <arm> [<arm> ...]     N scenes at a time (default 1)
# Each scene has its own Blender on its own port, so parallel scenes cannot touch each other.
cd "$(dirname "$0")/../../../.." || exit 1
jobs=1
if [ "$1" = "-j" ]; then jobs=$2; shift 2; fi
one() {
  arm=$1 brief=$2
  [ -f "bench/$arm/$brief/metrics.json" ] && exit 0
  echo "=== $(date +%H:%M:%S) start $arm $brief"
  out=$(.venv/bin/python packages/ai-db-scene/eval/agent_bench/bench.py run "$arm" "$brief" 2>&1 | tail -1)
  echo "=== $(date +%H:%M:%S) done  $arm $brief: $out"
}
export -f one
# Remaining scenes round-robin over arms (one per arm per round), raw and aidb of a model kept
# apart: parallel slots hold different models, which spreads the free tier's per-model limits.
python3 - "$@" <<'PY' | xargs -P "$jobs" -L 1 bash -c 'one "$0" "$1"'
import os, sys
briefs = ["living_room", "classroom", "cafe", "round_lobby", "park_corner", "warehouse"]
arms = sys.argv[1:]
arms = [a for a in arms if a.endswith("-raw")] + [a for a in arms if not a.endswith("-raw")]
todo = {a: [b for b in briefs if not os.path.exists(f"bench/{a}/{b}/metrics.json")] for a in arms}
for r in range(len(briefs)):
    for a in arms:
        if r < len(todo[a]):
            print(a, todo[a][r])
PY
echo "=== $(date +%H:%M:%S) all done: $*"
