#!/usr/bin/env bash
# The whole Oval Office photo -> scene experiment (photo_exp.py), unattended.
# usage: oval_photo_run.sh [repeats=1] [workers=3]
# Refuses to start unless the GPU is free: no aidb-bench-vllm container, < 2 GB GPU memory used,
# no bench.py running. Then: VL server (LOCAL_MODEL=qwen2.5-vl-7b local_model.sh) -> smoke test
# (the photo + one tool call) -> deepen calls (cell B) -> 6 cells x repeats builds -> bench.py
# rescore -> bench/_oval_photo/REPORT.md, blind_sheet.png, blind_key.json -> server stopped (also
# on error). Finished builds (metrics.json) are skipped, so a rerun resumes.
# Runtime estimate (RTX 5090 laptop): server start ~2-4 min, deepen ~10-30 s per repeat, a build
# <= 20 min (cap; Qwen3-14B aidb builds took ~3 min, raw ones often ran to the cap), so with 3
# workers ~40 min per repeat (two rounds of <= 20 min): 1 repeat ~45 min, 2 repeats ~1 h 25 min.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
REPEATS=${1:-1}
WORKERS=${2:-3}
PY="$REPO/.venv/bin/python"
export LOCAL_MODEL=qwen2.5-vl-7b HERE
PORT=${LOCAL_MODEL_PORT:-8011}

if docker ps --format '{{.Names}}' | grep -qx aidb-bench-vllm; then echo "refused: aidb-bench-vllm is running"; exit 1; fi
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$mem" -ge 2048 ]; then echo "refused: ${mem} MiB of GPU memory in use"; exit 1; fi
if pgrep -f 'agent_bench/bench\.py run' >/dev/null; then echo "refused: a bench.py run is in progress"; exit 1; fi

trap '"$HERE/local_model.sh" stop' EXIT
"$HERE/local_model.sh" start

"$PY" - "$PORT" <<'PY'
import json, os, sys, urllib.request
sys.path.insert(0, os.environ["HERE"])
import photo_exp, strata_agent
body = {"model": photo_exp.SERVED, "temperature": 0, "max_tokens": 256,
        "tools": [{"type": "function", "function": {"name": "report_room", "description": "Report the room's shape.",
                   "parameters": {"type": "object", "properties": {"shape": {"type": "string"}}, "required": ["shape"]}}}],
        "messages": [{"role": "user", "content": strata_agent.user_content(
            "What shape is the room in this photo? Answer by calling report_room.", photo_exp.IMAGE)}]}
req = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
print("smoke:", json.dumps(msg.get("tool_calls") or msg.get("content"))[:400])
if not msg.get("tool_calls"):
    sys.exit("smoke test failed: the model answered without a tool call (chat template / parser)")
PY

"$PY" "$HERE/photo_exp.py" run --repeats "$REPEATS" --workers "$WORKERS"
arms=$("$PY" -c "import sys; sys.path.insert(0, '$HERE'); import photo_exp; print(' '.join(photo_exp.builds($REPEATS)))")
# shellcheck disable=SC2086
"$PY" "$HERE/bench.py" rescore $arms
"$PY" "$HERE/photo_report.py" report "$REPEATS"
"$PY" "$HERE/photo_report.py" sheet "$REPEATS"
echo "done: $REPO/bench/_oval_photo/{REPORT.md,blind_sheet.png,blind_key.json}"
