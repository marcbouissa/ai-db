#!/usr/bin/env bash
# The local weak-model arm's server: Qwen3-14B-AWQ (default) or the vision model
# Qwen2.5-VL-7B-Instruct-AWQ (LOCAL_MODEL=qwen2.5-vl-7b) on vLLM (Docker, the local GPU), an
# OpenAI-compatible API with tool calling on 127.0.0.1:${LOCAL_MODEL_PORT:-8011}/v1 only.
# usage: [LOCAL_MODEL=qwen3-14b|qwen2.5-vl-7b] local_model.sh start|stop|status|logs
# Weights are cached in ~/.cache/huggingface (mounted). 0.75 of the GPU leaves room for Blender.
set -euo pipefail
IMAGE=vllm/vllm-openai:v0.31.0-cu129  # pinned: CUDA 12.9 build, runs on Blackwell (sm_120)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=aidb-bench-vllm
PORT=${LOCAL_MODEL_PORT:-8011}
CTX=${LOCAL_MODEL_CTX:-32768}
case "${LOCAL_MODEL:-qwen3-14b}" in
  qwen3-14b)
    MODEL=Qwen/Qwen3-14B-AWQ
    SERVED=qwen3-14b
    GPU_UTIL=${LOCAL_MODEL_GPU_UTIL:-0.75}
    EXTRA=(--tool-call-parser hermes --reasoning-parser qwen3)
    ;;
  qwen2.5-vl-7b)
    # Qwen2.5-VL calls tools in Qwen2.5's hermes style (<tool_call> JSON </tool_call>), but its own
    # chat template has no tools section and no tool role: qwen2_5_vl_tools.jinja is Qwen2.5's tool
    # template with the VL template's image handling. One image per request, at most 1280 x 784 px
    # worth of vision tokens (~1.3k); not a thinking model (no reasoning parser).
    MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ
    SERVED=qwen2.5-vl-7b
    GPU_UTIL=${LOCAL_MODEL_GPU_UTIL:-0.6}
    EXTRA=(--tool-call-parser hermes --chat-template /templates/qwen2_5_vl_tools.jinja
           --chat-template-content-format openai --limit-mm-per-prompt '{"image": 1, "video": 0}'
           --mm-processor-kwargs '{"max_pixels": 1003520}')
    ;;
  *) echo "LOCAL_MODEL must be qwen3-14b or qwen2.5-vl-7b"; exit 2 ;;
esac

# Docker's daemon.json may pin a DNS server that is not reachable on this network: use the
# host's upstream resolver (systemd-resolved) for the container.
# The cu129 image ships a torchcodec built for CUDA 13 (libnvrtc.so.13 missing): its import
# raises OSError, which vLLM does not catch. An empty tmpfs over the package turns that into an
# ImportError, which vLLM handles (audio input only; not needed here).
dns() {
  awk '/^nameserver/ {print $2; exit}' /run/systemd/resolve/resolv.conf 2>/dev/null || echo 1.1.1.1
}

case "${1:-}" in
  start)
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "already running"; exit 0; fi
    # never share the GPU with an Isaac Sim / stage_ov run (it records and can run out of memory)
    sim=$(docker ps --format '{{.Names}}' | grep -E 'stage_ov-sim|isaac' || true)
    if [ -n "$sim" ] && [ -z "${AIDB_ALLOW_WITH_SIM:-}" ]; then
      echo "refused: simulation container running ($sim); set AIDB_ALLOW_WITH_SIM=1 to override" >&2; exit 1
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
      --dns "$(dns)" -p "127.0.0.1:$PORT:8000" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -v "$HERE/qwen2_5_vl_tools.jinja:/templates/qwen2_5_vl_tools.jinja:ro" \
      --tmpfs /usr/local/lib/python3.12/dist-packages/torchcodec \
      "$IMAGE" \
      --model "$MODEL" --served-model-name "$SERVED" \
      --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL" \
      --enable-auto-tool-choice "${EXTRA[@]}" \
      --max-num-seqs 4
    echo "waiting for http://127.0.0.1:$PORT/v1/models"
    for _ in $(seq 1 120); do
      if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then echo "ready"; exit 0; fi
      if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then docker logs --tail 40 "$NAME"; exit 1; fi
      sleep 5
    done
    echo "not ready after 10 min"; docker logs --tail 40 "$NAME"; exit 1
    ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped" || echo "not running" ;;
  status) curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" && echo || { echo "down"; exit 1; } ;;
  logs) docker logs --tail "${2:-50}" "$NAME" ;;
  *) echo "usage: $0 start|stop|status|logs"; exit 2 ;;
esac
