#!/usr/bin/env bash
# The local weak-model arm's server: Qwen3-14B-AWQ on vLLM (Docker, the local GPU), an
# OpenAI-compatible API with tool calling on 127.0.0.1:${LOCAL_MODEL_PORT:-8011}/v1 only.
# usage: local_model.sh start|stop|status|logs
# Weights are cached in ~/.cache/huggingface (mounted). 0.75 of the GPU leaves room for Blender.
set -euo pipefail
IMAGE=vllm/vllm-openai:v0.31.0-cu129  # pinned: CUDA 12.9 build, runs on Blackwell (sm_120)
MODEL=Qwen/Qwen3-14B-AWQ
SERVED=qwen3-14b
NAME=aidb-bench-vllm
PORT=${LOCAL_MODEL_PORT:-8011}
CTX=${LOCAL_MODEL_CTX:-32768}
GPU_UTIL=${LOCAL_MODEL_GPU_UTIL:-0.75}

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
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --runtime nvidia --gpus all --ipc=host \
      --dns "$(dns)" -p "127.0.0.1:$PORT:8000" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      --tmpfs /usr/local/lib/python3.12/dist-packages/torchcodec \
      "$IMAGE" \
      --model "$MODEL" --served-model-name "$SERVED" \
      --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL" \
      --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
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
