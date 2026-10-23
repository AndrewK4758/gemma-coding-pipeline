#!/usr/bin/env zsh
# Phase 1 GPU Mode Switcher for Gemma 2 9B Coding Pipeline
# Controls local-ai-offload.service and dual-GPU visibility (RTX 5070 Ti + RTX 5080).

set -euo pipefail

SERVICE_NAME="local-ai-offload.service"
HEALTH_URL="http://127.0.0.1:11434/health"

print_vram() {
  echo "=== Dual-GPU Blackwell Status ==="
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,memory.free,compute_cap --format=csv
}

check_health() {
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "llama-swap (:11434/health): HEALTHY"
    return 0
  else
    echo "llama-swap (:11434/health): DOWN"
    return 1
  fi
}

cmd_status() {
  local svc_state
  svc_state="$(systemctl --user is-active "${SERVICE_NAME}" 2>/dev/null || true)"
  echo "Service (${SERVICE_NAME}): ${svc_state}"
  check_health || true
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset (all GPUs visible)>}"
  print_vram
}

cmd_train() {
  echo "[train] Stopping ${SERVICE_NAME} to release VRAM on RTX 5080 and RTX 5070 Ti..."
  systemctl --user stop "${SERVICE_NAME}"
  sleep 2
  export CUDA_VISIBLE_DEVICES=0,1
  echo "[train] Exported CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  print_vram
  echo "[train] Ready for dual-GPU training. Run: export CUDA_VISIBLE_DEVICES=0,1"
}

cmd_inference() {
  echo "[inference] Starting ${SERVICE_NAME}..."
  systemctl --user start "${SERVICE_NAME}"
  local retries=15
  while (( retries > 0 )); do
    if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "[inference] ${SERVICE_NAME} is active and :11434/health is OK."
      print_vram
      return 0
    fi
    sleep 1
    (( retries-- ))
  done
  echo "[inference] ERROR: ${SERVICE_NAME} failed health check at ${HEALTH_URL}" >&2
  return 1
}

case "${1:-status}" in
  status)
    cmd_status
    ;;
  train)
    cmd_train
    ;;
  inference)
    cmd_inference
    ;;
  *)
    echo "Usage: $0 {status|train|inference}" >&2
    exit 1
    ;;
esac
