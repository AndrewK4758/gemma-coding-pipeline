#!/usr/bin/env zsh
# Phase 1 GPU Mode Switcher for Gemma Coding Pipeline
# Controls optional local inference user service ($LOCAL_OFFLOAD_SERVICE) and dual-GPU visibility.

set -euo pipefail

SERVICE_NAME="${LOCAL_OFFLOAD_SERVICE:-}"
HEALTH_URL="${INFERENCE_HEALTH_URL:-http://127.0.0.1:11434/health}"

print_vram() {
  echo "=== Dual-GPU CUDA Status ==="
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,memory.free,compute_cap --format=csv
}

check_health() {
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "Inference endpoint (${HEALTH_URL}): HEALTHY"
    return 0
  else
    echo "Inference endpoint (${HEALTH_URL}): DOWN"
    return 1
  fi
}

cmd_status() {
  if [[ -n "${SERVICE_NAME}" ]]; then
    local svc_state
    svc_state="$(systemctl --user is-active "${SERVICE_NAME}" 2>/dev/null || true)"
    echo "Service (${SERVICE_NAME}): ${svc_state}"
  fi
  check_health || true
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset (all GPUs visible)>}"
  print_vram
}

cmd_train() {
  if [[ -n "${SERVICE_NAME}" ]]; then
    echo "[train] Stopping ${SERVICE_NAME} to release VRAM across GPUs..."
    systemctl --user stop "${SERVICE_NAME}"
    sleep 2
  fi
  export CUDA_VISIBLE_DEVICES=0,1
  echo "[train] Exported CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  print_vram
  echo "[train] Ready for dual-GPU training. Run: export CUDA_VISIBLE_DEVICES=0,1"
}

cmd_inference() {
  if [[ -z "${SERVICE_NAME}" ]]; then
    echo "[inference] LOCAL_OFFLOAD_SERVICE is unset; skipping systemd service start."
    print_vram
    return 0
  fi
  echo "[inference] Starting ${SERVICE_NAME}..."
  systemctl --user start "${SERVICE_NAME}"
  local retries=15
  while (( retries > 0 )); do
    if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "[inference] ${SERVICE_NAME} is active and ${HEALTH_URL} is OK."
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
  inference|serve)
    cmd_inference
    ;;
  *)
    echo "Usage: $0 {status|train|inference|serve}" >&2
    exit 1
    ;;
esac
