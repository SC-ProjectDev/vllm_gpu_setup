#!/usr/bin/env bash
# Source this. Provides write_status.
VLLM_STATUS_FILE="${VLLM_STATUS_FILE:-/var/log/vllm.status}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-/var/log/vllm.log}"

write_status() {
  mkdir -p "$(dirname "$VLLM_STATUS_FILE")"
  printf '%s\n' "$1" > "$VLLM_STATUS_FILE"
  echo "[status] $1"
}
