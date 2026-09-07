#!/usr/bin/env bash
# Usage: serve.sh <gpu-id> <model-id>. Launches vllm serve on 127.0.0.1:8000 and manages the status file.
# Profile: profiles/<gpu-id>/<model-id>.yaml. If it has a `download:` key, the bucket is
# synced into the profile's `model:` path first. VLLM_API_KEY (if set) becomes --api-key.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/status.sh
source "$REPO_ROOT/lib/status.sh"

VLLM_BIN="${VLLM_BIN:-vllm}"
CURL_BIN="${CURL_BIN:-curl}"
HF_BIN="${HF_BIN:-hf}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_POLL_SECS="${HEALTH_POLL_SECS:-5}"

gpu_id="${1:-}"
model_id="${2:-}"
profile_file="$REPO_ROOT/profiles/${gpu_id}/${model_id}.yaml"
if [[ -z "$gpu_id" || -z "$model_id" || ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '${gpu_id}/${model_id}' (expected $REPO_ROOT/profiles/<gpu>/<model>.yaml)"
  exit 2
fi

mapfile -t argv < <(python3 "$REPO_ROOT/lib/profile.py" "$profile_file" | tr -d '\r')
mkdir -p "$(dirname "$VLLM_LOG_FILE")"
write_status "STARTING"
echo "[serve] profile=$gpu_id/$model_id" | tee -a "$VLLM_LOG_FILE"

report_failure() {
  local code="$1"
  write_status "FAILED: vllm exited $code"
  echo "[serve] last 40 log lines:"
  tail -n 40 "$VLLM_LOG_FILE"
  if grep -qi "out of memory" "$VLLM_LOG_FILE"; then
    echo "[serve] hint: OOM at startup — set MAX_MODEL_LEN to a lower value in .env and re-run."
  fi
}

# Optional bucket sync: `download: hf://buckets/...` -> the local `model:` directory (argv[0]).
download="$(sed -n 's/^download:[[:space:]]*//p' "$profile_file" | head -n 1 | tr -d '\r')"
if [[ -n "$download" ]]; then
  echo "[serve] syncing $download -> ${argv[0]}" | tee -a "$VLLM_LOG_FILE"
  sync_rc=0
  "$HF_BIN" buckets sync "$download" "${argv[0]}" >> "$VLLM_LOG_FILE" 2>&1 || sync_rc=$?
  if [[ $sync_rc -ne 0 ]]; then
    write_status "FAILED: download failed (rc $sync_rc)"
    echo "[serve] last 40 log lines:"
    tail -n 40 "$VLLM_LOG_FILE"
    exit 6
  fi
fi

api_args=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  api_args=(--api-key "$VLLM_API_KEY")
fi

"$VLLM_BIN" serve "${argv[@]}" ${api_args[@]+"${api_args[@]}"} --host 127.0.0.1 --port 8000 >> "$VLLM_LOG_FILE" 2>&1 &
vllm_pid=$!

# Wait for health or exit.
while kill -0 "$vllm_pid" 2>/dev/null; do
  if "$CURL_BIN" -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
    write_status "READY"
    wait "$vllm_pid" || code=$?
    code="${code:-0}"
    if [[ $code -ne 0 ]]; then
      report_failure "$code"
      exit "$code"
    fi
    exit 0
  fi
  sleep "$HEALTH_POLL_SECS"
done

wait "$vllm_pid" || code=$?
code="${code:-0}"
report_failure "$code"
exit "$code"
