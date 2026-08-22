#!/usr/bin/env bash
# Usage: serve.sh <profile-id>. Launches vllm serve on 127.0.0.1:8000 and manages the status file.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/status.sh
source "$REPO_ROOT/lib/status.sh"

VLLM_BIN="${VLLM_BIN:-vllm}"
CURL_BIN="${CURL_BIN:-curl}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_POLL_SECS="${HEALTH_POLL_SECS:-5}"

profile_id="${1:-}"
profile_file="$REPO_ROOT/profiles/${profile_id}.yaml"
if [[ -z "$profile_id" || ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '${profile_id}' (expected $REPO_ROOT/profiles/<id>.yaml)"
  exit 2
fi

mapfile -t argv < <(python3 "$REPO_ROOT/lib/profile.py" "$profile_file" | tr -d '\r')
mkdir -p "$(dirname "$VLLM_LOG_FILE")"
write_status "STARTING"
echo "[serve] profile=$profile_id" | tee -a "$VLLM_LOG_FILE"

report_failure() {
  local code="$1"
  write_status "FAILED: vllm exited $code"
  echo "[serve] last 40 log lines:"
  tail -n 40 "$VLLM_LOG_FILE"
  if grep -qi "out of memory" "$VLLM_LOG_FILE"; then
    echo "[serve] hint: OOM at startup — set MAX_MODEL_LEN to a lower value in .env and re-run."
  fi
}

"$VLLM_BIN" serve "${argv[@]}" --host 127.0.0.1 --port 8000 >> "$VLLM_LOG_FILE" 2>&1 &
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
