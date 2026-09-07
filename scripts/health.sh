#!/usr/bin/env bash
# /health (unauthenticated) + /v1/models (bearer VLLM_API_KEY when set).
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
CURL_BIN="${CURL_BIN:-curl}"
auth=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then auth=(-H "Authorization: Bearer $VLLM_API_KEY"); fi
"$CURL_BIN" -fsS -m 5 "$BASE/health" >/dev/null && echo "health: ok"
"$CURL_BIN" -fsS -m 5 ${auth[@]+"${auth[@]}"} "$BASE/v1/models"
echo
