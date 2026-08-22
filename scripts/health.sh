#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
CURL_BIN="${CURL_BIN:-curl}"
"$CURL_BIN" -fsS -m 5 "$BASE/health" >/dev/null && echo "health: ok"
"$CURL_BIN" -fsS -m 5 "$BASE/v1/models"
echo
