#!/usr/bin/env bash
# Vast onstart entrypoint: env -> vLLM version check -> GPU detect -> token check -> serve.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/status.sh
source "$REPO_ROOT/lib/status.sh"

MIN_VLLM="0.17"
VLLM_VERSION_CMD="${VLLM_VERSION_CMD:-python3 -c \"import importlib.metadata as m; print(m.version('vllm'))\"}"
DETECT_GPU="${DETECT_GPU:-$REPO_ROOT/lib/detect_gpu.sh}"
SERVE="${SERVE:-$REPO_ROOT/scripts/serve.sh}"
DETECT_ERR_FILE="${DETECT_ERR_FILE:-$REPO_ROOT/.detect.err}"

# 1. .env (only sets vars that are unset or empty in the environment)
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS='=' read -r k v || [[ -n "$k" ]]; do
    k="${k%$'\r'}"; v="${v%$'\r'}"
    [[ -z "$k" || "$k" == \#* ]] && continue
    if [[ ! "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "[bootstrap] ignoring malformed .env line: $k" >&2; continue
    fi
    # strip an inline " #comment" and one matching pair of surrounding quotes
    v="${v%%[[:space:]]#*}"
    while [[ "$v" =~ [[:space:]]$ ]]; do v="${v%?}"; done
    if [[ ${#v} -ge 2 ]]; then
      if [[ "${v:0:1}" == '"' && "${v: -1}" == '"' ]]; then
        v="${v:1:-1}"
      elif [[ "${v:0:1}" == "'" && "${v: -1}" == "'" ]]; then
        v="${v:1:-1}"
      fi
    fi
    if [[ -z "${!k:-}" ]]; then export "$k=$v"; fi
  done < "$REPO_ROOT/.env"
fi
export HF_TOKEN="${HF_TOKEN:-}" PROFILE="${PROFILE:-}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# 2. vLLM version (probed via importlib.metadata so we never import vllm itself)
ver="$(bash -c "$VLLM_VERSION_CMD" 2>/dev/null | tail -n 1 | tr -d '\r' || true)"
if [[ -z "$ver" ]]; then
  write_status "FAILED: vllm not importable"; exit 4
fi
if [[ "$(printf '%s\n%s\n' "$MIN_VLLM" "$ver" | sort -V | head -n1)" != "$MIN_VLLM" ]]; then
  write_status "FAILED: vllm $ver < $MIN_VLLM"; exit 4
fi
echo "[bootstrap] vllm $ver"

# 3. profile
if [[ -z "$PROFILE" ]]; then
  if ! out="$(bash "$DETECT_GPU" 2>"$DETECT_ERR_FILE")"; then
    first_err="$(head -n 1 "$DETECT_ERR_FILE" 2>/dev/null || true)"
    write_status "FAILED: unknown gpu: ${first_err}"; exit 2
  fi
  PROFILE="${out%% *}"
  echo "[bootstrap] detected profile=$PROFILE gpus=${out#* }"
else
  echo "[bootstrap] PROFILE override=$PROFILE"
fi

# 4. token
profile_file="$REPO_ROOT/profiles/${PROFILE}.yaml"
if [[ ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '$PROFILE'"; exit 2
fi
if grep -qE '^requires_hf_token:\s*true' "$profile_file" && [[ -z "$HF_TOKEN" ]]; then
  write_status "FAILED: HF_TOKEN required by profile $PROFILE"; exit 3
fi
[[ -n "$HF_TOKEN" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# 5. curl must be present before we hand off to serve.sh (which polls /health with it)
command -v "${CURL_BIN:-curl}" >/dev/null 2>&1 || { write_status "FAILED: curl missing"; exit 5; }

# 6. serve
exec bash "$SERVE" "$PROFILE"
