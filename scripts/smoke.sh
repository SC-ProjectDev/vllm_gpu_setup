#!/usr/bin/env bash
# One thinking-mode chat completion. Asserts reasoning + content, prints tok/s.
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen}"

payload=$(cat <<JSON
{"model":"$MODEL","stream":false,"temperature":1.0,"top_p":0.95,"max_tokens":512,
 "messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],
 "chat_template_kwargs":{"enable_thinking":true}}
JSON
)

start="${SMOKE_START:-$(date +%s.%N)}"
resp="$(curl -fsS -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$payload")"
end="${SMOKE_END:-$(date +%s.%N)}"

RESP="$resp" START="$start" END="$end" python3 - <<'PY'
import json, os, sys
r = json.loads(os.environ["RESP"])
msg = r["choices"][0]["message"]
reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
content = msg.get("content") or ""
if not reasoning.strip():
    print("FAIL: reasoning_content empty (is --reasoning-parser qwen3 set?)"); sys.exit(1)
if not content.strip():
    print("FAIL: content empty"); sys.exit(1)
print("reasoning: ok")
print("content:", content.strip()[:80])
toks = r.get("usage", {}).get("completion_tokens", 0)
secs = float(os.environ["END"]) - float(os.environ["START"])
if secs > 0:
    print(f"tok/s: {toks / secs:.1f}  ({toks} tokens in {secs:.1f}s)")
else:
    print(f"tok/s: n/a  ({toks} tokens, elapsed time below clock resolution)")
PY
