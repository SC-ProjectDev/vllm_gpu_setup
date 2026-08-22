# vLLM GPU Setup — Milestone 1 Design

**Date:** 2026-08-22
**Status:** Approved design, pending implementation plan
**Supersedes:** `pre_planning.md` (kept for history; its "Qwen3.5" correction is stale)

## 1. Goal

A disposable repo that turns a freshly rented Vast.ai GPU instance into an
OpenAI-compatible vLLM server for **Qwen3.8-27B**, reachable from the desktop
over an SSH tunnel at `http://127.0.0.1:8000`, so the existing `llm-cli`
harness (`F:\Coding\VirtualTrashcan47\llm\repo`) works against it with a config
change only.

Milestone 1 success = the acceptance run in §8 passes on a rented RTX 5090.

## 2. Decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Model | `Qwen3.8-27B` (released 2026-08-14, Apache 2.0, 27B dense, hybrid Gated DeltaNet + Gated Attention, 262K native ctx) | Newest 27B; what the user wants |
| First GPU | RTX 5090 (32 GB, Blackwell) | Only consumer card on vLLM's documented path via NVFP4. 3090/4090 need a patched vLLM (`syv-ai/qwen38-27b-rtx3090`) — deferred |
| Engine | vLLM only, official image `vllm/vllm-openai:v0.27.1` | Qwen3.8 needs vLLM ≥ 0.17 (recurrent-layer kernels). v0.27.1 is latest as of 2026-08-11 |
| Instance shape | vLLM image **is** the Vast template; repo is an onstart script | Vast instances are containers — no Docker-in-Docker. Fastest cold start |
| Desktop tooling | Small stdlib-only Python script: tunnel + health + logs | Manual rental on Vast website for M1; Vast API is M3 |
| Repo location | This folder (`llm/vllm_gpu_setup`) as its own git repo | It is the thing cloned onto the instance; `llm-cli` untouched |
| Weights | Re-download each rental (~25 GB) | Simple; revisit with a Vast volume if slow |
| Exposure | vLLM bound to `127.0.0.1:8000` only; SSH `-L` tunnel | vLLM's `--api-key` does not protect every endpoint |

Deferred (not M1): FastAPI control plane, Vast API provisioning, 3090/4090
profiles, persistent volumes, auto-teardown, Ollama/SGLang backends.

## 3. Architecture

```
DESKTOP (Windows 11)
  llm-cli  --LLM_BASE_URL=http://127.0.0.1:8000-->  ssh.exe -N -L 8000:127.0.0.1:8000
  desktop/gpu_llm.py  (tunnel | status | logs | down)          |
                                                                v  (encrypted)
VAST INSTANCE  image: vllm/vllm-openai:v0.27.1
  onstart: git clone <repo> && bootstrap.sh
    detect_gpu.sh -> profile id -> profile.py -> vllm serve argv -> serve.sh
    vllm serve ... --host 127.0.0.1 --port 8000 --served-model-name qwen
    logs: /var/log/vllm.log   status: /var/log/vllm.status
```

The desktop never learns which GPU, quant, or context length is in play; it
always sees model `qwen` at the same URL.

## 4. Instance repo layout

```
vllm_gpu_setup/
├── bootstrap.sh            # onstart entrypoint
├── lib/
│   ├── detect_gpu.sh       # nvidia-smi -> profile id (or fail)
│   └── profile.py          # yaml profile -> vllm serve argv (pure function)
├── profiles/
│   ├── 5090.yaml
│   ├── a100-80.yaml
│   ├── h100-80.yaml
│   └── h200.yaml
├── scripts/
│   ├── serve.sh            # launch vllm, write status, tail on failure
│   ├── health.sh           # curl /health and /v1/models
│   └── smoke.sh            # one chat completion w/ thinking; prints tok/s
├── desktop/
│   └── gpu_llm.py          # desktop CLI (stdlib only)
├── vast/
│   └── template.md         # exact image tag, onstart line, disk, env
├── tests/                  # pytest, no GPU required
├── .env.example
├── README.md
└── docs/superpowers/specs/ # this file
```

### 4.1 `bootstrap.sh`

`set -euo pipefail`. Steps, each failing with a named reason written to
`/var/log/vllm.status` (`FAILED: <reason>`) and non-zero exit:

1. Load `.env` if present (`HF_TOKEN`, `PROFILE`, `MAX_MODEL_LEN`).
2. Check vLLM version ≥ 0.17 (`python3 -c "import vllm; print(vllm.__version__)"`).
3. `PROFILE=${PROFILE:-$(lib/detect_gpu.sh)}`; unknown GPU → print the
   `nvidia-smi` line and the profile table, exit 2.
4. If profile requires `HF_TOKEN` and it is unset → exit 3.
5. `exec scripts/serve.sh "$PROFILE"`.

### 4.2 `lib/detect_gpu.sh`

Reads `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.
Exact-name table first (`RTX 5090`→`5090`, `A100-SXM4-80GB`/`A100 80GB PCIe`→
`a100-80`, `H100`→`h100-80`, `H200`→`h200`), then a VRAM-class fallback
(≥130 GB→`h200`, ≥75 GB→`h100-80`, ≥30 GB→`5090`), then fail. GPU count is
printed as a second field for `tensor_parallel_size` (M1 profiles all use 1).

### 4.3 Profiles

YAML, one per GPU class. Parsed by `profile.py` with a minimal flat-YAML
reader (no PyYAML dependency guaranteed in the image). Fields:

```yaml
model: Inferact/Qwen3.8-27B-NVFP4
served_model_name: qwen
max_model_len: 32768
gpu_memory_utilization: 0.90
kv_cache_dtype: fp8
enforce_eager: true
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --reasoning-parser qwen3
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
```

| Profile | model | max_model_len | notes |
|---|---|---|---|
| `5090` | `Inferact/Qwen3.8-27B-NVFP4` | 32768 | `enforce_eager: true` (CUDA-graph OOM otherwise), kv fp8 |
| `a100-80` | `Qwen/Qwen3.8-27B` (bf16) | 131072 | kv fp8 |
| `h100-80` | `Qwen/Qwen3.8-27B-FP8` | 262144 | kv fp8 |
| `h200` | `Qwen/Qwen3.8-27B` (bf16) | 262144 | kv fp8 |

`MAX_MODEL_LEN` env overrides the profile value (for OOM tuning without
editing files). `profile.py` exposes a pure function
`profile_to_argv(profile: dict, overrides: dict) -> list[str]`; it is the unit
under test.

### 4.4 `scripts/serve.sh`

Runs `vllm serve` with the argv plus `--host 127.0.0.1 --port 8000`,
stdout+stderr to `/var/log/vllm.log`. Writes `STARTING` to
`/var/log/vllm.status`, then `READY` once `/health` returns 200, or
`FAILED: vllm exited <code>` (with a hint to lower `MAX_MODEL_LEN` if the log
contains "out of memory") and the last 40 log lines to stdout.

### 4.5 `scripts/smoke.sh`

One non-streaming `/v1/chat/completions` request with thinking enabled
(temp 1.0, top_p 0.95 per Qwen recommendations), asserts `reasoning_content`
non-empty and `content` non-empty, prints completion tokens / wall time as
tok/s. Exit non-zero on any assertion failure.

## 5. Desktop CLI (`desktop/gpu_llm.py`)

Python 3.11+, stdlib only (`argparse`, `subprocess`, `urllib.request`,
`tomllib`, `json`). Uses Windows built-in `ssh.exe`. State in
`~/.gpu-llm/state.json` (tunnel PID, host, port, local port); defaults in
`~/.gpu-llm/config.toml` (`host`, `ssh_port`, `local_port`, `ssh_key`).

| Command | Behaviour |
|---|---|
| `tunnel [--host --port --local]` | Spawn `ssh -N -L <local>:127.0.0.1:8000 -p <port> root@<host>`; save state; poll `http://127.0.0.1:<local>/health` until 200 or timeout (default 15 min). Every 30 s, print `tail -n 3 /var/log/vllm.log` and `/var/log/vllm.status` via a one-shot SSH so downloads are visible. Stop early with the status reason if status starts with `FAILED`. On success print `LLM_BASE_URL=http://127.0.0.1:<local>` and `LLM_MODEL=qwen`. |
| `status` | Tunnel PID alive? `/v1/models` JSON? One `nvidia-smi` line over SSH. |
| `logs [-f]` | `ssh ... tail [-f] -n 100 /var/log/vllm.log`. |
| `down` | Kill tunnel PID, clear state, print reminder that the Vast instance is still billing. |

`llm-cli` integration: no code change. User sets `LLM_BASE_URL` and
`coder_model = "qwen"` in `~/.config/llm-cli/config.toml` (documented in README).

## 6. Vast template (`vast/template.md`)

- Image: `vllm/vllm-openai:v0.27.1`
- Launch mode: SSH, entrypoint overridden (image default is `vllm serve`
  with no args — must be replaced by the onstart command).
- Onstart: `cd /root && git clone <repo-url> vllm_gpu_setup && cd vllm_gpu_setup && cp -n .env.example .env && nohup ./bootstrap.sh > /var/log/bootstrap.log 2>&1 &`
- Disk: ≥ 60 GB (weights + HF cache + image layers).
- Env: `HF_TOKEN` (optional for public repos), `PROFILE` override.
- Filters to rent by: GPU = RTX 5090, download speed ≥ 500 Mbps, reliability ≥ 98%.
- Ports: none exposed beyond SSH.

## 7. Error handling summary

| Failure | Where caught | Surface |
|---|---|---|
| Unknown GPU | `detect_gpu.sh` | status `FAILED: unknown gpu <name>`; bootstrap exit 2 |
| vLLM < 0.17 | `bootstrap.sh` | status `FAILED: vllm <ver> < 0.17`; exit 4 |
| Missing HF token | `bootstrap.sh` | status `FAILED: HF_TOKEN required`; exit 3 |
| OOM / vllm crash | `serve.sh` | status `FAILED: vllm exited N` + log tail + `MAX_MODEL_LEN` hint |
| Tunnel cannot connect | `gpu_llm.py tunnel` | ssh exit code + stderr; no state written |
| Server never ready | `gpu_llm.py tunnel` | timeout with last status/log lines shown |

## 8. Testing

**Local (pytest, no GPU):**
- `profile_to_argv` for each shipped profile and for `MAX_MODEL_LEN` override.
- `detect_gpu` mapping against fixture `nvidia-smi` strings: 5090, A100 80GB
  (both names), H100, H200, 2×H100 (count), unknown → exit 2.
- `gpu_llm.py` health polling and `FAILED` early-exit against a fake local HTTP
  server; state file round-trip; `down` with a dead PID.

**M1 acceptance run (rented 5090, documented in README with expected timings):**
1. Rent with `vast/template.md` settings.
2. `gpu_llm.py tunnel` → sees download progress → prints `READY` within ~10 min.
3. Over SSH, `scripts/smoke.sh` passes and prints tok/s.
4. From desktop: `llm-cli health`, `llm-cli chat` (streaming + reasoning shown),
   `llm-cli agent` completes a tool-calling task against a local folder.
5. `gpu_llm.py down`; destroy instance on Vast.

Record achieved tok/s, peak VRAM (`nvidia-smi`), and whether `max_model_len`
can be raised above 32768 on the 5090 profile — feeds milestone 2 tuning.

## 9. Sources

- Qwen3.8-27B model card: https://huggingface.co/Qwen/Qwen3.8-27B
- vLLM recipe (min version, NVFP4, enforce-eager): https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- GPU ladder / 3090-4090 caveat: https://www.orcarouter.ai/blog/qwen-3-8-27b-vllm
- Hardware requirements: https://www.yottalabs.ai/post/qwen-3-8-27b-specs-hardware-requirements-how-to-run-2026
- Patched 3090 recipe (deferred): https://github.com/syv-ai/qwen38-27b-rtx3090
- vLLM releases: https://github.com/vllm-project/vllm/releases
