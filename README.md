# vllm_gpu_setup

Disposable bootstrap for a rented Vast.ai GPU that serves **Qwen3.8-27B** via
vLLM's OpenAI-compatible API, tunnelled to your desktop at
`http://127.0.0.1:8000`. Design: `docs/superpowers/specs/2026-08-22-vllm-gpu-setup-design.md`.

## Desktop prerequisites

- Windows 11 with built-in OpenSSH (`ssh.exe`) and your Vast SSH key loaded.
- Python 3.11+.
- `llm-cli` from `F:\Coding\VirtualTrashcan47\llm\repo` installed.

## One-time desktop config

`%USERPROFILE%\.gpu-llm\config.toml`:

```toml
host = "ssh5.vast.ai"   # from the Vast instance's SSH button
ssh_port = 12345
local_port = 8000
# ssh_key = "C:/Users/you/.ssh/id_ed25519"   # optional
```

`%USERPROFILE%\.config\llm-cli\config.toml` (only lines that change):

```toml
base_url = "http://127.0.0.1:8000"
model = "qwen"
coder_model = "qwen"
```

## Daily loop

1. Rent an instance using `vast/template.md` (5090 filter). Copy host/port into `config.toml`.
2. `python desktop/gpu_llm.py tunnel` — opens the tunnel and prints instance
   progress every 30 s until `READY` (5–10 min cold).
3. Use `llm-cli chat`, `llm-cli code`, `llm-cli agent` as usual.
4. `python desktop/gpu_llm.py status` / `logs -f` when curious.
5. `python desktop/gpu_llm.py down`, then **destroy the instance in the Vast console**.

## Instance-side scripts

| Script | Purpose |
|---|---|
| `bootstrap.sh` | onstart entrypoint: version check → GPU detect → profile → serve |
| `lib/detect_gpu.sh` | `nvidia-smi` → profile id |
| `lib/profile.py` | profile YAML → `vllm serve` argv |
| `scripts/serve.sh <profile>` | launch vLLM, manage `/var/log/vllm.status` |
| `scripts/health.sh` | `/health` + `/v1/models` |
| `scripts/smoke.sh` | one thinking-mode completion; prints tok/s |

Status file values: `STARTING`, `READY`, `FAILED: <reason>`.
Bootstrap exit codes: 2 unknown GPU, 3 `HF_TOKEN` required, 4 vLLM < 0.17 or
not importable, 5 `curl` missing.

## Troubleshooting

- **`REMOTE HOST IDENTIFICATION HAS CHANGED`**: Vast reuses hostnames like
  `ssh5.vast.ai:PORT` across different rented instances, so a new rental can
  present a different host key on the same host:port pair OpenSSH already has
  cached. `gpu_llm.py tunnel` writes the tunnel ssh's stderr to
  `~/.gpu-llm/ssh.log` — check there for this message, then fix it with
  `ssh-keygen -R "[host]:port"` (matching the `host`/`ssh_port` from
  `config.toml`) and re-run `tunnel`.

## Tests (no GPU needed)

```
python -m pytest
```

Requires `bash` on PATH (Git Bash on Windows).

## Milestone 1 acceptance run (RTX 5090)

Budget ~1 hour of rental. Record the values in the table at the end.

- [ ] Rent per `vast/template.md`; note host/port; `config.toml` updated.
- [ ] `python desktop/gpu_llm.py tunnel` prints download progress, then
      `vLLM is READY.` within 15 min. (If it prints `FAILED: ...`, read
      `logs` and fix before spending more time.)
- [ ] SSH in: `bash /root/vllm_gpu_setup/scripts/smoke.sh` prints `reasoning: ok` and a `tok/s:` line.
- [ ] Desktop: `llm-cli health` OK; `llm-cli models` lists `qwen`.
- [ ] Desktop: `llm-cli chat` streams a reply and shows reasoning.
- [ ] Desktop: `llm-cli agent` completes a small tool-calling task (e.g. "list the files in this folder and summarise them").
- [ ] SSH in: `nvidia-smi` — record peak memory.used.
- [ ] Optional: set `MAX_MODEL_LEN=49152` in `.env`, re-run `bootstrap.sh`, see if it stays READY.
- [ ] `python desktop/gpu_llm.py down`; destroy the instance.

| Metric | Value |
|---|---|
| Time to READY | ~10 min (2026-08-23, RTX 5090, weights cold) |
| smoke.sh tok/s | 15.4 (32K ctx, cold) / 20.1 (49K ctx, warm) |
| Peak VRAM used | 29,514 MiB of 32,607 |
| Max stable max_model_len | 49152 (READY, health + completion verified) |
| Cost of run | $0.592/hr (≈$0.60 for the ~1 h acceptance run) |

## Deferred (milestones 2–3)

FastAPI control plane, Vast API provisioning (`gpu-llm up`), 3090/4090
profiles (need patched vLLM), persistent weight volumes, auto-teardown.
