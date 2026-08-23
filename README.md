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
api_key = "..."          # Vast API key, from https://cloud.vast.ai/manage-keys/
                          # (or set the VAST_API_KEY env var instead)
local_port = 8000
# ssh_key = "C:/Users/you/.ssh/id_ed25519"   # optional

# Optional stricter offer filters (defaults: inet_down 500, reliability 0.98,
# any country):
# [filters]
# inet_down = 1000
# reliability = 0.99
# country = ["US", "CA"]
# host / ssh_port only needed for the manual-rental fallback below —
# `up`/`down`/`status` record these automatically once an instance is rented.
```

`%USERPROFILE%\.config\llm-cli\config.toml` (only lines that change):

```toml
base_url = "http://127.0.0.1:8000"
model = "qwen"
coder_model = "qwen"
```

## Daily loop

1. `python desktop/gpu_llm.py up` — searches Vast (`--gpu 5090` is the
   default) and shows the 5 cheapest matching offers, numbered, with $/hr,
   Mbps, reliability, country, and offer id; pick one by number (Enter
   aborts). `--yes` auto-rents the cheapest, `--list N` changes the count,
   and `--offer <id>` rents a specific offer id straight from the console.
   It then rents, waits for the instance to come up, opens the tunnel, and
   prints `vLLM is READY.` (5–10 min cold).
2. Use `llm-cli chat`, `llm-cli code`, `llm-cli agent` as usual.
3. `python desktop/gpu_llm.py status` / `logs -f` when curious.
4. `python desktop/gpu_llm.py down` — closes the tunnel **and destroys the
   Vast instance**, so nothing keeps billing. Pass `--keep` to close the
   tunnel but leave the instance running (e.g. to reconnect later with
   `gpu-llm tunnel`); a plain `down` afterwards destroys it.

### Fallback: manual rental

If you'd rather rent by hand (or don't have a Vast API key configured), rent
an instance using `vast/template.md`, copy host/port into `config.toml`, and
run `python desktop/gpu_llm.py tunnel` instead of `up`. In this mode `down`
only closes the tunnel — it does not know about a manually-rented instance,
so you still need to **destroy the instance in the Vast console** yourself.

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

## Monitoring a running instance

All from the desktop, while the tunnel is up:

- **Live throughput feed**: `python desktop/gpu_llm.py logs -f` — vLLM logs
  avg prompt/generation tok/s and running/waiting request counts every ~10 s.
- **Prometheus metrics (no ssh needed)**: vLLM serves them on the API port,
  so `curl http://127.0.0.1:8000/metrics` works through the tunnel. Key
  series: `vllm:generation_tokens_total` (sample twice to compute tok/s),
  `vllm:num_requests_running`/`_waiting`, `vllm:gpu_cache_usage_perc`, and
  TTFT / per-token latency histograms.
- **GPU load**: `python desktop/gpu_llm.py status` shows VRAM; for
  utilization/power/temp, ssh in and run
  `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader`.

Observed on the 5090 profile (M2 acceptance, single stream): ~9 tok/s
generation during an agent coding session at only ~17% GPU util / 139 W —
single-request decode is memory-bandwidth-bound and `--enforce-eager` taxes
it further. Batched/concurrent requests are where the card's headroom is;
aggregate throughput scales well past the single-stream number.

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

## Milestone 2 acceptance run (paid, ~30 min)

- [ ] `python desktop/gpu_llm.py up` — offer shown with $/hr, confirm, READY,
      `llm-cli health` + `chat` work.
- [ ] `python desktop/gpu_llm.py status` shows `instance: <id> ($/hr)`.
- [ ] `python desktop/gpu_llm.py down` — instance destroyed; verify gone in
      the console.
- [ ] `python desktop/gpu_llm.py up --yes --max-price 0.05` exits 1 with the
      over-cap listing; rents nothing.
- [ ] Verify `GPU_FILTERS` gpu_name spellings against the live offers seen
      above; fix the table if Vast spells any differently.
- [ ] Record: time from `up` to READY, total cost.

## Deferred (milestone 3)

FastAPI control plane, 3090/4090 profiles (need patched vLLM), persistent
weight volumes, `gpu-llm stats` subcommand (sample `/metrics` twice and
print live tok/s + queue depth + KV-cache/VRAM in one shot).
