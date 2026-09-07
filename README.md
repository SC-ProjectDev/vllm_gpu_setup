# vllm_gpu_setup

Disposable bootstrap for a rented Vast.ai GPU that serves a model you pick
from `profiles/` (default **Qwen3.8-27B**; also gpt-oss-120b and an
abliterated FP8 Qwen on 80 GB cards — see `profiles/README.md`) via vLLM's
OpenAI-compatible API, tunnelled to your desktop at `http://127.0.0.1:8000`.
Whatever the model, it is served under the constant name **`local`** behind a
bearer API key, so llm-cli and Rider configs never change.
Designs: `docs/superpowers/specs/` (M1 2026-08-22, M2 2026-08-23, M3 2026-09-07).

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
# llm_api_key = "..."     # optional bearer token for vLLM; if unset, `up` generates
                          # one into %USERPROFILE%\.gpu-llm\llm_api_key (LLM_API_KEY env overrides)

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
model = "local"
coder_model = "local"
```

plus llm-cli's API key set to the `LLM_API_KEY` value that `up` prints
(the same string lives in `%USERPROFILE%\.gpu-llm\llm_api_key`). vLLM
rejects `/v1/*` requests without it.

## Daily loop

1. `python desktop/gpu_llm.py up` — searches Vast (`--gpu 5090` is the
   default) and shows the 5 cheapest matching offers, numbered, with $/hr,
   Mbps, reliability, country, and offer id; pick one by number (Enter
   aborts). `--yes` auto-rents the cheapest, `--list N` changes the count,
   and `--offer <id>` rents a specific offer id straight from the console.
   If the GPU has more than one model in `profiles/<gpu>/`, a second
   numbered picker follows (quant, size, context; Enter = default), or pass
   `--model <id>` (e.g. `--gpu a100-80 --model gpt-oss-120b`). `--yes`
   takes the default model. Disk is sized from the profile's `disk_gb`.
   It then rents, waits for the instance to come up, opens the tunnel, and
   prints `vLLM is READY.` (5–10 min cold; longer for the 60–100 GB models)
   followed by `LLM_BASE_URL`, `LLM_MODEL=local`, `LLM_API_KEY`, and
   `LLM_CONTEXT` — everything a client needs.
   **Push before `up`**: the instance clones `main`, so your local
   `profiles/` and the instance's must agree.
2. Use `llm-cli chat`, `llm-cli code`, `llm-cli agent` as usual.
3. `python desktop/gpu_llm.py status` / `logs -f` when curious.
4. `python desktop/gpu_llm.py down` — closes the tunnel **and destroys the
   Vast instance**, so nothing keeps billing. Pass `--keep` to close the
   tunnel but leave the instance running (e.g. to reconnect later with
   `gpu-llm tunnel`); a plain `down` afterwards destroys it.

### Fallback: manual rental

If you'd rather rent by hand (or don't have a Vast API key configured), rent
an instance using `vast/template.md` (set `-e MODEL=` to the model id you
want and `-e VLLM_API_KEY=` to the contents of `llm_api_key` so the desktop's
key matches), copy host/port into `config.toml`, and
run `python desktop/gpu_llm.py tunnel` instead of `up`. In this mode `down`
only closes the tunnel — it does not know about a manually-rented instance,
so you still need to **destroy the instance in the Vast console** yourself.

## Instance-side scripts

| Script | Purpose |
|---|---|
| `bootstrap.sh` | onstart entrypoint: version check → GPU detect → model (`MODEL` or the GPU's default) → serve |
| `lib/detect_gpu.sh` | `nvidia-smi` → GPU id (= `profiles/<gpu>/`) |
| `lib/profile.py` | profile YAML → `vllm serve` argv; `list_models`/`default_model` catalog |
| `profiles/<gpu>/<model>.yaml` | one complete vLLM config per GPU × model; `profiles/README.md` has the matrix |
| `scripts/serve.sh <gpu> <model>` | optional `hf buckets sync`, launch vLLM with `--api-key`, manage `/var/log/vllm.status` |
| `scripts/health.sh` | `/health` + `/v1/models` (bearer `VLLM_API_KEY`) |
| `scripts/smoke.sh` | one thinking-mode completion against `local`; prints tok/s |

Status file values: `STARTING`, `READY`, `FAILED: <reason>`.
Bootstrap exit codes: 2 unknown GPU / model / no default, 3 `HF_TOKEN`
required, 4 vLLM < 0.17 or not importable, 5 `curl` missing, 6 bucket
download failed.

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

## Milestone 1 acceptance run (RTX 5090) — PASSED 2026-08-23 (pre-catalog: model was served as `qwen`)

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

## Milestone 2 acceptance run — PASSED 2026-08-23

- [x] `python desktop/gpu_llm.py up` — picker shown (US/CA + 1 Gbps filters
      applied), rented instance 48505562 at $0.396/hr, READY; `llm-cli
      health`/`models`/`chat` all worked through the tunnel.
- [x] `python desktop/gpu_llm.py status` showed `instance: 48505562
      ($0.396/hr)` plus tunnel/model/GPU lines.
- [x] `python desktop/gpu_llm.py down` — printed `instance 48505562
      destroyed.`; verified gone via `GET /api/v1/instances` (empty) and
      state.json cleared.
- [x] `up --yes --max-price 0.05` exited 1 with the 3 cheapest over-cap
      offers; rented nothing.
- [x] `GPU_FILTERS` spelling confirmed live: `RTX 5090` matched real offers.
- [x] Session metrics: ~9 tok/s single-stream generation during agent
      coding, 29.5 GB VRAM, 17% GPU util / 139 W. Cost: $0.396/hr (billed
      total in the Vast console).

## Using the model from JetBrains Rider

Rider's AI Assistant accepts OpenAI-compatible endpoints, which is exactly
what vLLM exposes through the tunnel. With `up` (or `tunnel`) showing READY:

1. **Settings | Tools | AI Assistant | Providers & API keys** → *Third-party
   AI providers* → **OpenAI Compatible**.
   - URL: `http://127.0.0.1:8000/v1` (if *Test Connection* fails, try
     `http://127.0.0.1:8000` — the docs don't say which form it wants).
   - API key: the `LLM_API_KEY` value from the banner. The field is
     mandatory in Rider and vLLM enforces it on `/v1/*`.
   - *Tool calling*: on (every profile passes vLLM's tool-call parser).
   - *Test Connection*, then *Apply*.
2. **Models Assignment**: set *Core features* to `local`. Set the model
   **context window to `LLM_CONTEXT`** from the banner — Rider's default
   is 64K, which is larger than the 5090 profile's 32K and will error.
   *Instant helpers* can also be `local`.
3. Skip the **AI Completion** section (inline completion). A 27B thinking
   model is slow and poorly suited to fill-in-the-middle; leave it on the
   JetBrains default or off.
4. Reasoning comes back as `reasoning_content`, which Rider ignores; only
   the final answer shows. Thinking still costs tokens and latency.

Known unknowns for the acceptance run: the exact URL form, and whether
AI Assistant's agent mode (MCP tools) works with a custom provider — the
docs say MCP tools are unsupported "with local models".

**Junie**: the IDE plugin has no custom-model setting. The Junie CLI does,
via a JSON profile (`$JUNIE_HOME/models/local.json` or `.junie/models/`):

```json
{
  "id": "local",
  "displayName": "Vast vLLM (local)",
  "apiType": "OpenAICompletion",
  "baseUrl": "http://127.0.0.1:8000/v1/chat/completions",
  "apiKey": "${LLM_API_KEY}",
  "maxContextLength": 32768
}
```

## Milestone 3 acceptance run — A100 80GB PASSED 2026-09-07 (instance 50203301, Czechia, $0.951/hr)

Run with the default `qwen3.8-27b-bf16`; the gpt-oss-120b and modded-fp8
profiles and the 5090 regression are still to be exercised (below).

- [x] `up --gpu a100-80 --model qwen3.8-27b-bf16 --offer 46878740 --yes` →
      READY in ~11 min cold (56 GB at ~1.9 Gbps; weights 3 min, torch.compile
      84 s, engine init 205 s). Banner printed `LLM_API_KEY` and
      `LLM_CONTEXT=131072`. Bootstrap log: `MODEL override=qwen3.8-27b-bf16`.
- [x] `status`: `models: local, qwen3.8-27b-bf16` and
      `model: qwen3.8-27b-bf16 (context 131072)`.
- [x] `/v1/models`: no bearer → 401, wrong bearer → 401, banner key → 200;
      `/health` open (200) as designed.
- [x] `scripts/smoke.sh` through the tunnel: `reasoning: ok`.
- [x] Tool calling: a `tools=[get_weather]` request returned
      `finish_reason: tool_calls` with parsed arguments (qwen3_coder parser).
- [x] `llm-cli health/models/chat --think/agent` via env overrides
      (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL=local`): agent read two
      files, fixed a bug, test passed.
- [x] Spectre001 bucket: `hf buckets ls` + `cp config.json` succeed with **no
      token** from the instance (`hf` 1.27.0 in the image); config carries a
      standard `quant_method: fp8` block on `Qwen3_5ForConditionalGeneration`.
      Full sync not yet timed.
- [ ] Rider: *Test Connection* URL form; chat with `local`; tool-calling
      toggle; agent mode. (Needs the IDE settings UI.)
- [ ] gpt-oss-120b on A100; modded-fp8 full sync + READY; 5090 regression.
- [ ] `down` destroys the instance (left up after this run for the Rider test).

Two desktop bugs were found and fixed during this run: the a100-80 offer
search matched 40 GB A100s (Vast names both "A100 SXM4"; now filtered by
`gpu_ram`), and `ssh_run` crashed on UTF-8 progress bars under Windows'
cp1252 default.

| Metric | Value |
|---|---|
| A100 BF16 time to READY | ~11 min cold (rent 21:14 → READY 21:25 UTC) |
| A100 BF16 tok/s, single stream | 17.6 (smoke, thinking on, 26 tok) / **27.7** (512 tok, thinking off) |
| A100 BF16 VRAM | 51.1 GiB weights; 74.7 GB reserved at 0.92 util; KV cache 580K tokens (4.4× concurrency at 131K — 262K would fit) |
| A100 idle draw | 59 W, 0 % util |
| modded-fp8 bucket public? | yes (no HF_TOKEN needed) |
| 5090 time to READY (regression) | |
| gpt-oss-120b on A100: time to READY / tok/s / peak VRAM | |
| modded-fp8 bucket sync time | |
| Rider URL form that passed Test Connection | |
| Rider agent mode with `local` | |

## Deferred (milestone 4)

FastAPI control plane, 3090/4090 profiles (need patched vLLM), persistent
weight volumes, `gpu-llm stats` subcommand (sample `/metrics` twice and
print live tok/s + queue depth + KV-cache/VRAM in one shot),
Qwen3.8-Flash-Next (needs multi-GPU or a newer image — see
`profiles/README.md`), MTP speculative decoding for the Qwen 27B profiles.
