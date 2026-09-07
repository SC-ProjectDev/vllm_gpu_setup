# vllm_gpu_setup — Milestone 3 design: model catalog, `local` alias, API key, Rider

2026-09-07. Follows the M2 design (`2026-08-23-vllm-gpu-setup-m2-design.md`).
M2 status: accepted on an RTX 5090 2026-08-23 (README).

## 1. Goal

Choose *which* model (and quant) a rented GPU serves, not only which GPU,
and make the endpoint consumable from JetBrains Rider's AI Assistant:

- `gpu-llm up --gpu a100-80 --model gpt-oss-120b`, or a numbered picker
  after the offer picker, chooses from the models that fit that GPU.
- Every model is served under the constant name `local` (plus its own id),
  so llm-cli and Rider configs never change when the model does.
- vLLM runs with a real `--api-key`; the desktop generates and prints it.
- README documents the Rider "OpenAI Compatible" provider setup.

Still deferred: FastAPI control plane, 3090/4090 profiles, persistent
weight volumes, multi-GPU offers, `gpu-llm stats`.

## 2. Model verdicts (researched 2026-09-07)

| Model | Verdict | Why |
|---|---|---|
| `Inferact/Qwen3.8-27B-NVFP4` | keep (5090 default) | Unchanged from M1/M2 |
| `Qwen/Qwen3.8-27B` (BF16, ~56 GB) | keep (a100-80 / h200 default) | Largest that fits 80 GB |
| `Qwen/Qwen3.8-27B-FP8` (~28 GB) | keep (h100-80 default), add to a100-80 / h200 | Official FP8; A100 runs it weight-only via Marlin |
| `Spectre001/Qwen3.8-27B-Modded-FP8-bucket` (~31 GB) | **add** to a100-80 / h100-80 / h200 | Abliterated + block-FP8 Qwen3.8-27B, Apache 2.0. It is a HF *storage bucket*, not a model repo: must be synced to disk with `hf buckets sync` and served from the local path. Its README passes `--trust-remote-code`. |
| `openai/gpt-oss-120b` (MXFP4, ~63 GB) | **add** to a100-80 / h100-80 / h200 | 117B MoE / 5B active. vLLM recipe: works on Ampere by default (Triton attention + Marlin MXFP4), fits one A100 80GB. Parsers `openai_gptoss` / `openai`. Needs a 100 GB disk. |
| `Qwen/Qwen3.8-Flash-Next` | **defer** (catalog note only) | 125B MoE + 51B n-gram table: 335 GiB BF16, 173 GiB FP8, ~74 GiB/rank NVFP4 + table. Smallest recipe config is 4×H100 with the table offloaded to 51 GB host RAM; needs vLLM ≥ 0.29 (image is 0.27.1). Does not fit any single GPU we rent. |

Context lengths per (GPU, model) below are starting points to be verified
in the acceptance run; `MAX_MODEL_LEN` still overrides.

## 3. Decisions

| Decision | Choice | Why |
|---|---|---|
| Catalog shape | `profiles/<gpu>/<model>.yaml`, each a complete flat YAML exactly as today | No parser or merge logic; each (GPU, model) pair is one tested file; only combos that fit exist |
| Default per GPU | `default: true` on exactly one file per GPU dir | Explicit; bootstrap needs no desktop input for manual rentals |
| Served names | `--served-model-name local <model-id>` | Stable client config; `status` still shows the real id |
| Desktop → instance channel | `export MODEL=… VLLM_API_KEY=…; ` prepended to the onstart text at rent time | onstart is already the only channel; `.env` loading never overrides an exported var, so the desktop choice wins |
| API key | `llm_api_key` in config.toml, `LLM_API_KEY` env, or auto-generated once into `~/.gpu-llm/llm_api_key` | Rider's key field is mandatory; generating avoids a config step. The key only matters inside the ssh tunnel (vLLM binds loopback), so it is a client-identification key, not a perimeter |
| Bucket models | `download:` key = `hf://buckets/...`; `model:` = local dir; serve.sh runs `hf buckets sync` before `vllm serve` | Only way to load a bucket; keeps profile.py a pure argv builder |
| Disk | `disk_gb` per profile (default 60) read by `up` at rent time | gpt-oss-120b (63 GB) and BF16 27B (56 GB) do not fit 60 GB |
| Speculative decoding (MTP) | not in this milestone | Optional per recipes; add after a clean acceptance |
| Image | stay on `vllm/vllm-openai:v0.27.1` | All shipped models supported; Flash-Next is the only thing needing newer |

## 4. Profile matrix

`profiles/<gpu>/<model>.yaml`. `served_model_name` must equal the file stem
(a test enforces it). New keys: `quant`, `size_gb` (approx weights on disk),
`disk_gb`, `default`, optional `download`, optional `notes`.

| GPU | model id | HF source | max_model_len | disk_gb | notes |
|---|---|---|---|---|---|
| 5090 | qwen3.8-27b-nvfp4 (default) | Inferact/Qwen3.8-27B-NVFP4 | 32768 | 60 | enforce_eager (unchanged) |
| a100-80 | qwen3.8-27b-bf16 (default) | Qwen/Qwen3.8-27B | 131072 | 80 | unchanged args |
| a100-80 | qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | 262144 | 60 | weight-only FP8 on Ampere |
| a100-80 | qwen3.8-27b-modded-fp8 | bucket Spectre001/… | 262144 | 60 | download + trust-remote-code |
| a100-80 | gpt-oss-120b | openai/gpt-oss-120b | 65536 | 100 | kv auto; Marlin MXFP4 |
| h100-80 | qwen3.8-27b-fp8 (default) | Qwen/Qwen3.8-27B-FP8 | 262144 | 60 | unchanged |
| h100-80 | qwen3.8-27b-modded-fp8 | bucket | 262144 | 60 | |
| h100-80 | qwen3.8-27b-bf16 | Qwen/Qwen3.8-27B | 131072 | 80 | |
| h100-80 | gpt-oss-120b | openai/gpt-oss-120b | 131072 | 100 | |
| h200 | qwen3.8-27b-bf16 (default) | Qwen/Qwen3.8-27B | 262144 | 80 | unchanged |
| h200 | qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | 262144 | 60 | |
| h200 | qwen3.8-27b-modded-fp8 | bucket | 262144 | 60 | |
| h200 | gpt-oss-120b | openai/gpt-oss-120b | 131072 | 100 | |

gpt-oss profiles: `kv_cache_dtype: auto`, `gpu_memory_utilization: 0.90`,
`extra_args: --reasoning-parser openai_gptoss`, `--enable-auto-tool-choice`,
`--tool-call-parser openai`. Qwen profiles keep `qwen3` / `qwen3_coder`.
Modded-FP8: `model: /root/models/qwen3.8-27b-modded-fp8`,
`download: hf://buckets/Spectre001/Qwen3.8-27B-Modded-FP8-bucket`, extra
`--trust-remote-code`.

`profiles/README.md` lists the matrix and records why Flash-Next is absent.

## 5. Instance side

- `lib/profile.py`: `profile_to_argv` emits `--served-model-name local <served_model_name>`
  (no other argv change). New `list_models(profiles_root, gpu) -> list[dict]`
  (each: `id`, `default`, `quant`, `size_gb`, `max_model_len`, `disk_gb`,
  `model`) sorted default-first then by id, and `default_model(profiles_root, gpu) -> str`
  (raises `ValueError` if there is not exactly one default).
- `bootstrap.sh`: reads `MODEL` and `VLLM_API_KEY` from env/.env like the
  other vars. After `PROFILE` is known, `profiles/$PROFILE/` must be a dir
  (else `FAILED: no profile '<id>'`, exit 2). If `MODEL` is empty, pick the
  single file with `^default: *true`; if none/many, `FAILED: no default model for <gpu>`,
  exit 2. Then `profiles/$PROFILE/$MODEL.yaml` must exist (else
  `FAILED: no model '<id>' for <gpu>`, exit 2). HF-token check reads that
  file. Logs `[bootstrap] model=<id>`. Execs `serve.sh <gpu> <model>`.
- `scripts/serve.sh <gpu> <model>`: profile path `profiles/<gpu>/<model>.yaml`.
  If the profile has a `download:` key, runs
  `"$HF_BIN" buckets sync "<download>" "<model dir>"` (log to vllm.log,
  status stays `STARTING`, log line `[serve] syncing <download>`); non-zero →
  `FAILED: download failed (rc N)`, exit 6. Appends `--api-key "$VLLM_API_KEY"`
  when the var is non-empty.
- `scripts/health.sh`, `scripts/smoke.sh`: send `Authorization: Bearer $VLLM_API_KEY`
  on `/v1/*` when set. `smoke.sh` default `MODEL` becomes `local`.
- `.env.example` and `vast/template.md` gain `MODEL=` and `VLLM_API_KEY=`
  with a one-line explanation each. `vast/onstart.sh` is unchanged.

## 6. Desktop side (`desktop/gpu_llm.py`)

- `up --model <id>` (optional). Flow after the offer is chosen and before
  renting: build the model list for `args.gpu` from the local checkout's
  `profiles/`. If `--model` is given it must be in the list (else print the
  list and exit 1, nothing rented). Otherwise, if there is one model, use it
  silently; if more, print a numbered list (`1. qwen3.8-27b-bf16  BF16  56 GB  131072 ctx  (default)`)
  and prompt `model? [1-N, Enter = default]`; `--yes` takes the default.
- Disk: `rent_offer(..., disk=profile["disk_gb"] or 60, ...)`.
- Onstart text: `export MODEL='<id>' VLLM_API_KEY='<key>'; ` + file content.
  Ids and keys are validated to `[A-Za-z0-9._-]+` before quoting.
- API key: `resolve_llm_api_key(cfg)` = `LLM_API_KEY` env → `cfg["llm_api_key"]`
  → `~/.gpu-llm/llm_api_key` file → generate `secrets.token_urlsafe(24)`,
  write the file (mode 0600 on POSIX), return it.
- State gains `model`. `status` prints `model: <id> (context N)` when known
  and sends the bearer header on `/v1/models`.
- READY banner (shared by `up` and `tunnel`) prints:
  `LLM_BASE_URL=http://127.0.0.1:<port>`, `LLM_MODEL=local`,
  `LLM_API_KEY=<key>`, and `LLM_CONTEXT=<max_model_len>` when the state has
  gpu+model.
- Requires the local checkout and the instance clone to agree on
  `profiles/` — same constraint as bootstrap itself; README says push first.

## 7. Rider (documentation + acceptance)

README section "Using the model from JetBrains Rider":

- Settings | Tools | AI Assistant | Providers & API keys → Third-party AI
  providers → OpenAI Compatible. URL `http://127.0.0.1:8000/v1` (fall back to
  `http://127.0.0.1:8000` if Test Connection fails), API key = `LLM_API_KEY`
  from the banner, Tool calling on.
- Models Assignment: Core = `local`; set the context window to the printed
  `LLM_CONTEXT` (default 64K exceeds the 5090's 32K).
- Skip AI Completion (a 27B thinking model is a poor inline-completion fit).
- Junie IDE plugin has no custom-model support; Junie CLI does via a JSON
  profile (`apiType: OpenAICompletion`, `baseUrl` = full `/v1/chat/completions`
  URL, `apiKey: ${LLM_API_KEY}`, `id: local`).
- Known unknowns for the acceptance run: exact URL form, agent mode with a
  custom provider.

## 8. Error handling

| Failure | Where | Behaviour |
|---|---|---|
| `--model` not in the GPU's list | `up` | list printed, exit 1, nothing rented |
| No/many `default: true` in a GPU dir | bootstrap | `FAILED: no default model for <gpu>`, exit 2 |
| Model file missing on instance (desktop/instance clone drift) | bootstrap | `FAILED: no model '<id>' for <gpu>`, exit 2 — visible in `up`'s progress line |
| `hf buckets sync` fails / `hf` missing | serve.sh | `FAILED: download failed (rc N)`, exit 6, log tail printed |
| Bad key from client | vLLM | 401 on `/v1/*`; `status` prints `models: unauthorized (check LLM_API_KEY)` on 401 |
| Key file unwritable | `up` | key still used for this run; warning printed |

## 9. Testing

Existing suites keep passing with path updates. New/changed:

- `test_profile.py`: every shipped profile parses; `served_model_name == stem`;
  exactly one default per GPU dir; argv for a gpt-oss and a bucket profile;
  `local` alias present; `list_models`/`default_model` including the error cases.
- `test_serve.py`: two-arg form; bucket profile runs fake `hf buckets sync`
  before vllm; sync failure → `FAILED: download failed`; `--api-key` present
  only when `VLLM_API_KEY` set.
- `test_bootstrap.py`: default resolution; `MODEL` override; missing model;
  no-default dir; passes `<gpu> <model>` to serve.
- `test_smoke.py` / health: bearer header sent when key set.
- `test_gpu_llm.py`: `--model` accepted/rejected; picker default on Enter;
  `--yes` picks default; onstart text starts with the export line; disk from
  profile; key resolution order + generation; banner lines; `status` model
  line and 401 message.
- `test_onstart.py`: unchanged (file is unchanged).

## 10. Acceptance (M3)

1. 5090, default model: `up` → READY; `llm-cli models` shows `local` and
   `qwen3.8-27b-nvfp4`; unauthenticated `/v1/models` returns 401; banner key works.
2. A100 80GB: `up --gpu a100-80 --model gpt-oss-120b` → READY within the
   timeout; smoke.sh passes (reasoning + tool parsers); record VRAM and tok/s.
3. A100 80GB: `--model qwen3.8-27b-modded-fp8` → bucket syncs, READY; record
   sync time and whether the bucket needed a token.
4. Rider: provider Test Connection succeeds; chat with `local` streams; tool
   calling toggle behaviour recorded; agent mode result recorded.
5. `down` destroys as before.

Budget: ~1 h on the 5090, ~1.5 h on the A100.
