# Profiles: `profiles/<gpu>/<model>.yaml`

One complete flat-YAML file per GPU × model pair. The directory is the GPU
id that `lib/detect_gpu.sh` prints (or `PROFILE=` overrides); the file stem
is the model id that `gpu-llm up --model` / `MODEL=` selects. Exactly one
file per directory carries `default: true`.

Every model is served under two names: `local` (what llm-cli and Rider are
configured with, so it never changes) and its own id.

## Matrix

| GPU | model id | HF source | quant | ~GB | max_model_len | disk_gb |
|---|---|---|---|---|---|---|
| 5090 | **qwen3.8-27b-nvfp4** (default) | Inferact/Qwen3.8-27B-NVFP4 | NVFP4 | 17 | 32768 | 60 |
| a100-80 | **qwen3.8-27b-bf16** (default) | Qwen/Qwen3.8-27B | BF16 | 56 | 131072 | 80 |
| a100-80 | qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 262144 | 60 |
| a100-80 | qwen3.8-27b-modded-fp8 | bucket Spectre001/Qwen3.8-27B-Modded-FP8-bucket | FP8 | 31 | 262144 | 60 |
| a100-80 | gpt-oss-120b | openai/gpt-oss-120b | MXFP4 | 63 | 65536 | 100 |
| h100-80 | **qwen3.8-27b-fp8** (default) | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 262144 | 60 |
| h100-80 | qwen3.8-27b-modded-fp8 | bucket (as above) | FP8 | 31 | 262144 | 60 |
| h100-80 | qwen3.8-27b-bf16 | Qwen/Qwen3.8-27B | BF16 | 56 | 131072 | 80 |
| h100-80 | gpt-oss-120b | openai/gpt-oss-120b | MXFP4 | 63 | 131072 | 100 |
| h200 | **qwen3.8-27b-bf16** (default) | Qwen/Qwen3.8-27B | BF16 | 56 | 262144 | 80 |
| h200 | qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 262144 | 60 |
| h200 | qwen3.8-27b-modded-fp8 | bucket (as above) | FP8 | 31 | 262144 | 60 |
| h200 | gpt-oss-120b | openai/gpt-oss-120b | MXFP4 | 63 | 131072 | 100 |

Context lengths are starting points; `MAX_MODEL_LEN` in `.env` overrides
without editing a profile. Verified: 5090 NVFP4 at 32K (49K also stable);
a100-80 BF16 at 131K (2026-09-07: 580K-token KV cache, so 262K would fit
too). The rest are to be confirmed live. The Spectre001 bucket is publicly
readable without a token (checked from an instance with `hf` 1.27.0).

## Keys

| Key | Meaning |
|---|---|
| `model` | HF repo id, or a local directory when `download` is set |
| `download` | optional `hf://buckets/...` URL; `serve.sh` runs `hf buckets sync <download> <model>` first |
| `served_model_name` | must equal the file stem (test-enforced); served alongside `local` |
| `quant`, `size_gb` | shown in the `gpu-llm up` picker; `size_gb` is approximate weights on disk |
| `disk_gb` | Vast container disk `gpu-llm up` requests for this model |
| `default` | `true` on exactly one file per GPU dir |
| `max_model_len`, `gpu_memory_utilization`, `kv_cache_dtype`, `enforce_eager`, `tensor_parallel_size`, `extra_args` | passed straight to `vllm serve` |
| `requires_hf_token` | bootstrap fails fast (exit 3) without `HF_TOKEN` |

## Adding a model

Copy the closest file in the same GPU dir, change `model`, `served_model_name`
(= new file stem), `quant`, `size_gb`, `disk_gb`, parsers in `extra_args`,
and `max_model_len`. Run `python -m pytest tests/test_profile.py`. Push
before `gpu-llm up` — the instance clones `main`.

## Not included: `Qwen/Qwen3.8-Flash-Next`

Despite the name it is a 125B MoE (6B active) plus a 51B n-gram embedding
table: 335 GiB BF16, 173 GiB FP8, ~74 GiB/rank NVFP4 before the table.
vLLM's smallest documented configuration is 4×H100 with the table offloaded
to 51 GB of host RAM, and it needs vLLM ≥ 0.29 (the pinned image is 0.27.1).
Nothing we rent as a single GPU can serve it. Revisit when a single-GPU
recipe and a matching image exist.

Also deferred: MTP speculative decoding (`--speculative-config
'{"method":"mtp","num_speculative_tokens":3}'`) for the Qwen 27B variants —
add after a clean acceptance run.
