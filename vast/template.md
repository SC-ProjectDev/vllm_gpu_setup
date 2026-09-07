# Vast.ai rental settings (Milestone 1)

| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.27.1` |
| Launch mode | **SSH** (interactive shell; do NOT use the image's default entrypoint) |
| Disk | the model's `disk_gb` from `profiles/README.md` (60 GB for the Qwen quants, 80 GB BF16, 100 GB gpt-oss-120b) |
| GPU filter | RTX 5090 (1x). Also works: A100 80GB, H100 80GB, H200 |
| Offer filters | Download speed >= 500 Mbps, reliability >= 98%, CUDA >= 12.8 |
| Ports | none (vLLM binds 127.0.0.1 only; everything goes through the SSH tunnel) |

## Environment variables (template "Docker options" / env section)

```
-e HF_TOKEN=            # optional; only for gated repos
-e PROFILE=             # optional GPU override: 5090 | a100-80 | h100-80 | h200
-e MODEL=               # optional model id under profiles/<gpu>/ (empty = that GPU's default)
-e VLLM_API_KEY=        # bearer token for /v1/*; use the value in %USERPROFILE%\.gpu-llm\llm_api_key
-e MAX_MODEL_LEN=       # optional; lower after an OOM
```

Without `VLLM_API_KEY` vLLM accepts any key, but `gpu-llm status` and Rider
still send the desktop's key, so setting it keeps the two flows identical.

## On-start script

Paste the contents of [`vast/onstart.sh`](onstart.sh) (one line). `gpu-llm up`
sends the same file automatically, so manual and API rentals cannot drift.

```
cd /root && git clone https://github.com/SC-ProjectDev/vllm_gpu_setup.git && cd vllm_gpu_setup && cp -n .env.example .env && nohup bash bootstrap.sh > /var/log/bootstrap.log 2>&1 &
```

If the GitHub repo is private, the plain https clone above will fail; for a private
repo, use a deploy token in the URL or bake an SSH deploy key into the
template's env and switch to the `git@` URL.

## What to expect

- `/var/log/bootstrap.log` — version check, detected profile.
- `/var/log/vllm.status` — `STARTING` → `READY`, or `FAILED: <reason>`.
- `/var/log/vllm.log` — vLLM output; weight download (~25 GB for NVFP4) shows here.
- Cold start on a fast host: 5–10 minutes to READY.

## Known 5090 settings

`profiles/5090/qwen3.8-27b-nvfp4.yaml` uses `Inferact/Qwen3.8-27B-NVFP4`, 32K context,
`--enforce-eager` (required to avoid CUDA-graph OOM on a single 32 GB card),
fp8 KV cache. Raise `MAX_MODEL_LEN` only after checking `nvidia-smi` headroom.
