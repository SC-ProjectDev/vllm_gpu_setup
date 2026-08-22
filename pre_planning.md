Steven:
Hey Claude so basically i was thinking of doing something like, renting a GPU on vast ai - could be either a 3090, 4090, 5090 depending on the avg pricing or if im doing something where i need power il usually get an a100 or an h100 or h200. Now the goal is this basically to have a disposable set up repo we bring into the instance and set up a local open weights LLM - usually depending on which GPU i rent or what level of work i might be doing. Now for starters wel aim for the 3090-5090 and start with a quant of qwen3.8 27b just to make sure the set up works and all that but the idea is we bring in vllm or ollama with a fast api set up that runs back to my desktop cli harness


Claude:
Yes — this is a solid design, and I’d make vLLM the primary backend rather than putting FastAPI in front of Ollama initially.
One small correction: I think the model you mean is Qwen3.5-27B, not Qwen3.8-27B. The official Qwen3.5-27B repo is about 55.6 GB at full precision, explicitly supports vLLM, and has a native 262K context window. Hugging Face
The architecture I’d aim for is:
YOUR DESKTOP
─────────────────────────────────────────────────
your CLI harness
      │
      │ OpenAI SDK
      │ base_url=http://127.0.0.1:8000/v1
      ▼
 SSH tunnel
 localhost:8000 ─────────────────────────┐
                                         │
                                         ▼
VAST.AI GPU INSTANCE
─────────────────────────────────────────────────
 disposable repo
 ├── bootstrap.sh
 ├── .env
 ├── configs/
 │    ├── 24gb.yaml
 │    ├── 32gb.yaml
 │    └── 80gb.yaml
 ├── scripts/
 │    ├── serve.sh
 │    ├── health.sh
 │    └── benchmark.sh
 └── compose.yaml
                                         │
                                         ▼
                              vLLM OpenAI server
                              127.0.0.1:8000
                                         │
                                         ▼
                                   Qwen3.5-27B
And importantly, I wouldn't actually add FastAPI yet.
vLLM itself already exposes an OpenAI-compatible server including /v1/chat/completions, /v1/completions, model listing, streaming, tool calling, etc. Your desktop harness can therefore treat the rented GPU exactly like an OpenAI-style provider simply by changing base_url. vLLM
So your desktop could literally do:
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local"
)

response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Implement this function..."}
    ],
)

print(response.choices[0].message.content)
That gives you a beautiful abstraction:
OpenAI
Anthropic
local Mac model
Vast 3090
Vast 5090
Vast H100
       │
       ▼
same CLI harness
The disposable-instance workflow
I would optimize the repo around this exact sequence:
# Vast instance starts

git clone git@github.com:you/vast-llm.git
cd vast-llm

cp .env.example .env

./bootstrap.sh qwen27b
And have bootstrap.sh detect the GPU automatically:
RTX 3090   → 24 GB profile
RTX 4090   → 24 GB profile
RTX 5090   → 32 GB profile
A100 40GB  → 40 GB profile
A100 80GB  → 80 GB profile
H100 80GB  → 80 GB profile
H200 141GB → 140 GB profile
Then:
nvidia-smi --query-gpu=name,memory.total \
           --format=csv,noheader
selects the appropriate config.
The repo shouldn't care which Vast machine you rented.
For the first milestone
I'd intentionally make version 1 extremely boring:
Vast instance
     ↓
Docker
     ↓
vllm/vllm-openai
     ↓
Qwen3.5-27B 4-bit
     ↓
127.0.0.1:8000
     ↓
SSH port forwarding
     ↓
your desktop
vLLM provides an official NVIDIA Docker image specifically for this use:
docker run \
    --gpus all \
    --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 127.0.0.1:8000:8000 \
    vllm/vllm-openai:latest \
    --model YOUR_QUANTIZED_MODEL
The official docs recommend the same vllm/vllm-openai container approach. vLLM
I'd pin the image version once we have the build working rather than leaving latest.
3090 / 4090 / 5090
This is where the quantization matters.
Qwen3.5-27B's published model files are ~55.6 GB, so FP16/BF16 isn't fitting on a 24 GB 3090/4090. Hugging Face
Conceptually:
                 VRAM
3090             24 GB
4090             24 GB
5090             32 GB

Qwen3.5-27B BF16 ~56 GB weights alone
                 ❌

Qwen3.5-27B 8bit ~28 GB-ish
                 ❌ 3090/4090
                 marginal 5090

Qwen3.5-27B 4bit ~14-18 GB-ish
                 ✓
The remaining VRAM is then available for KV/cache/runtime overhead.
For the first test I'd deliberately cap context much lower than Qwen's enormous native limit.
Something along the lines of:
3090 / 4090
Qwen3.5-27B 4-bit
16K–32K initial context

5090
Qwen3.5-27B 4-bit
32K-ish initial context
Then tune upward after measuring memory.
Qwen officially says the model supports 262,144 tokens natively, but also specifically warns that context length substantially affects memory consumption and recommends reducing it when OOM occurs. Hugging Face
For an agent/coding harness, I'd eventually target 32K–64K usable context on the consumer cards rather than chasing the headline 262K.
SSH rather than exposing vLLM
This part is important.
Don't do:
Internet
    ↓
Vast public IP:8000
    ↓
vLLM
Instead bind the model server to loopback and tunnel it.
From your desktop:
ssh \
  -N \
  -L 8000:127.0.0.1:8000 \
  -p "$VAST_SSH_PORT" \
  root@"$VAST_HOST"
Then locally:
curl http://127.0.0.1:8000/v1/models
Your CLI thinks the inference server is running on your desktop.
This is preferable to relying on vLLM's API key alone because vLLM's own documentation notes that its --api-key protection does not protect every inference endpoint. vLLM
So:
vLLM:
127.0.0.1:8000

SSH:
encrypted tunnel

public firewall:
no inference port
Much cleaner.
Where FastAPI eventually becomes useful
Once basic inference works, then I'd introduce your own tiny gateway.
Not because vLLM needs one, but because your harness eventually will want functionality independent of whichever inference backend is running.
For example:
Desktop harness
       │
       ▼
 FastAPI gateway
       │
       ├── /v1/chat/completions
       ├── /health
       ├── /gpu
       ├── /metrics
       ├── /model
       └── /shutdown
                │
                ▼
        inference backend
       ├── vLLM
       ├── Ollama
       └── SGLang
That gateway becomes your control plane.
Then your desktop doesn't have to know:
which GPU?
which model?
which quant?
which inference engine?
tensor parallel?
one GPU?
four GPUs?
It simply sees:
http://localhost:8000/v1
That's where this project gets really useful.
I would also support profiles instead of hardcoding models
Something like:
# profiles/qwen27b-24gb.yaml

model: your-qwen3.5-27b-quant
engine: vllm

max_model_len: 32768
gpu_memory_utilization: 0.90

served_model_name:
  - qwen
Versus:
# profiles/qwen27b-h100.yaml

model: Qwen/Qwen3.5-27B
engine: vllm

dtype: bfloat16
max_model_len: 131072
gpu_memory_utilization: 0.92

served_model_name:
  - qwen
Then:
./serve.sh qwen27b
could select the best implementation automatically.
Eventually:
./serve.sh coding
./serve.sh reasoning
./serve.sh vision
./serve.sh huge-context
would be even better.
The user-facing abstraction should be workload, not Hugging Face model ID.
And this gets much more interesting on H100/H200
Once the basic 3090 → desktop loop works, the same repo can scale into:
./bootstrap.sh
      ↓
detect:
4x H100 80GB
      ↓
choose:
large model
tensor-parallel=4
      ↓
launch vLLM
      ↓
same exact desktop URL
Your harness never changes.
That's the part I like most about the idea:
$ gpu-llm up coding
Rent:
RTX 5090
and later:
$ gpu-llm up heavy
Rent:
H100 / H200
while your actual agent environment continues talking to:
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local
vLLM is particularly suited to this because its OpenAI-compatible server is an explicit first-class part of the project rather than something we'd need to emulate ourselves. vLLM
So I'd build this in three milestones:
1. Vast → vLLM → Qwen3.5-27B quant → SSH tunnel → desktop curl/OpenAI SDK.
2. Add automatic GPU detection + model/config profiles + one-command bootstrap.
3. Add our FastAPI control plane, health/metrics, automatic teardown, and eventually Vast API provisioning so the desktop CLI can go from nothing → rented GPU → serving model automatically.
Once milestone 3 exists, you basically have your own ephemeral private LLM cloud where the hardware underneath can change every session without your local tooling caring about it.