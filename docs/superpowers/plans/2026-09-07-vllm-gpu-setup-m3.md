# vllm_gpu_setup M3 Implementation Plan — model catalog, `local` alias, API key, Rider

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `gpu-llm up` choose which model/quant a rented GPU serves, serve every model as `local` behind a real vLLM `--api-key`, and document Rider's OpenAI-compatible setup.

**Architecture:** Profiles move to `profiles/<gpu>/<model>.yaml` (one complete flat YAML per GPU×model pair; one `default: true` per GPU). The desktop picks a model from the local checkout, prepends `export MODEL=… VLLM_API_KEY=…;` to the onstart text, and sizes the Vast disk from the profile. Bootstrap resolves the model (env or default), serve.sh optionally syncs a Hugging Face bucket and passes `--api-key`. Clients use bearer auth.

**Tech Stack:** Python 3.11 stdlib (`tomllib`, `secrets`, `urllib`), bash, pytest; vLLM image `vllm/vllm-openai:v0.27.1`; `hf` CLI (huggingface_hub ≥ 1.5) for buckets.

**Spec:** `docs/superpowers/specs/2026-09-07-vllm-gpu-setup-m3-model-catalog-design.md`

## Global Constraints

- Stdlib only on the desktop and in `lib/profile.py`; no new dependencies.
- Image stays `vllm/vllm-openai:v0.27.1`; `MIN_VLLM` stays `0.17`.
- `served_model_name` in every profile equals its file stem; argv is `--served-model-name local <stem>`.
- Exactly one `default: true` file per `profiles/<gpu>/` directory.
- `vast/onstart.sh` is byte-identical before and after (test_onstart enforces the template match).
- Model ids and API keys must match `^[A-Za-z0-9._-]+$` before being quoted into onstart.
- Tests run with `python -m pytest` from the repo root; bash tests need Git Bash on PATH.
- Commit trailer on every commit:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01PD9wokT4vgp7DXUnFWgqCJ`.

## File Structure

| File | Responsibility |
|---|---|
| `profiles/<gpu>/<model>.yaml` (13 files) | complete vLLM settings for one GPU×model pair |
| `profiles/README.md` | the matrix + why Flash-Next is absent |
| `lib/profile.py` | YAML parse, argv build (`local` alias), `list_models`, `default_model` |
| `scripts/serve.sh` | `serve.sh <gpu> <model>`: optional bucket sync, `--api-key`, status file |
| `bootstrap.sh` | env → version → GPU → **model** → token → serve |
| `scripts/health.sh`, `scripts/smoke.sh` | bearer header; smoke default model `local` |
| `desktop/gpu_llm.py` | `--model`, model picker, key resolution, onstart export, disk, banner, status |
| `.env.example`, `vast/template.md`, `README.md` | docs for `MODEL`, `VLLM_API_KEY`, Rider, M3 acceptance |

---

### Task 1: Profile matrix + `lib/profile.py` (`local` alias, `list_models`, `default_model`)

**Files:**
- Move: `profiles/5090.yaml` → `profiles/5090/qwen3.8-27b-nvfp4.yaml`; `profiles/a100-80.yaml` → `profiles/a100-80/qwen3.8-27b-bf16.yaml`; `profiles/h100-80.yaml` → `profiles/h100-80/qwen3.8-27b-fp8.yaml`; `profiles/h200.yaml` → `profiles/h200/qwen3.8-27b-bf16.yaml`
- Create: 9 more profiles (matrix below), `profiles/README.md`
- Modify: `lib/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Produces: `list_models(profiles_root: str|Path, gpu: str) -> list[dict]` with keys `id, default, quant, size_gb, max_model_len, disk_gb, model`, sorted default-first then by id; raises `ValueError` if `profiles_root/gpu` is not a dir. `default_model(profiles_root, gpu) -> str`; raises `ValueError` unless exactly one default. `profile_to_argv` unchanged signature, emits `--served-model-name local <served_model_name>`.

- [ ] **Step 1: Write the failing tests** (replace `test_argv_5090`, `test_argv_omits_enforce_eager_when_false`, `test_max_model_len_override` path usages and add new tests)

```python
# tests/test_profile.py  (additions / replacements)
import pytest
from lib.profile import load_profile, profile_to_argv, list_models, default_model

GPUS = ("5090", "a100-80", "h100-80", "h200")
ALL_PROFILES = sorted(p for g in GPUS for p in (PROFILES / g).glob("*.yaml"))


def test_argv_5090():
    argv = profile_to_argv(load_profile(PROFILES / "5090" / "qwen3.8-27b-nvfp4.yaml"), {})
    assert argv == [
        "Inferact/Qwen3.8-27B-NVFP4",
        "--served-model-name", "local", "qwen3.8-27b-nvfp4",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.9",
        "--kv-cache-dtype", "fp8",
        "--tensor-parallel-size", "1",
        "--enforce-eager",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]


def test_argv_omits_enforce_eager_when_false():
    argv = profile_to_argv(load_profile(PROFILES / "h100-80" / "qwen3.8-27b-fp8.yaml"), {})
    assert "--enforce-eager" not in argv
    assert argv[0] == "Qwen/Qwen3.8-27B-FP8"
    assert argv[argv.index("--max-model-len") + 1] == "262144"


def test_max_model_len_override():
    argv = profile_to_argv(load_profile(PROFILES / "5090" / "qwen3.8-27b-nvfp4.yaml"), {"max_model_len": "16384"})
    assert argv[argv.index("--max-model-len") + 1] == "16384"


def test_argv_gpt_oss_uses_openai_parsers_and_auto_kv():
    argv = profile_to_argv(load_profile(PROFILES / "a100-80" / "gpt-oss-120b.yaml"), {})
    assert argv[0] == "openai/gpt-oss-120b"
    assert argv[argv.index("--kv-cache-dtype") + 1] == "auto"
    assert argv[argv.index("--reasoning-parser") + 1] == "openai_gptoss"
    assert argv[argv.index("--tool-call-parser") + 1] == "openai"
    assert argv[argv.index("--max-model-len") + 1] == "65536"


def test_bucket_profile_serves_local_path_with_download():
    p = load_profile(PROFILES / "a100-80" / "qwen3.8-27b-modded-fp8.yaml")
    assert p["download"] == "hf://buckets/Spectre001/Qwen3.8-27B-Modded-FP8-bucket"
    assert p["model"] == "/root/models/qwen3.8-27b-modded-fp8"
    assert "--trust-remote-code" in profile_to_argv(p, {})


@pytest.mark.parametrize("path", ALL_PROFILES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_profile_is_complete_and_named_after_its_file(path):
    p = load_profile(path)
    assert p["served_model_name"] == path.stem
    for key in ("model", "max_model_len", "gpu_memory_utilization", "kv_cache_dtype",
                "tensor_parallel_size", "quant", "size_gb", "disk_gb"):
        assert key in p, f"{path} missing {key}"
    assert isinstance(p["disk_gb"], int) and p["disk_gb"] >= 60


@pytest.mark.parametrize("gpu", GPUS)
def test_exactly_one_default_per_gpu(gpu):
    assert default_model(PROFILES, gpu) in {m["id"] for m in list_models(PROFILES, gpu)}
    assert sum(m["default"] for m in list_models(PROFILES, gpu)) == 1


def test_list_models_sorted_default_first_then_id():
    ids = [m["id"] for m in list_models(PROFILES, "a100-80")]
    assert ids == ["qwen3.8-27b-bf16", "gpt-oss-120b", "qwen3.8-27b-fp8", "qwen3.8-27b-modded-fp8"]
    m = list_models(PROFILES, "a100-80")[1]
    assert m == {"id": "gpt-oss-120b", "default": False, "quant": "MXFP4", "size_gb": 63,
                 "max_model_len": 65536, "disk_gb": 100, "model": "openai/gpt-oss-120b"}


def test_list_models_unknown_gpu_raises():
    with pytest.raises(ValueError):
        list_models(PROFILES, "4090")


def test_default_model_requires_exactly_one(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "a.yaml").write_text("model: a\nserved_model_name: a\n")
    (tmp_path / "g" / "b.yaml").write_text("model: b\nserved_model_name: b\n")
    with pytest.raises(ValueError):
        default_model(tmp_path, "g")
    (tmp_path / "g" / "a.yaml").write_text("model: a\nserved_model_name: a\ndefault: true\n")
    assert default_model(tmp_path, "g") == "a"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_profile.py -q`
Expected: ImportError on `list_models` / FileNotFoundError on new paths.

- [ ] **Step 3: Move and create the profiles**

`git mv profiles/5090.yaml profiles/5090/qwen3.8-27b-nvfp4.yaml` (etc. for the four). Then edit/create so each file is:

```yaml
# profiles/5090/qwen3.8-27b-nvfp4.yaml
model: Inferact/Qwen3.8-27B-NVFP4
served_model_name: qwen3.8-27b-nvfp4
quant: NVFP4
size_gb: 17
disk_gb: 60
default: true
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

Qwen variants (same block, differing lines only):

| file | model | quant | size_gb | disk_gb | default | max_model_len | enforce_eager | gpu_mem | extra |
|---|---|---|---|---|---|---|---|---|---|
| a100-80/qwen3.8-27b-bf16 | Qwen/Qwen3.8-27B | BF16 | 56 | 80 | true | 131072 | false | 0.92 | |
| a100-80/qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 60 | | 262144 | false | 0.92 | |
| a100-80/qwen3.8-27b-modded-fp8 | /root/models/qwen3.8-27b-modded-fp8 | FP8 | 31 | 60 | | 262144 | false | 0.92 | `download:` + `--trust-remote-code` |
| h100-80/qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 60 | true | 262144 | false | 0.92 | |
| h100-80/qwen3.8-27b-modded-fp8 | (bucket) | FP8 | 31 | 60 | | 262144 | false | 0.92 | as above |
| h100-80/qwen3.8-27b-bf16 | Qwen/Qwen3.8-27B | BF16 | 56 | 80 | | 131072 | false | 0.92 | |
| h200/qwen3.8-27b-bf16 | Qwen/Qwen3.8-27B | BF16 | 56 | 80 | true | 262144 | false | 0.92 | |
| h200/qwen3.8-27b-fp8 | Qwen/Qwen3.8-27B-FP8 | FP8 | 28 | 60 | | 262144 | false | 0.92 | |
| h200/qwen3.8-27b-modded-fp8 | (bucket) | FP8 | 31 | 60 | | 262144 | false | 0.92 | as above |

Bucket profile (a100-80 shown; h100-80/h200 identical):

```yaml
# profiles/a100-80/qwen3.8-27b-modded-fp8.yaml
# Abliterated + block-FP8 Qwen3.8-27B from a Hugging Face *bucket*; serve.sh
# syncs `download` into `model` before launching vLLM.
model: /root/models/qwen3.8-27b-modded-fp8
download: hf://buckets/Spectre001/Qwen3.8-27B-Modded-FP8-bucket
served_model_name: qwen3.8-27b-modded-fp8
quant: FP8
size_gb: 31
disk_gb: 60
max_model_len: 262144
gpu_memory_utilization: 0.92
kv_cache_dtype: fp8
enforce_eager: false
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --trust-remote-code
  - --reasoning-parser qwen3
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
```

gpt-oss profile (a100-80 `max_model_len: 65536`; h100-80 and h200 `131072`):

```yaml
# profiles/a100-80/gpt-oss-120b.yaml
# 117B MoE / 5B active, native MXFP4 (~63 GB). Ampere uses Triton attention +
# Marlin MXFP4 automatically (vLLM recipe). Harmony parsers below.
model: openai/gpt-oss-120b
served_model_name: gpt-oss-120b
quant: MXFP4
size_gb: 63
disk_gb: 100
max_model_len: 65536
gpu_memory_utilization: 0.90
kv_cache_dtype: auto
enforce_eager: false
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --reasoning-parser openai_gptoss
  - --enable-auto-tool-choice
  - --tool-call-parser openai
```

- [ ] **Step 4: Implement `lib/profile.py` changes**

```python
# in profile_to_argv, replace the served-model-name line:
        "--served-model-name", "local", str(profile["served_model_name"]),

# new functions (after profile_to_argv):
MODEL_KEYS = ("quant", "size_gb", "max_model_len", "disk_gb", "model")


def list_models(profiles_root: str | Path, gpu: str) -> list[dict]:
    """Catalog entries for one GPU dir, default first then by id."""
    d = Path(profiles_root) / gpu
    if not d.is_dir():
        raise ValueError(f"no profiles for gpu {gpu!r} under {profiles_root}")
    out = []
    for f in sorted(d.glob("*.yaml")):
        p = load_profile(f)
        entry = {"id": f.stem, "default": p.get("default") is True}
        for k in MODEL_KEYS:
            entry[k] = p.get(k)
        out.append(entry)
    out.sort(key=lambda m: (not m["default"], m["id"]))
    return out


def default_model(profiles_root: str | Path, gpu: str) -> str:
    defaults = [m["id"] for m in list_models(profiles_root, gpu) if m["default"]]
    if len(defaults) != 1:
        raise ValueError(f"expected exactly one default model for {gpu}, found {len(defaults)}")
    return defaults[0]
```

Note `entry["quant"]` may be `None` when absent; `test_list_models_sorted_default_first_then_id` expects the exact dict shown, so every shipped profile carries `quant`.

- [ ] **Step 5: Write `profiles/README.md`** — the matrix table from the spec §4, the key glossary (`default`, `download`, `disk_gb`, `size_gb`), and this paragraph:

> **Not included: `Qwen/Qwen3.8-Flash-Next`.** 125B MoE plus a 51B n-gram embedding table: 335 GiB BF16, 173 GiB FP8, ~74 GiB/rank NVFP4 before the table. vLLM's smallest documented config is 4×H100 with the table offloaded to 51 GB host RAM, and it needs vLLM ≥ 0.29 (image is 0.27.1). Nothing we rent single-GPU can serve it. Revisit when a single-GPU recipe and image exist.

- [ ] **Step 6: Run tests** — `python -m pytest tests/test_profile.py -q` → all pass. (`test_serve.py`/`test_bootstrap.py` now fail on old paths; Tasks 2–3 fix them.)

- [ ] **Step 7: Commit** — `git add profiles lib/profile.py tests/test_profile.py && git commit -m "feat: profiles/<gpu>/<model> catalog, local alias, list_models/default_model"`

---

### Task 2: `serve.sh <gpu> <model>` — bucket sync + `--api-key`

**Files:**
- Modify: `scripts/serve.sh`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: profile paths from Task 1; profile `download:` key.
- Produces: CLI `serve.sh <gpu> <model>`; env `HF_BIN` (default `hf`), `VLLM_API_KEY`; exit 6 + `FAILED: download failed (rc N)` on sync failure.

- [ ] **Step 1: Update existing tests to the two-arg form and add new ones**

Replace every `run_bash(SERVE, "5090", env=env)` with `run_bash(SERVE, "5090", "qwen3.8-27b-nvfp4", env=env)`; in `test_serve_success_writes_ready_and_passes_argv` change the served-name assertion to `assert "--served-model-name local qwen3.8-27b-nvfp4" in args` and add `assert "--api-key" not in args`. In `test_serve_unknown_profile_fails` call `run_bash(SERVE, "5090", "nope", env=env)`. Add:

```python
def test_serve_passes_api_key_when_set(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "ARGS: $*"; sleep 1\n')
    write_fake(fakes, "curl", "exit 0\n")
    env, status, log = env_for(tmp_path, fakes, VLLM_API_KEY="sekrit")
    r = run_bash(SERVE, "5090", "qwen3.8-27b-nvfp4", env=env)
    assert r.returncode == 0, r.stderr
    assert "--api-key sekrit" in log.read_text()


def test_serve_syncs_bucket_before_vllm(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "ARGS: $*"; sleep 1\n')
    write_fake(fakes, "curl", "exit 0\n")
    write_fake(fakes, "hf", 'echo "HF $*"\n')
    env, status, log = env_for(tmp_path, fakes, HF_BIN=str(fakes / "hf"))
    r = run_bash(SERVE, "a100-80", "qwen3.8-27b-modded-fp8", env=env)
    assert r.returncode == 0, r.stderr
    text = log.read_text()
    assert "HF buckets sync hf://buckets/Spectre001/Qwen3.8-27B-Modded-FP8-bucket /root/models/qwen3.8-27b-modded-fp8" in text
    assert text.index("HF buckets sync") < text.index("ARGS: serve /root/models/qwen3.8-27b-modded-fp8")
    assert status.read_text().strip() == "READY"


def test_serve_bucket_sync_failure_writes_failed_exit_6(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "SHOULD NOT RUN"; exit 0\n')
    write_fake(fakes, "curl", "exit 0\n")
    write_fake(fakes, "hf", 'echo "403 forbidden" >&2; exit 3\n')
    env, status, log = env_for(tmp_path, fakes, HF_BIN=str(fakes / "hf"))
    r = run_bash(SERVE, "a100-80", "qwen3.8-27b-modded-fp8", env=env)
    assert r.returncode == 6
    assert status.read_text().strip() == "FAILED: download failed (rc 3)"
    assert "SHOULD NOT RUN" not in log.read_text()
    assert "403 forbidden" in r.stdout          # log tail printed


def test_serve_non_bucket_profile_never_calls_hf(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "ARGS: $*"; sleep 1\n')
    write_fake(fakes, "curl", "exit 0\n")
    write_fake(fakes, "hf", 'echo "HF CALLED"; exit 1\n')
    env, status, log = env_for(tmp_path, fakes, HF_BIN=str(fakes / "hf"))
    r = run_bash(SERVE, "5090", "qwen3.8-27b-nvfp4", env=env)
    assert r.returncode == 0, r.stderr
    assert "HF CALLED" not in log.read_text()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_serve.py -q` → new tests fail (exit 2, no profile).

- [ ] **Step 3: Implement** — replace the argument/profile block and the launch line in `scripts/serve.sh`:

```bash
# Usage: serve.sh <gpu-id> <model-id>. Launches vllm serve on 127.0.0.1:8000 and manages the status file.
...
HF_BIN="${HF_BIN:-hf}"

gpu_id="${1:-}"
model_id="${2:-}"
profile_file="$REPO_ROOT/profiles/${gpu_id}/${model_id}.yaml"
if [[ -z "$gpu_id" || -z "$model_id" || ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '${gpu_id}/${model_id}' (expected $REPO_ROOT/profiles/<gpu>/<model>.yaml)"
  exit 2
fi

mapfile -t argv < <(python3 "$REPO_ROOT/lib/profile.py" "$profile_file" | tr -d '\r')
mkdir -p "$(dirname "$VLLM_LOG_FILE")"
write_status "STARTING"
echo "[serve] profile=$gpu_id/$model_id" | tee -a "$VLLM_LOG_FILE"

report_failure() { ...unchanged... }

# Optional bucket sync: `download: hf://buckets/...` -> local `model:` path.
download="$(sed -n 's/^download:[[:space:]]*//p' "$profile_file" | head -n 1 | tr -d '\r')"
if [[ -n "$download" ]]; then
  echo "[serve] syncing $download -> ${argv[0]}" | tee -a "$VLLM_LOG_FILE"
  if ! "$HF_BIN" buckets sync "$download" "${argv[0]}" >> "$VLLM_LOG_FILE" 2>&1; then
    code=$?
    write_status "FAILED: download failed (rc $code)"
    echo "[serve] last 40 log lines:"
    tail -n 40 "$VLLM_LOG_FILE"
    exit 6
  fi
fi

api_args=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then api_args=(--api-key "$VLLM_API_KEY"); fi

"$VLLM_BIN" serve "${argv[@]}" ${api_args[@]+"${api_args[@]}"} --host 127.0.0.1 --port 8000 >> "$VLLM_LOG_FILE" 2>&1 &
```

- [ ] **Step 4: Run** `python -m pytest tests/test_serve.py -q` → pass.
- [ ] **Step 5: Commit** — `git commit -am "feat: serve.sh <gpu> <model>, bucket sync, --api-key"`

---

### Task 3: bootstrap model resolution, `.env.example`, template env

**Files:**
- Modify: `bootstrap.sh`, `.env.example`, `vast/template.md`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `profiles/<gpu>/` dirs, `default: true`, `serve.sh <gpu> <model>` (Task 2).
- Produces: env `MODEL`, `VLLM_API_KEY` honoured from env/.env; statuses `FAILED: no profile '<gpu>'`, `FAILED: no default model for <gpu> (found N)`, `FAILED: no model '<id>' for <gpu>` (all exit 2).

- [ ] **Step 1: Update tests**

In `setup()` add `"MODEL": ""` to the env dict. Change expectations: `test_happy_path_detects_and_serves` → `"SERVE 5090 qwen3.8-27b-nvfp4"`; `test_profile_env_override_skips_detection` → `"SERVE h200 qwen3.8-27b-bf16"`; the three dotenv tests → `"SERVE a100-80 qwen3.8-27b-bf16"`. Rewrite the gated test and add three:

```python
import shutil

def test_requires_hf_token_exits_3(tmp_path, fakes):
    gated_dir = ROOT / "profiles" / "zz-gated-test"
    gated_dir.mkdir()
    (gated_dir / "m.yaml").write_text(
        (ROOT / "profiles" / "5090" / "qwen3.8-27b-nvfp4.yaml").read_text()
        .replace("requires_hf_token: false", "requires_hf_token: true"))
    try:
        env, status = setup(tmp_path, fakes)
        env["PROFILE"] = "zz-gated-test"
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 3
        assert status.read_text().startswith("FAILED: HF_TOKEN required")
    finally:
        shutil.rmtree(gated_dir)


def test_model_env_override_selects_profile(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="a100-80 1")
    env["MODEL"] = "gpt-oss-120b"
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "SERVE a100-80 gpt-oss-120b" in r.stdout
    assert "MODEL override=gpt-oss-120b" in r.stdout


def test_unknown_model_fails_2(tmp_path, fakes):
    env, status = setup(tmp_path, fakes)
    env["MODEL"] = "nope"
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 2
    assert status.read_text().strip() == "FAILED: no model 'nope' for 5090"


def test_gpu_dir_without_default_fails_2(tmp_path, fakes):
    d = ROOT / "profiles" / "zz-nodefault-test"
    d.mkdir()
    (d / "a.yaml").write_text("model: a\nserved_model_name: a\nrequires_hf_token: false\n")
    try:
        env, status = setup(tmp_path, fakes)
        env["PROFILE"] = "zz-nodefault-test"
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 2
        assert status.read_text().strip() == "FAILED: no default model for zz-nodefault-test (found 0)"
    finally:
        shutil.rmtree(d)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_bootstrap.py -q` → failures.

- [ ] **Step 3: Implement in `bootstrap.sh`**

Export line becomes:
```bash
export HF_TOKEN="${HF_TOKEN:-}" PROFILE="${PROFILE:-}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" \
       MODEL="${MODEL:-}" VLLM_API_KEY="${VLLM_API_KEY:-}"
```
Replace section 4 with:
```bash
# 4. model (env/.env MODEL, else the single `default: true` file) + token
profile_dir="$REPO_ROOT/profiles/${PROFILE}"
if [[ ! -d "$profile_dir" ]]; then
  write_status "FAILED: no profile '$PROFILE'"; exit 2
fi
if [[ -z "$MODEL" ]]; then
  mapfile -t defaults < <(grep -lE '^default:[[:space:]]*true' "$profile_dir"/*.yaml 2>/dev/null || true)
  if [[ ${#defaults[@]} -ne 1 ]]; then
    write_status "FAILED: no default model for $PROFILE (found ${#defaults[@]})"; exit 2
  fi
  MODEL="$(basename "${defaults[0]}" .yaml)"
  echo "[bootstrap] model=$MODEL (default for $PROFILE)"
else
  echo "[bootstrap] MODEL override=$MODEL"
fi
profile_file="$profile_dir/$MODEL.yaml"
if [[ ! -f "$profile_file" ]]; then
  write_status "FAILED: no model '$MODEL' for $PROFILE"; exit 2
fi
if grep -qE '^requires_hf_token:\s*true' "$profile_file" && [[ -z "$HF_TOKEN" ]]; then
  write_status "FAILED: HF_TOKEN required by profile $PROFILE/$MODEL"; exit 3
fi
[[ -n "$HF_TOKEN" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```
and the final line `exec bash "$SERVE" "$PROFILE" "$MODEL"`. Update the header comment to `env -> vLLM version check -> GPU detect -> model -> token check -> serve`.

- [ ] **Step 4: `.env.example`** — append:
```
# Model id under profiles/<gpu>/ (e.g. gpt-oss-120b). Empty = that GPU's default. `gpu-llm up` sets this.
MODEL=
# Bearer token vLLM requires on /v1/*. `gpu-llm up` sets this to your ~/.gpu-llm/llm_api_key.
VLLM_API_KEY=
```
`vast/template.md` env block gains `-e MODEL=` and `-e VLLM_API_KEY=` with the same comments, and the "Known 5090 settings" paragraph points at `profiles/5090/qwen3.8-27b-nvfp4.yaml`. Update the PROFILE comment in both to say the gpu id.

- [ ] **Step 5: Run** `python -m pytest tests/test_bootstrap.py tests/test_onstart.py -q` → pass.
- [ ] **Step 6: Commit** — `git commit -am "feat: bootstrap resolves MODEL (env or per-GPU default); MODEL/VLLM_API_KEY in .env + template"`

---

### Task 4: bearer auth in `health.sh` / `smoke.sh`; smoke default `local`

**Files:**
- Modify: `scripts/health.sh`, `scripts/smoke.sh`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Tests** — extend `serve_json` to record headers and body and answer GET:

```python
def serve_json(payload: dict, seen: dict | None = None):
    class H(BaseHTTPRequestHandler):
        def _reply(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            if seen is not None:
                seen["auth"] = self.headers.get("Authorization")
                seen["body"] = json.loads(raw)
            self._reply(json.dumps(payload).encode())

        def do_GET(self):
            if seen is not None:
                seen.setdefault("gets", []).append((self.path, self.headers.get("Authorization")))
            self._reply(b'{"data":[{"id":"local"}]}' if self.path == "/v1/models" else b"ok")
        ...

HEALTH = ROOT / "scripts" / "health.sh"


def test_smoke_sends_bearer_and_uses_local_model():
    seen = {}
    srv, base = serve_json(completion("thinking...", "4"), seen)
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base, "VLLM_API_KEY": "sekrit"})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["model"] == "local"


def test_smoke_without_key_sends_no_auth_header():
    seen = {}
    srv, base = serve_json(completion("thinking...", "4"), seen)
    try:
        run_bash(SMOKE, env={"BASE_URL": base, "VLLM_API_KEY": ""})
    finally:
        _stop(srv)
    assert seen["auth"] is None


def test_health_sends_bearer_only_to_v1():
    seen = {}
    srv, base = serve_json({}, seen)
    try:
        r = run_bash(HEALTH, env={"BASE_URL": base, "VLLM_API_KEY": "sekrit"})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("/health", None) in seen["gets"]
    assert ("/v1/models", "Bearer sekrit") in seen["gets"]
```

- [ ] **Step 2: Run** → the three new tests fail.
- [ ] **Step 3: Implement**

`scripts/health.sh`:
```bash
auth=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then auth=(-H "Authorization: Bearer $VLLM_API_KEY"); fi
"$CURL_BIN" -fsS -m 5 "$BASE/health" >/dev/null && echo "health: ok"
"$CURL_BIN" -fsS -m 5 ${auth[@]+"${auth[@]}"} "$BASE/v1/models"
```
`scripts/smoke.sh`: `MODEL="${MODEL:-local}"`, same `auth` array, and the curl line becomes
`curl -fsS -m 600 ${auth[@]+"${auth[@]}"} "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$payload"`.

- [ ] **Step 4: Run** `python -m pytest tests/test_smoke.py -q` → pass.
- [ ] **Step 5: Commit** — `git commit -am "feat: health/smoke send bearer token; smoke targets local"`

---

### Task 5: desktop — API key, `--model`, picker, onstart export, disk, banner, status

**Files:**
- Modify: `desktop/gpu_llm.py`
- Test: `tests/test_gpu_llm.py`

**Interfaces:**
- Consumes: `lib.profile.list_models`, `default_model` (Task 1); `rent_offer(api_key, offer_id, image, disk, onstart)` unchanged.
- Produces: `resolve_llm_api_key(cfg) -> str`; `llm_api_key_path() -> Path`; `choose_model(models, requested, yes) -> dict | None`; `format_model(m) -> str`; `onstart_text(model_id, api_key) -> str`; `print_client_settings(local_port)`; `state_context(st) -> int | None`; `http_get(url, timeout=3.0, headers=None)`; `up --model ID`; state key `model`.

- [ ] **Step 1: Tests** (add; also change line 226's `"LLM_MODEL=qwen"` to `"LLM_MODEL=local"`, and the `/v1/models` fake body at line 105 / assertion at 311 to `local`)

```python
from lib.profile import list_models


def test_resolve_llm_api_key_order_env_config_file(home, monkeypatch):
    (home / "llm_api_key").write_text("from-file\n")
    assert gpu_llm.resolve_llm_api_key({}) == "from-file"
    assert gpu_llm.resolve_llm_api_key({"llm_api_key": "from-cfg"}) == "from-cfg"
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    assert gpu_llm.resolve_llm_api_key({"llm_api_key": "from-cfg"}) == "from-env"


def test_resolve_llm_api_key_generates_and_persists(home, capsys):
    k1 = gpu_llm.resolve_llm_api_key({})
    k2 = gpu_llm.resolve_llm_api_key({})
    assert k1 == k2 and len(k1) >= 24
    assert (home / "llm_api_key").read_text().strip() == k1
    assert "Generated LLM API key" in capsys.readouterr().out


def test_resolve_llm_api_key_rejects_unsafe():
    with pytest.raises(SystemExit):
        gpu_llm.resolve_llm_api_key({"llm_api_key": "bad key'; rm -rf /"})


def test_onstart_text_prepends_exports_and_keeps_script():
    text = gpu_llm.onstart_text("gpt-oss-120b", "abc-123")
    assert text.startswith("export MODEL='gpt-oss-120b' VLLM_API_KEY='abc-123'; cd /root && git clone")
    with pytest.raises(ValueError):
        gpu_llm.onstart_text("x y", "k")


def test_up_yes_uses_default_model_disk_and_exports(up_env, capsys):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    rc = gpu_llm.main(["up", "--yes"])
    assert rc == 0
    _, _, disk, onstart = up_env["rented"][0]
    assert disk == 60
    key = gpu_llm.resolve_llm_api_key({})
    assert onstart.startswith(f"export MODEL='qwen3.8-27b-nvfp4' VLLM_API_KEY='{key}'; ")
    assert gpu_llm.load_state()["model"] == "qwen3.8-27b-nvfp4"


def test_up_model_flag_sets_disk_from_profile(up_env, capsys):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    rc = gpu_llm.main(["up", "--gpu", "a100-80", "--model", "gpt-oss-120b", "--yes"])
    assert rc == 0
    assert up_env["rented"][0][2] == 100
    assert "MODEL='gpt-oss-120b'" in up_env["rented"][0][3]


def test_up_unknown_model_lists_and_exits_1_before_search(up_env, capsys):
    rc = gpu_llm.main(["up", "--gpu", "a100-80", "--model", "nope", "--yes"])
    out = capsys.readouterr().out
    assert rc == 1
    assert up_env["rented"] == [] and "queries" not in up_env
    assert "qwen3.8-27b-bf16" in out and "gpt-oss-120b" in out


def test_up_model_picker_enter_takes_default(up_env, monkeypatch, capsys):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    answers = iter(["1", ""])                      # offer #1, then Enter for the model
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    rc = gpu_llm.main(["up", "--gpu", "a100-80"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(default)" in out and "MXFP4" in out and "63 GB" in out
    assert gpu_llm.load_state()["model"] == "qwen3.8-27b-bf16"


def test_up_model_picker_number_picks(up_env, monkeypatch):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    answers = iter(["1", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    rc = gpu_llm.main(["up", "--gpu", "a100-80"])
    assert rc == 0
    assert gpu_llm.load_state()["model"] == list_models(gpu_llm.REPO_ROOT / "profiles", "a100-80")[1]["id"]


def test_up_model_picker_bad_choice_rents_nothing(up_env, monkeypatch, capsys):
    up_env["offers"] = [OFFER]
    answers = iter(["1", "9"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    rc = gpu_llm.main(["up", "--gpu", "a100-80"])
    assert rc == 0 and up_env["rented"] == []
    assert "Nothing rented" in capsys.readouterr().out


def test_up_single_model_gpu_skips_picker(up_env, monkeypatch):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    answers = iter(["1"])                          # only the offer prompt
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert gpu_llm.main(["up"]) == 0


def test_print_client_settings_shows_key_and_context(home, capsys):
    (home / "llm_api_key").write_text("k-1\n")
    gpu_llm.save_state({"pid": 0, "gpu": "5090", "model": "qwen3.8-27b-nvfp4"})
    gpu_llm.print_client_settings(8000)
    out = capsys.readouterr().out
    assert "LLM_BASE_URL=http://127.0.0.1:8000" in out
    assert "LLM_MODEL=local" in out
    assert "LLM_API_KEY=k-1" in out
    assert "LLM_CONTEXT=32768" in out


def test_status_sends_bearer_and_reports_401(home, monkeypatch, capsys):
    (home / "llm_api_key").write_text("k-1\n")
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "gpu-line"))
    seen = {}
    def fake_get(url, timeout=3.0, headers=None):
        seen["headers"] = headers
        return 401, ""
    monkeypatch.setattr(gpu_llm, "http_get", fake_get)
    gpu_llm.save_state({"pid": os.getpid(), "host": "h", "ssh_port": 22, "local_port": 8000,
                       "image": "python", "gpu": "5090", "model": "qwen3.8-27b-nvfp4"})
    gpu_llm.main(["status"])
    out = capsys.readouterr().out
    assert seen["headers"] == {"Authorization": "Bearer k-1"}
    assert "models: unauthorized (check LLM_API_KEY)" in out
    assert "model: qwen3.8-27b-nvfp4 (context 32768)" in out
```

- [ ] **Step 2: Run** `python -m pytest tests/test_gpu_llm.py -q` → new tests fail.

- [ ] **Step 3: Implement in `desktop/gpu_llm.py`**

Imports: add `import re`, `import secrets`. After `REPO_ROOT = ...`:
```python
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from lib.profile import list_models  # noqa: E402

PROFILES_DIR = REPO_ROOT / "profiles"
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")
```

Key resolution:
```python
def llm_api_key_path() -> Path:
    return home() / "llm_api_key"


def resolve_llm_api_key(cfg: dict) -> str:
    """LLM_API_KEY env > config llm_api_key > ~/.gpu-llm/llm_api_key > generate+save."""
    key = os.environ.get("LLM_API_KEY") or cfg.get("llm_api_key")
    f = llm_api_key_path()
    if not key and f.exists():
        key = f.read_text(encoding="utf-8").strip()
    if not key:
        key = secrets.token_urlsafe(24)
        try:
            f.write_text(key + "\n", encoding="utf-8")
            if sys.platform != "win32":
                f.chmod(0o600)
            print(f"Generated LLM API key -> {f}")
        except OSError as e:
            print(f"WARNING: could not save LLM API key to {f} ({e}); using it for this run only.")
    if not SAFE_TOKEN.match(key):
        raise SystemExit("llm_api_key must match [A-Za-z0-9._-]+")
    return key
```

Model picker + onstart:
```python
def format_model(m: dict) -> str:
    parts = [m["id"], str(m.get("quant") or "?")]
    if m.get("size_gb") is not None:
        parts.append(f"{m['size_gb']} GB")
    if m.get("max_model_len"):
        parts.append(f"{m['max_model_len']} ctx")
    return "  ".join(parts) + ("  (default)" if m["default"] else "")


def print_models(models: list[dict]) -> None:
    for i, m in enumerate(models, 1):
        print(f"{i}. {format_model(m)}")


def choose_model(models: list[dict], requested: str | None, yes: bool) -> dict | None:
    """Pick a catalog entry: --model wins, then --yes/single-entry -> default,
    else a numbered prompt (Enter = default). None means abort."""
    default = next((m for m in models if m["default"]), models[0])
    if requested:
        return next((m for m in models if m["id"] == requested), None)
    if yes or len(models) == 1:
        return default
    print_models(models)
    try:
        raw = input(f"model? [1-{len(models)}, Enter = {default['id']}] ").strip()
    except EOFError:
        return None
    if raw == "":
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(models):
        return models[int(raw) - 1]
    return None


def onstart_text(model_id: str, api_key: str) -> str:
    for v in (model_id, api_key):
        if not SAFE_TOKEN.match(v):
            raise ValueError(f"unsafe value for onstart: {v!r}")
    base = (REPO_ROOT / "vast" / "onstart.sh").read_text(encoding="utf-8")
    return f"export MODEL='{model_id}' VLLM_API_KEY='{api_key}'; {base}"
```

In `_cmd_up_body`, right after the stale-state block and before `filters = ...`:
```python
    models = list_models(PROFILES_DIR, args.gpu)
    if args.model and args.model not in {m["id"] for m in models}:
        print(f"Unknown model '{args.model}' for {args.gpu}. Available:")
        print_models(models)
        return 1
```
Right before `dph = offer.get(...)`:
```python
    model = choose_model(models, args.model, args.yes)
    if model is None:
        print("Nothing rented.")
        return 0
    llm_key = resolve_llm_api_key(cfg)
    disk = int(model.get("disk_gb") or DISK_GB)
```
Replace the `onstart = ...`/`rent_offer`/`save_state` lines with:
```python
    onstart = onstart_text(model["id"], llm_key)
    iid = vast_api.rent_offer(api_key, offer["id"], IMAGE, disk, onstart)
    save_state({"pid": 0, "instance_id": iid, "gpu": args.gpu, "model": model["id"], "dph": dph})
    print(f"Rented instance {iid} ({model['id']}, {disk} GB disk) at ${dph:.3f}/hr. Waiting for SSH info...")
```
Parser: `u.add_argument("--model", default=None, help="model id under profiles/<gpu>/ (default: that GPU's default)")`.

Banner + context + status:
```python
def state_context(st: dict) -> int | None:
    if not (st.get("gpu") and st.get("model")):
        return None
    try:
        return next(m for m in list_models(PROFILES_DIR, st["gpu"]) if m["id"] == st["model"]).get("max_model_len")
    except (ValueError, StopIteration):
        return None


def print_client_settings(local_port: int) -> None:
    st = load_state() or {}
    print(f"LLM_BASE_URL=http://127.0.0.1:{local_port}")
    print("LLM_MODEL=local")
    print(f"LLM_API_KEY={resolve_llm_api_key(load_config())}")
    ctx = state_context(st)
    if ctx:
        print(f"LLM_CONTEXT={ctx}")
```
In `open_tunnel_and_wait` replace the two `print(f"LLM_BASE_URL...")`/`print("LLM_MODEL=qwen")` lines with `print_client_settings(local_port)`.

`http_get` gains headers:
```python
def http_get(url: str, timeout: float = 3.0, headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
```
`cmd_status`: replace the `/v1/models` call and its branches with
```python
    key = resolve_llm_api_key(load_config())
    code, body = http_get(f"http://127.0.0.1:{st['local_port']}/v1/models", timeout=3.0,
                          headers={"Authorization": f"Bearer {key}"})
    if code == 200:
        ...unchanged parse/print...
    elif code == 401:
        print("models: unauthorized (check LLM_API_KEY)")
    else:
        print("models: unreachable")
    if st.get("model"):
        ctx = state_context(st)
        print(f"model: {st['model']}" + (f" (context {ctx})" if ctx else ""))
```

- [ ] **Step 4: Run** `python -m pytest tests/test_gpu_llm.py -q` → pass. Then `python -m pytest -q` → all green.
- [ ] **Step 5: Commit** — `git commit -am "feat: gpu-llm up --model/picker, LLM api key, onstart exports, per-model disk, client banner"`

---

### Task 6: README (config, daily loop, Rider, M3 acceptance) + memory note

**Files:**
- Modify: `README.md`
- Test: `python -m pytest -q` (no new tests; docs)

- [ ] **Step 1: README edits**
  - Intro: "serves a model you pick from `profiles/` (default Qwen3.8-27B) as `local`".
  - llm-cli config block: `model = "local"`, `coder_model = "local"`, and a line "API key: the `LLM_API_KEY` value printed by `up` (also in `%USERPROFILE%\.gpu-llm\llm_api_key`)". `~/.gpu-llm/config.toml` block gains `# llm_api_key = "..."   # optional; generated into llm_api_key otherwise`.
  - Daily loop step 1: add "then a model picker when the GPU has more than one (Enter = default), or `--model <id>`; `profiles/README.md` lists the matrix". Mention `LLM_API_KEY`/`LLM_CONTEXT` in the READY banner. Note "push before `up`: the instance clones `main`, so your local `profiles/` must match".
  - Manual-rental fallback: set `-e MODEL=` and `-e VLLM_API_KEY=<contents of llm_api_key>`.
  - Scripts table: `serve.sh <gpu> <model>`; exit code 6 `hf buckets sync` failed; `profiles/README.md` row.
  - New section **Using the model from JetBrains Rider** with the seven bullets from spec §7 verbatim (settings path, URL with `/v1` first, key, tool calling on, Models Assignment Core=`local`, context window = `LLM_CONTEXT`, skip AI Completion, Junie CLI JSON example).
  - New section **Milestone 3 acceptance run** = spec §10 as a checklist with an empty results table (Time to READY, tok/s, peak VRAM, bucket sync time, Rider URL form that worked, agent-mode result).
  - Deferred: remove "3090/4090 profiles"? No — keep; add "Qwen3.8-Flash-Next (see profiles/README.md)", "MTP speculative decoding".
- [ ] **Step 2: Run** `python -m pytest -q` → green.
- [ ] **Step 3: Commit** — `git commit -am "docs: M3 README — model picker, local alias, LLM_API_KEY, Rider setup, acceptance checklist"`
- [ ] **Step 4: Push** `git push` (the instance clones `main`; the acceptance run needs the new profiles there).

---

## Self-review

- Spec coverage: §3 catalog shape/default → T1; served names → T1; channel → T5; API key → T5 (desktop), T2 (serve), T4 (clients); bucket → T1/T2; disk → T1/T5; instance side §5 → T2/T3/T4; desktop §6 → T5; Rider §7 → T6; errors §8 → T2/T3/T5; tests §9 → each task; acceptance §10 → T6 checklist.
- Names used consistently: `list_models`, `default_model`, `choose_model`, `format_model`, `print_models`, `onstart_text`, `resolve_llm_api_key`, `llm_api_key_path`, `print_client_settings`, `state_context`, `PROFILES_DIR`, `SAFE_TOKEN`, `HF_BIN`, `VLLM_API_KEY`, `MODEL`.
