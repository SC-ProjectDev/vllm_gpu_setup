# vLLM GPU Setup — Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a rented Vast.ai RTX 5090 into an OpenAI-compatible vLLM server for Qwen3.8-27B, reachable from the desktop at `http://127.0.0.1:8000` over an SSH tunnel, with the existing `llm-cli` needing only a config change.

**Architecture:** The Vast template is the official `vllm/vllm-openai:v0.27.1` image; this repo is cloned by its onstart command and `bootstrap.sh` detects the GPU, converts a YAML profile into `vllm serve` argv, and launches on loopback. A stdlib-only desktop script `desktop/gpu_llm.py` opens the SSH tunnel, polls health while showing instance logs, and prints the env vars for `llm-cli`.

**Tech Stack:** Bash (instance side, runs in the vLLM image), Python 3.11+ (`lib/profile.py` on instance — image ships Python 3.12; `desktop/gpu_llm.py` on Windows), pytest for local tests. No third-party Python dependencies anywhere.

**Spec:** `docs/superpowers/specs/2026-08-22-vllm-gpu-setup-design.md`

## Global Constraints

- Image pinned: `vllm/vllm-openai:v0.27.1`. Minimum vLLM version enforced by bootstrap: `0.17`.
- Served model name is always `qwen`; server binds `--host 127.0.0.1 --port 8000` only.
- Instance status file: `/var/log/vllm.status` with values `STARTING`, `READY`, or `FAILED: <reason>`. Instance log: `/var/log/vllm.log`.
- Bootstrap exit codes: 2 unknown GPU, 3 missing `HF_TOKEN`, 4 vLLM too old.
- No PyYAML, no `requests`, no `openai` SDK. Python stdlib only. Bash scripts use `set -euo pipefail`.
- Every Bash script must be LF line endings (set `.gitattributes`: `*.sh text eol=lf`) — they run on Linux.
- Tests run on Windows via Git Bash (`bash` on PATH) and `python -m pytest`. Tests must not require a GPU, network, or real `ssh`.
- Profiles: `5090`, `a100-80`, `h100-80`, `h200` with the exact fields and values in spec §4.3.
- Desktop state: `~/.gpu-llm/state.json`; desktop config: `~/.gpu-llm/config.toml`. Both overridable via `GPU_LLM_HOME` env (for tests).
- Commit after every task. Commit messages: conventional (`feat:`, `test:`, `docs:`).

---

## File Structure

| Path | Responsibility |
|---|---|
| `.gitattributes` | Force LF on `*.sh` |
| `.env.example` | `HF_TOKEN`, `PROFILE`, `MAX_MODEL_LEN` documented |
| `pyproject.toml` | pytest config only (`testpaths = ["tests"]`) |
| `profiles/*.yaml` | One flat YAML per GPU class |
| `lib/profile.py` | `load_profile(path) -> dict`, `profile_to_argv(profile, overrides) -> list[str]`; CLI prints argv one-per-line |
| `lib/detect_gpu.sh` | `nvidia-smi` → `"<profile> <count>"` on stdout, exit 2 if unknown |
| `lib/status.sh` | `write_status "<text>"` helper sourced by bootstrap/serve |
| `scripts/serve.sh` | Launch vLLM with profile argv; manage status file |
| `scripts/health.sh` | curl `/health` and `/v1/models` |
| `scripts/smoke.sh` | One thinking-mode chat completion; asserts; prints tok/s |
| `bootstrap.sh` | onstart entrypoint: env → version check → detect → token check → serve |
| `desktop/gpu_llm.py` | `tunnel`, `status`, `logs`, `down` |
| `vast/template.md` | Exact rental settings |
| `README.md` | Setup, `llm-cli` config, acceptance checklist |
| `tests/test_profile.py` | profile parsing + argv |
| `tests/test_detect_gpu.py` | bash script with fake `nvidia-smi` |
| `tests/test_bootstrap.py` | bootstrap failure paths with fakes |
| `tests/test_gpu_llm.py` | desktop CLI with fake HTTP server + fake ssh |

Paths inside the instance: repo is cloned to `/root/vllm_gpu_setup`. Scripts locate the repo root via `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` so tests can run them from any checkout. Status/log paths are taken from env `VLLM_STATUS_FILE` (default `/var/log/vllm.status`) and `VLLM_LOG_FILE` (default `/var/log/vllm.log`) so tests can redirect them.

---

### Task 1: Scaffold, profiles, and `lib/profile.py`

**Files:**
- Create: `.gitattributes`, `.env.example`, `pyproject.toml`
- Create: `profiles/5090.yaml`, `profiles/a100-80.yaml`, `profiles/h100-80.yaml`, `profiles/h200.yaml`
- Create: `lib/__init__.py` (empty), `lib/profile.py`
- Test: `tests/__init__.py` (empty), `tests/test_profile.py`

**Interfaces:**
- Produces: `lib.profile.load_profile(path: str | Path) -> dict` — flat YAML reader supporting `key: value` scalars (str/int/float/bool) and a list under a key given as `- item` lines. `lib.profile.profile_to_argv(profile: dict, overrides: dict[str, str]) -> list[str]` — argv **after** `vllm serve`. `python lib/profile.py <profile.yaml>` prints argv one per line (honours env `MAX_MODEL_LEN`).

- [ ] **Step 1: Scaffold files**

`.gitattributes`:
```
*.sh text eol=lf
```

`.env.example`:
```
# Hugging Face token. Only needed for gated repos; M1 profiles are all public.
HF_TOKEN=
# Force a profile instead of auto-detecting from nvidia-smi: 5090 | a100-80 | h100-80 | h200
PROFILE=
# Override the profile's max_model_len (e.g. lower it after an OOM).
MAX_MODEL_LEN=
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

`profiles/5090.yaml`:
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

`profiles/a100-80.yaml`:
```yaml
model: Qwen/Qwen3.8-27B
served_model_name: qwen
max_model_len: 131072
gpu_memory_utilization: 0.92
kv_cache_dtype: fp8
enforce_eager: false
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --reasoning-parser qwen3
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
```

`profiles/h100-80.yaml`:
```yaml
model: Qwen/Qwen3.8-27B-FP8
served_model_name: qwen
max_model_len: 262144
gpu_memory_utilization: 0.92
kv_cache_dtype: fp8
enforce_eager: false
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --reasoning-parser qwen3
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
```

`profiles/h200.yaml`:
```yaml
model: Qwen/Qwen3.8-27B
served_model_name: qwen
max_model_len: 262144
gpu_memory_utilization: 0.92
kv_cache_dtype: fp8
enforce_eager: false
tensor_parallel_size: 1
requires_hf_token: false
extra_args:
  - --reasoning-parser qwen3
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
```

Create empty `lib/__init__.py` and `tests/__init__.py`.

- [ ] **Step 2: Write the failing tests**

`tests/test_profile.py`:
```python
from pathlib import Path

import pytest

from lib.profile import load_profile, profile_to_argv

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"


def test_load_profile_parses_scalars_and_list(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(
        "model: a/b\nmax_model_len: 4096\ngpu_memory_utilization: 0.9\n"
        "enforce_eager: true\nrequires_hf_token: false\n"
        "extra_args:\n  - --foo bar\n  - --baz\n"
    )
    d = load_profile(p)
    assert d["model"] == "a/b"
    assert d["max_model_len"] == 4096
    assert d["gpu_memory_utilization"] == 0.9
    assert d["enforce_eager"] is True
    assert d["requires_hf_token"] is False
    assert d["extra_args"] == ["--foo bar", "--baz"]


def test_load_profile_ignores_comments_and_blank_lines(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("# c\n\nmodel: m  # trailing\n")
    assert load_profile(p) == {"model": "m"}


def test_argv_5090():
    argv = profile_to_argv(load_profile(PROFILES / "5090.yaml"), {})
    assert argv == [
        "Inferact/Qwen3.8-27B-NVFP4",
        "--served-model-name", "qwen",
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
    argv = profile_to_argv(load_profile(PROFILES / "h100-80.yaml"), {})
    assert "--enforce-eager" not in argv
    assert argv[0] == "Qwen/Qwen3.8-27B-FP8"
    assert argv[argv.index("--max-model-len") + 1] == "262144"


def test_max_model_len_override():
    argv = profile_to_argv(load_profile(PROFILES / "5090.yaml"), {"max_model_len": "16384"})
    assert argv[argv.index("--max-model-len") + 1] == "16384"


def test_empty_override_is_ignored():
    argv = profile_to_argv(load_profile(PROFILES / "5090.yaml"), {"max_model_len": ""})
    assert argv[argv.index("--max-model-len") + 1] == "32768"


@pytest.mark.parametrize("name", ["5090", "a100-80", "h100-80", "h200"])
def test_all_shipped_profiles_load_and_serve_qwen(name):
    d = load_profile(PROFILES / f"{name}.yaml")
    argv = profile_to_argv(d, {})
    assert d["served_model_name"] == "qwen"
    assert "--reasoning-parser" in argv and "qwen3" in argv
    assert argv[argv.index("--kv-cache-dtype") + 1] == "fp8"


def test_cli_prints_argv_one_per_line(tmp_path):
    import os, subprocess, sys
    env = dict(os.environ, MAX_MODEL_LEN="8192")
    out = subprocess.run(
        [sys.executable, str(ROOT / "lib" / "profile.py"), str(PROFILES / "5090.yaml")],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.splitlines()
    assert out[0] == "Inferact/Qwen3.8-27B-NVFP4"
    assert out[out.index("--max-model-len") + 1] == "8192"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.profile'`

- [ ] **Step 4: Implement `lib/profile.py`**

```python
"""Flat-YAML profile loader and vllm serve argv builder. Stdlib only."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _scalar(raw: str):
    v = raw.split("#", 1)[0].strip() if not raw.strip().startswith(("'", '"')) else raw.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.startswith(("'", '"')) and v.endswith(v[0]) and len(v) >= 2:
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_profile(path: str | Path) -> dict:
    """Parse a flat YAML file: `key: scalar` lines and `key:` followed by `- item` lines."""
    data: dict = {}
    current_list: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list is not None:
            data[current_list].append(stripped[2:].split("  #", 1)[0].strip())
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}: cannot parse line: {line!r}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        if rest.strip() == "":
            data[key] = []
            current_list = key
        else:
            data[key] = _scalar(rest)
            current_list = None
    return data


def profile_to_argv(profile: dict, overrides: dict[str, str]) -> list[str]:
    """Return the argv that follows `vllm serve` for this profile.

    `overrides` maps profile keys to string values; empty strings are ignored.
    Only `max_model_len` is honoured in M1.
    """
    max_len = profile["max_model_len"]
    if overrides.get("max_model_len"):
        max_len = int(overrides["max_model_len"])
    argv = [
        str(profile["model"]),
        "--served-model-name", str(profile["served_model_name"]),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--kv-cache-dtype", str(profile["kv_cache_dtype"]),
        "--tensor-parallel-size", str(profile["tensor_parallel_size"]),
    ]
    if profile.get("enforce_eager") is True:
        argv.append("--enforce-eager")
    for extra in profile.get("extra_args", []):
        argv.extend(extra.split())
    return argv


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: profile.py <profile.yaml>", file=sys.stderr)
        return 1
    profile = load_profile(argv[1])
    for item in profile_to_argv(profile, {"max_model_len": os.environ.get("MAX_MODEL_LEN", "")}):
        print(item)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_profile.py -v`
Expected: 9 passed (note `gpu_memory_utilization` 0.90 parses to float `0.9`, which `str()` renders as `0.9` — matches the test).

- [ ] **Step 6: Commit**

```bash
git add .gitattributes .env.example pyproject.toml profiles lib tests
git commit -m "feat: profiles and flat-YAML profile_to_argv builder"
```

---

### Task 2: `lib/detect_gpu.sh`

**Files:**
- Create: `lib/detect_gpu.sh`
- Test: `tests/test_detect_gpu.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `lib/detect_gpu.sh` prints `"<profile> <gpu_count>"` to stdout (e.g. `5090 1`) and exits 0; on unknown GPU prints the raw `nvidia-smi` line and the profile table to stderr and exits 2. Honours `NVIDIA_SMI` env (default `nvidia-smi`) so tests can inject a fake.

- [ ] **Step 1: Write `tests/conftest.py` helper for fake commands**

```python
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_fake(dir_: Path, name: str, body: str) -> Path:
    """Create an executable bash script `dir_/name` with `body` (no shebang needed)."""
    p = dir_ / name
    p.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def run_bash(script: Path, *args: str, env: dict | None = None, cwd: Path | None = None):
    """Run a repo bash script through `bash`, returning CompletedProcess with text output."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True, text=True, env=full_env, cwd=cwd or ROOT,
    )


@pytest.fixture
def fakes(tmp_path):
    d = tmp_path / "fakes"
    d.mkdir()
    return d
```

- [ ] **Step 2: Write the failing tests**

`tests/test_detect_gpu.py`:
```python
import pytest

from tests.conftest import ROOT, run_bash, write_fake

SCRIPT = ROOT / "lib" / "detect_gpu.sh"


def detect(fakes, smi_output: str):
    fake = write_fake(fakes, "fake_smi", f"printf '%s\\n' {smi_output!r}\n")
    return run_bash(SCRIPT, env={"NVIDIA_SMI": str(fake)})


@pytest.mark.parametrize("line,expected", [
    ("NVIDIA GeForce RTX 5090, 32607 MiB", "5090 1"),
    ("NVIDIA A100-SXM4-80GB, 81920 MiB", "a100-80 1"),
    ("NVIDIA A100 80GB PCIe, 81920 MiB", "a100-80 1"),
    ("NVIDIA H100 80GB HBM3, 81559 MiB", "h100-80 1"),
    ("NVIDIA H200, 143771 MiB", "h200 1"),
])
def test_known_names(fakes, line, expected):
    r = detect(fakes, line)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


def test_multi_gpu_count(fakes):
    r = detect(fakes, "NVIDIA H100 80GB HBM3, 81559 MiB\nNVIDIA H100 80GB HBM3, 81559 MiB")
    assert r.returncode == 0
    assert r.stdout.strip() == "h100-80 2"


def test_vram_fallback_for_unknown_name_with_80gb(fakes):
    r = detect(fakes, "NVIDIA Mystery Card, 81920 MiB")
    assert r.returncode == 0
    assert r.stdout.strip() == "h100-80 1"


def test_unknown_small_gpu_exits_2_with_table(fakes):
    r = detect(fakes, "NVIDIA GeForce RTX 4090, 24564 MiB")
    assert r.returncode == 2
    assert "RTX 4090" in r.stderr
    assert "5090" in r.stderr and "h200" in r.stderr


def test_a100_40gb_is_not_matched(fakes):
    r = detect(fakes, "NVIDIA A100-PCIE-40GB, 40960 MiB")
    assert r.returncode == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_detect_gpu.py -v`
Expected: FAIL — bash reports `lib/detect_gpu.sh: No such file or directory`

- [ ] **Step 4: Implement `lib/detect_gpu.sh`**

```bash
#!/usr/bin/env bash
# Map nvidia-smi output to a profile id. Prints "<profile> <count>". Exit 2 if unknown.
set -euo pipefail

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
PROFILE_TABLE='Known GPUs -> profiles:
  RTX 5090                      -> 5090
  A100-SXM4-80GB / A100 80GB    -> a100-80
  H100 (80GB)                   -> h100-80
  H200                          -> h200
Fallback by VRAM: >=130GB h200, >=75GB h100-80, >=30GB 5090.
Override with PROFILE=<id> in .env.'

mapfile -t lines < <("$NVIDIA_SMI" --query-gpu=name,memory.total --format=csv,noheader)
if [[ ${#lines[@]} -eq 0 ]]; then
  echo "detect_gpu: nvidia-smi returned no GPUs" >&2
  exit 2
fi

first="${lines[0]}"
name="${first%%,*}"
mem_raw="${first#*,}"
mem_mib="$(echo "$mem_raw" | tr -dc '0-9')"
mem_gb=$(( mem_mib / 1024 ))
count="${#lines[@]}"

profile=""
case "$name" in
  *"RTX 5090"*)                       profile="5090" ;;
  *"A100-SXM4-80GB"*|*"A100 80GB"*)   profile="a100-80" ;;
  *"H200"*)                            profile="h200" ;;
  *"H100"*)                            profile="h100-80" ;;
esac

if [[ -z "$profile" ]]; then
  if   (( mem_gb >= 130 )); then profile="h200"
  elif (( mem_gb >= 75 ));  then profile="h100-80"
  elif (( mem_gb >= 30 ));  then profile="5090"
  fi
fi

if [[ -z "$profile" ]]; then
  echo "detect_gpu: unknown GPU: $first" >&2
  echo "$PROFILE_TABLE" >&2
  exit 2
fi

echo "$profile $count"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_detect_gpu.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add lib/detect_gpu.sh tests/conftest.py tests/test_detect_gpu.py
git commit -m "feat: GPU detection to profile id"
```

---

### Task 3: `lib/status.sh`, `scripts/serve.sh`, `scripts/health.sh`

**Files:**
- Create: `lib/status.sh`, `scripts/serve.sh`, `scripts/health.sh`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `lib/profile.py` CLI (Task 1).
- Produces: `lib/status.sh` defines `write_status "<text>"` (writes to `$VLLM_STATUS_FILE`). `scripts/serve.sh <profile-id>` runs `$VLLM_BIN serve <argv> --host 127.0.0.1 --port 8000`, logs to `$VLLM_LOG_FILE`, writes `STARTING` → `READY` / `FAILED: vllm exited N`. Env knobs: `VLLM_BIN` (default `vllm`), `HEALTH_URL` (default `http://127.0.0.1:8000/health`), `HEALTH_POLL_SECS` (default 5), `CURL_BIN` (default `curl`). `scripts/health.sh` curls `/health` and `/v1/models`, exit non-zero if either fails.

- [ ] **Step 1: Write the failing tests**

`tests/test_serve.py`:
```python
from pathlib import Path

from tests.conftest import ROOT, run_bash, write_fake

SERVE = ROOT / "scripts" / "serve.sh"


def env_for(tmp_path: Path, fakes: Path, **extra):
    status = tmp_path / "vllm.status"
    log = tmp_path / "vllm.log"
    e = {
        "VLLM_STATUS_FILE": str(status),
        "VLLM_LOG_FILE": str(log),
        "HEALTH_POLL_SECS": "0",
        "VLLM_BIN": str(fakes / "vllm"),
        "CURL_BIN": str(fakes / "curl"),
    }
    e.update(extra)
    return e, status, log


def test_serve_success_writes_ready_and_passes_argv(tmp_path, fakes):
    # fake vllm: record argv, stay alive briefly
    write_fake(fakes, "vllm", 'echo "ARGS: $*"; sleep 1\n')
    write_fake(fakes, "curl", "exit 0\n")  # health always OK
    env, status, log = env_for(tmp_path, fakes)
    r = run_bash(SERVE, "5090", env=env)
    assert r.returncode == 0, r.stderr
    assert status.read_text().strip() == "READY"
    args = log.read_text()
    assert "serve Inferact/Qwen3.8-27B-NVFP4" in args
    assert "--served-model-name qwen" in args
    assert "--host 127.0.0.1 --port 8000" in args
    assert "--enforce-eager" in args


def test_serve_failure_writes_failed_with_oom_hint(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "torch.OutOfMemoryError: CUDA out of memory" >&2; exit 1\n')
    write_fake(fakes, "curl", "exit 7\n")  # never healthy
    env, status, log = env_for(tmp_path, fakes)
    r = run_bash(SERVE, "5090", env=env)
    assert r.returncode == 1
    assert status.read_text().strip() == "FAILED: vllm exited 1"
    assert "MAX_MODEL_LEN" in r.stdout  # hint printed
    assert "out of memory" in r.stdout  # log tail printed


def test_serve_max_model_len_override_reaches_vllm(tmp_path, fakes):
    write_fake(fakes, "vllm", 'echo "ARGS: $*"; sleep 1\n')
    write_fake(fakes, "curl", "exit 0\n")
    env, status, log = env_for(tmp_path, fakes, MAX_MODEL_LEN="12345")
    r = run_bash(SERVE, "5090", env=env)
    assert r.returncode == 0, r.stderr
    assert "--max-model-len 12345" in log.read_text()


def test_serve_unknown_profile_fails(tmp_path, fakes):
    write_fake(fakes, "vllm", "exit 0\n")
    write_fake(fakes, "curl", "exit 0\n")
    env, status, log = env_for(tmp_path, fakes)
    r = run_bash(SERVE, "nope", env=env)
    assert r.returncode != 0
    assert status.read_text().startswith("FAILED: no profile")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_serve.py -v`
Expected: FAIL — `scripts/serve.sh: No such file or directory`

- [ ] **Step 3: Implement the three scripts**

`lib/status.sh`:
```bash
#!/usr/bin/env bash
# Source this. Provides write_status.
VLLM_STATUS_FILE="${VLLM_STATUS_FILE:-/var/log/vllm.status}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-/var/log/vllm.log}"

write_status() {
  mkdir -p "$(dirname "$VLLM_STATUS_FILE")"
  printf '%s\n' "$1" > "$VLLM_STATUS_FILE"
  echo "[status] $1"
}
```

`scripts/serve.sh`:
```bash
#!/usr/bin/env bash
# Usage: serve.sh <profile-id>. Launches vllm serve on 127.0.0.1:8000 and manages the status file.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/status.sh
source "$REPO_ROOT/lib/status.sh"

VLLM_BIN="${VLLM_BIN:-vllm}"
CURL_BIN="${CURL_BIN:-curl}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_POLL_SECS="${HEALTH_POLL_SECS:-5}"

profile_id="${1:-}"
profile_file="$REPO_ROOT/profiles/${profile_id}.yaml"
if [[ -z "$profile_id" || ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '${profile_id}' (expected $REPO_ROOT/profiles/<id>.yaml)"
  exit 2
fi

mapfile -t argv < <(python3 "$REPO_ROOT/lib/profile.py" "$profile_file")
mkdir -p "$(dirname "$VLLM_LOG_FILE")"
write_status "STARTING"
echo "[serve] profile=$profile_id" | tee -a "$VLLM_LOG_FILE"

"$VLLM_BIN" serve "${argv[@]}" --host 127.0.0.1 --port 8000 >> "$VLLM_LOG_FILE" 2>&1 &
vllm_pid=$!

# Wait for health or exit.
while kill -0 "$vllm_pid" 2>/dev/null; do
  if "$CURL_BIN" -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
    write_status "READY"
    wait "$vllm_pid"; code=$?
    if [[ $code -ne 0 ]]; then
      write_status "FAILED: vllm exited $code"
      tail -n 40 "$VLLM_LOG_FILE"
      exit "$code"
    fi
    exit 0
  fi
  sleep "$HEALTH_POLL_SECS"
done

wait "$vllm_pid" || code=$?
code="${code:-0}"
write_status "FAILED: vllm exited $code"
echo "[serve] last 40 log lines:"
tail -n 40 "$VLLM_LOG_FILE"
if grep -qi "out of memory" "$VLLM_LOG_FILE"; then
  echo "[serve] hint: OOM at startup — set MAX_MODEL_LEN to a lower value in .env and re-run."
fi
exit "$code"
```

`scripts/health.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
CURL_BIN="${CURL_BIN:-curl}"
"$CURL_BIN" -fsS -m 5 "$BASE/health" >/dev/null && echo "health: ok"
"$CURL_BIN" -fsS -m 5 "$BASE/v1/models"
echo
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_serve.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add lib/status.sh scripts/serve.sh scripts/health.sh tests/test_serve.py
git commit -m "feat: serve.sh with status file and health wait"
```

---

### Task 4: `bootstrap.sh`

**Files:**
- Create: `bootstrap.sh`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `lib/detect_gpu.sh` (Task 2), `lib/status.sh`, `scripts/serve.sh` (Task 3), `lib/profile.py` (Task 1).
- Produces: `bootstrap.sh` — the onstart entrypoint. Env knobs for tests: `VLLM_VERSION_CMD` (default `python3 -c 'import vllm; print(vllm.__version__)'`), `DETECT_GPU` (default `$REPO_ROOT/lib/detect_gpu.sh`), `SERVE` (default `$REPO_ROOT/scripts/serve.sh`). Exit codes: 2 unknown GPU, 3 HF_TOKEN required, 4 vLLM too old.

- [ ] **Step 1: Write the failing tests**

`tests/test_bootstrap.py`:
```python
from pathlib import Path

from tests.conftest import ROOT, run_bash, write_fake

BOOT = ROOT / "bootstrap.sh"


def setup(tmp_path: Path, fakes: Path, *, version="0.27.1", detect_out="5090 1", detect_rc=0):
    status = tmp_path / "vllm.status"
    write_fake(fakes, "version", f"echo {version}\n")
    write_fake(fakes, "detect", f"echo '{detect_out}'; exit {detect_rc}\n")
    write_fake(fakes, "serve", 'echo "SERVE $*"\n')
    env = {
        "VLLM_STATUS_FILE": str(status),
        "VLLM_LOG_FILE": str(tmp_path / "vllm.log"),
        "VLLM_VERSION_CMD": str(fakes / "version"),
        "DETECT_GPU": str(fakes / "detect"),
        "SERVE": str(fakes / "serve"),
        "PROFILE": "",
        "HF_TOKEN": "",
    }
    return env, status


def test_happy_path_detects_and_serves(tmp_path, fakes):
    env, status = setup(tmp_path, fakes)
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "SERVE 5090" in r.stdout


def test_old_vllm_exits_4(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, version="0.16.2")
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 4
    assert status.read_text().startswith("FAILED: vllm 0.16.2 < 0.17")


def test_unknown_gpu_exits_2(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 2
    assert status.read_text().startswith("FAILED: unknown gpu")


def test_profile_env_override_skips_detection(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    env["PROFILE"] = "h200"
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "SERVE h200" in r.stdout


def test_requires_hf_token_exits_3(tmp_path, fakes):
    # make a temporary profile that requires a token
    gated = ROOT / "profiles" / "zz-gated-test.yaml"
    gated.write_text((ROOT / "profiles" / "5090.yaml").read_text().replace(
        "requires_hf_token: false", "requires_hf_token: true"))
    try:
        env, status = setup(tmp_path, fakes)
        env["PROFILE"] = "zz-gated-test"
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 3
        assert status.read_text().startswith("FAILED: HF_TOKEN required")
    finally:
        gated.unlink()


def test_dotenv_is_loaded_from_repo_root(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    dotenv = ROOT / ".env"
    assert not dotenv.exists(), "refusing to clobber a real .env"
    dotenv.write_text("PROFILE=a100-80\n")
    try:
        del env["PROFILE"]
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "SERVE a100-80" in r.stdout
    finally:
        dotenv.unlink()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bootstrap.py -v`
Expected: FAIL — `bootstrap.sh: No such file or directory`

- [ ] **Step 3: Implement `bootstrap.sh`**

```bash
#!/usr/bin/env bash
# Vast onstart entrypoint: env -> vLLM version check -> GPU detect -> token check -> serve.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/status.sh
source "$REPO_ROOT/lib/status.sh"

MIN_VLLM="0.17"
VLLM_VERSION_CMD="${VLLM_VERSION_CMD:-python3 -c 'import vllm; print(vllm.__version__)'}"
DETECT_GPU="${DETECT_GPU:-$REPO_ROOT/lib/detect_gpu.sh}"
SERVE="${SERVE:-$REPO_ROOT/scripts/serve.sh}"

# 1. .env (only sets vars that are unset or empty in the environment)
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    if [[ -z "${!k:-}" ]]; then export "$k=$v"; fi
  done < "$REPO_ROOT/.env"
fi
export HF_TOKEN="${HF_TOKEN:-}" PROFILE="${PROFILE:-}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# 2. vLLM version
ver="$(bash -c "$VLLM_VERSION_CMD" 2>/dev/null || true)"
if [[ -z "$ver" ]]; then
  write_status "FAILED: vllm not importable"; exit 4
fi
if [[ "$(printf '%s\n%s\n' "$MIN_VLLM" "$ver" | sort -V | head -n1)" != "$MIN_VLLM" ]]; then
  write_status "FAILED: vllm $ver < $MIN_VLLM"; exit 4
fi
echo "[bootstrap] vllm $ver"

# 3. profile
if [[ -z "$PROFILE" ]]; then
  if ! out="$(bash "$DETECT_GPU")"; then
    write_status "FAILED: unknown gpu (see bootstrap log)"; exit 2
  fi
  PROFILE="${out%% *}"
  echo "[bootstrap] detected profile=$PROFILE gpus=${out#* }"
else
  echo "[bootstrap] PROFILE override=$PROFILE"
fi

# 4. token
profile_file="$REPO_ROOT/profiles/${PROFILE}.yaml"
if [[ ! -f "$profile_file" ]]; then
  write_status "FAILED: no profile '$PROFILE'"; exit 2
fi
if grep -qE '^requires_hf_token:\s*true' "$profile_file" && [[ -z "$HF_TOKEN" ]]; then
  write_status "FAILED: HF_TOKEN required by profile $PROFILE"; exit 3
fi
[[ -n "$HF_TOKEN" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# 5. serve
exec bash "$SERVE" "$PROFILE"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bootstrap.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: all pass (9 + 9 + 4 + 6 = 28)

- [ ] **Step 6: Commit**

```bash
git add bootstrap.sh tests/test_bootstrap.py
git commit -m "feat: bootstrap.sh onstart entrypoint"
```

---

### Task 5: `scripts/smoke.sh`

**Files:**
- Create: `scripts/smoke.sh`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Produces: `scripts/smoke.sh` — POSTs one thinking-mode chat completion to `$BASE_URL/v1/chat/completions` (default `http://127.0.0.1:8000`), asserts `reasoning_content` and `content` are non-empty, prints `tok/s: N`. Exit 1 on assertion failure. Uses `python3` for JSON.

- [ ] **Step 1: Write the failing tests**

`tests/test_smoke.py`:
```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.conftest import ROOT, run_bash

SMOKE = ROOT / "scripts" / "smoke.sh"


def serve_json(payload: dict):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def completion(reasoning, content, tokens=42):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": tokens},
    }


def test_smoke_passes_and_prints_toks():
    srv, base = serve_json(completion("thinking...", "4"))
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base})
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tok/s:" in r.stdout
    assert "reasoning: ok" in r.stdout


def test_smoke_fails_without_reasoning():
    srv, base = serve_json(completion("", "4"))
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base})
    finally:
        srv.shutdown()
    assert r.returncode == 1
    assert "reasoning_content empty" in r.stdout + r.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `scripts/smoke.sh: No such file or directory`

- [ ] **Step 3: Implement `scripts/smoke.sh`**

```bash
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

start=$(date +%s.%N)
resp="$(curl -fsS -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$payload")"
end=$(date +%s.%N)

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
print(f"tok/s: {toks / secs:.1f}  ({toks} tokens in {secs:.1f}s)")
PY
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: 2 passed (Git Bash has `curl` and `date +%s.%N`; if `date` lacks `%N` on the test box, fall back to `python3 -c 'import time;print(time.time())'` for both timestamps — do this change if and only if the test fails on that line.)

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke.sh tests/test_smoke.py
git commit -m "feat: smoke.sh acceptance check"
```

---

### Task 6: `desktop/gpu_llm.py` — config, state, `down`

**Files:**
- Create: `desktop/__init__.py` (empty), `desktop/gpu_llm.py`
- Test: `tests/test_gpu_llm.py`

**Interfaces:**
- Produces (module `desktop.gpu_llm`):
  - `home() -> Path` — `$GPU_LLM_HOME` or `~/.gpu-llm`.
  - `load_config() -> dict` — from `home()/config.toml`; keys `host`, `ssh_port` (int), `local_port` (int, default 8000), `ssh_key` (str, optional).
  - `load_state() -> dict | None`, `save_state(d)`, `clear_state()` — `home()/state.json`.
  - `pid_alive(pid: int) -> bool`.
  - `cmd_down(args) -> int`.
  - `main(argv) -> int` with argparse subcommands `tunnel`, `status`, `logs`, `down` (the latter three wired in Task 7/8; `tunnel`/`status`/`logs` may be stubs returning 2 with "not implemented" in this task).

- [ ] **Step 1: Write the failing tests**

`tests/test_gpu_llm.py`:
```python
import json
import os
import subprocess
import sys

import pytest

from desktop import gpu_llm


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LLM_HOME", str(tmp_path))
    return tmp_path


def test_load_config_defaults_when_missing(home):
    cfg = gpu_llm.load_config()
    assert cfg["local_port"] == 8000
    assert cfg.get("host") is None


def test_load_config_reads_toml(home):
    (home / "config.toml").write_text('host = "1.2.3.4"\nssh_port = 2222\nlocal_port = 9000\nssh_key = "C:/k/id_ed25519"\n')
    cfg = gpu_llm.load_config()
    assert cfg == {"host": "1.2.3.4", "ssh_port": 2222, "local_port": 9000, "ssh_key": "C:/k/id_ed25519"}


def test_state_roundtrip(home):
    assert gpu_llm.load_state() is None
    gpu_llm.save_state({"pid": 1, "host": "h", "ssh_port": 22, "local_port": 8000})
    assert gpu_llm.load_state()["host"] == "h"
    gpu_llm.clear_state()
    assert gpu_llm.load_state() is None


def test_pid_alive_for_self_and_dead():
    assert gpu_llm.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not gpu_llm.pid_alive(p.pid)


def test_down_kills_tunnel_and_reminds(home, capsys):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    gpu_llm.save_state({"pid": p.pid, "host": "h", "ssh_port": 22, "local_port": 8000})
    rc = gpu_llm.main(["down"])
    p.wait(timeout=10)
    assert rc == 0
    assert gpu_llm.load_state() is None
    assert "still billing" in capsys.readouterr().out


def test_down_with_dead_pid_is_clean(home, capsys):
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22, "local_port": 8000})
    assert gpu_llm.main(["down"]) == 0
    assert gpu_llm.load_state() is None


def test_down_without_state(home, capsys):
    assert gpu_llm.main(["down"]) == 0
    assert "no tunnel" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'desktop'`

- [ ] **Step 3: Implement `desktop/gpu_llm.py` (config/state/down + CLI skeleton)**

```python
#!/usr/bin/env python3
"""gpu-llm: desktop side of vllm_gpu_setup. Stdlib only. Windows-first (uses ssh.exe)."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_LOCAL_PORT = 8000
REMOTE_PORT = 8000
STATUS_FILE = "/var/log/vllm.status"
LOG_FILE = "/var/log/vllm.log"


# ---------- config / state ----------

def home() -> Path:
    p = Path(os.environ.get("GPU_LLM_HOME") or Path.home() / ".gpu-llm")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config() -> dict:
    cfg = {"local_port": DEFAULT_LOCAL_PORT}
    f = home() / "config.toml"
    if f.exists():
        with f.open("rb") as fh:
            cfg.update(tomllib.load(fh))
    return cfg


def _state_file() -> Path:
    return home() / "state.json"


def load_state() -> dict | None:
    f = _state_file()
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def save_state(d: dict) -> None:
    _state_file().write_text(json.dumps(d, indent=2), encoding="utf-8")


def clear_state() -> None:
    f = _state_file()
    if f.exists():
        f.unlink()


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


# ---------- ssh helpers ----------

def ssh_bin() -> str:
    return os.environ.get("GPU_LLM_SSH", "ssh")


def ssh_base(cfg: dict) -> list[str]:
    cmd = [ssh_bin(), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
           "-p", str(cfg["ssh_port"])]
    if cfg.get("ssh_key"):
        cmd += ["-i", cfg["ssh_key"]]
    cmd.append(f"root@{cfg['host']}")
    return cmd


def ssh_run(cfg: dict, remote_cmd: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(ssh_base(cfg) + [remote_cmd], capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(ssh timed out)"


# ---------- http ----------

def http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError, ValueError):
        return 0, ""


# ---------- commands ----------

def resolve_conn(args) -> dict:
    cfg = load_config()
    if getattr(args, "host", None):
        cfg["host"] = args.host
    if getattr(args, "port", None):
        cfg["ssh_port"] = args.port
    if getattr(args, "local", None):
        cfg["local_port"] = args.local
    if not cfg.get("host") or not cfg.get("ssh_port"):
        raise SystemExit("host and ssh port required: pass --host/--port or set them in "
                         f"{home() / 'config.toml'}")
    return cfg


def cmd_down(args) -> int:
    st = load_state()
    if not st:
        print("No tunnel recorded (nothing to do).")
        return 0
    if pid_alive(st["pid"]):
        kill_pid(st["pid"])
        print(f"Tunnel pid {st['pid']} stopped.")
    else:
        print(f"Tunnel pid {st['pid']} was already gone.")
    clear_state()
    print("Reminder: the Vast instance is still billing — destroy it in the Vast console.")
    return 0


def cmd_tunnel(args) -> int:  # implemented in Task 7
    print("tunnel: not implemented", file=sys.stderr)
    return 2


def cmd_status(args) -> int:  # implemented in Task 8
    print("status: not implemented", file=sys.stderr)
    return 2


def cmd_logs(args) -> int:  # implemented in Task 8
    print("logs: not implemented", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu_llm", description="SSH tunnel + health for a Vast vLLM instance")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tunnel", help="open tunnel and wait for vLLM to be READY")
    t.add_argument("--host"); t.add_argument("--port", type=int); t.add_argument("--local", type=int)
    t.add_argument("--timeout", type=int, default=900, help="seconds to wait for READY (default 900)")
    t.add_argument("--interval", type=float, default=30.0, help="seconds between progress reports")
    t.set_defaults(fn=cmd_tunnel)

    s = sub.add_parser("status", help="tunnel + model + GPU status")
    s.add_argument("--host"); s.add_argument("--port", type=int); s.add_argument("--local", type=int)
    s.set_defaults(fn=cmd_status)

    l = sub.add_parser("logs", help="show instance vllm.log")
    l.add_argument("-f", "--follow", action="store_true")
    l.add_argument("--host"); l.add_argument("--port", type=int)
    l.set_defaults(fn=cmd_logs)

    d = sub.add_parser("down", help="close the tunnel")
    d.set_defaults(fn=cmd_down)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu_llm.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add desktop tests/test_gpu_llm.py
git commit -m "feat: gpu_llm.py config/state and down command"
```

---

### Task 7: `gpu_llm.py tunnel`

**Files:**
- Modify: `desktop/gpu_llm.py` (replace `cmd_tunnel` stub; add `wait_ready`)
- Test: `tests/test_gpu_llm.py` (append)

**Interfaces:**
- Consumes: `load_config`, `save_state`, `ssh_base`, `ssh_run`, `http_get` (Task 6).
- Produces: `wait_ready(cfg, local_port, timeout, interval, progress=print) -> tuple[bool, str]` — polls `http://127.0.0.1:{local_port}/health`; every `interval` seconds reports `cat /var/log/vllm.status; tail -n 3 /var/log/vllm.log` via `ssh_run`; returns `(True, "READY")`, `(False, "FAILED: ...")` on a remote `FAILED` status, or `(False, "timeout")`. `cmd_tunnel` spawns ssh `-N -L`, saves state, calls `wait_ready`, prints `LLM_BASE_URL=...` and `LLM_MODEL=qwen` on success.
- Test fakes: `GPU_LLM_SSH` env points to a fake ssh script; the fake must handle both `-N -L ...` (sleep) and a remote command (echo canned status/log).

- [ ] **Step 1: Write the failing tests (append to `tests/test_gpu_llm.py`)**

```python
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _health_server(ok_after: int):
    """HTTP server whose /health returns 503 for the first `ok_after` hits, then 200."""
    hits = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            if self.path == "/health" and hits["n"] > ok_after:
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            elif self.path == "/v1/models":
                body = b'{"data":[{"id":"qwen"}]}'
                self.send_response(200); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            else:
                self.send_response(503); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _fake_ssh(tmp_path, status_text: str):
    """Fake ssh: `-N` -> sleep; otherwise echo a canned status + log tail."""
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        "import sys, time\n"
        "if '-N' in sys.argv:\n"
        "    time.sleep(120)\n"
        "else:\n"
        f"    print({status_text!r}); print('log line 1'); print('log line 2')\n"
    )
    if sys.platform == "win32":
        bat = tmp_path / "fake_ssh.bat"
        bat.write_text(f'@"{sys.executable}" "{script}" %*\n')
        return str(bat)
    sh = tmp_path / "fake_ssh"
    sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    sh.chmod(0o755)
    return str(sh)


def test_wait_ready_returns_true_when_health_ok(home, monkeypatch):
    srv = _health_server(ok_after=2)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING"))
    cfg = {"host": "h", "ssh_port": 22}
    msgs = []
    try:
        ok, why = gpu_llm.wait_ready(cfg, srv.server_port, timeout=20, interval=0.2, progress=msgs.append)
    finally:
        srv.shutdown()
    assert ok and why == "READY"
    assert any("STARTING" in m for m in msgs)


def test_wait_ready_stops_early_on_remote_failed(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "FAILED: vllm exited 1"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=20, interval=0.2, progress=lambda m: None)
    finally:
        srv.shutdown()
    assert not ok and why.startswith("FAILED: vllm exited 1")


def test_wait_ready_times_out(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=1, interval=0.2, progress=lambda m: None)
    finally:
        srv.shutdown()
    assert (ok, why) == (False, "timeout")


def test_cmd_tunnel_spawns_ssh_saves_state_and_prints_env(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "READY"))
    try:
        rc = gpu_llm.main(["tunnel", "--host", "1.2.3.4", "--port", "2222",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0, out
    st = gpu_llm.load_state()
    assert st["host"] == "1.2.3.4" and st["ssh_port"] == 2222 and st["local_port"] == srv.server_port
    assert gpu_llm.pid_alive(st["pid"])
    assert f"LLM_BASE_URL=http://127.0.0.1:{srv.server_port}" in out
    assert "LLM_MODEL=qwen" in out
    gpu_llm.main(["down"])


def test_cmd_tunnel_failure_kills_ssh_and_returns_1(home, monkeypatch, capsys):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "FAILED: vllm 0.16 < 0.17"))
    try:
        rc = gpu_llm.main(["tunnel", "--host", "h", "--port", "22",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        srv.shutdown()
    assert rc == 1
    assert gpu_llm.load_state() is None
    assert "FAILED: vllm 0.16 < 0.17" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v -k "wait_ready or cmd_tunnel"`
Expected: FAIL — `AttributeError: module 'desktop.gpu_llm' has no attribute 'wait_ready'` and `cmd_tunnel` returning 2.

- [ ] **Step 3: Implement `wait_ready` and `cmd_tunnel` (replace the stub)**

```python
def wait_ready(cfg: dict, local_port: int, timeout: float, interval: float, progress=print) -> tuple[bool, str]:
    """Poll local /health; report remote status/log every `interval`; stop early on remote FAILED."""
    health = f"http://127.0.0.1:{local_port}/health"
    deadline = time.monotonic() + timeout
    next_report = 0.0
    while time.monotonic() < deadline:
        code, _ = http_get(health, timeout=2.0)
        if code == 200:
            return True, "READY"
        now = time.monotonic()
        if now >= next_report:
            next_report = now + interval
            remote = ssh_run(cfg, f"cat {STATUS_FILE} 2>/dev/null; tail -n 3 {LOG_FILE} 2>/dev/null")
            first = remote.splitlines()[0] if remote else ""
            progress(f"[instance] {remote or '(no output yet)'}")
            if first.startswith("FAILED"):
                return False, first
        time.sleep(min(1.0, interval))
    return False, "timeout"


def cmd_tunnel(args) -> int:
    cfg = resolve_conn(args)
    local_port = int(cfg["local_port"])
    old = load_state()
    if old and pid_alive(old["pid"]):
        print(f"Tunnel already running (pid {old['pid']}); run `down` first.")
        return 1
    cmd = ssh_base(cfg) + ["-N", "-L", f"{local_port}:127.0.0.1:{REMOTE_PORT}"]
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation)
    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"ssh exited immediately with code {proc.returncode}; check host/port/key.")
        return 1
    save_state({"pid": proc.pid, "host": cfg["host"], "ssh_port": int(cfg["ssh_port"]), "local_port": local_port})
    print(f"Tunnel pid {proc.pid}: 127.0.0.1:{local_port} -> {cfg['host']}:{REMOTE_PORT}. Waiting for vLLM...")
    ok, why = wait_ready(cfg, local_port, timeout=args.timeout, interval=args.interval)
    if not ok:
        print(f"Not ready: {why}")
        kill_pid(proc.pid)
        clear_state()
        return 1
    print("vLLM is READY.\n")
    print(f"LLM_BASE_URL=http://127.0.0.1:{local_port}")
    print("LLM_MODEL=qwen")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu_llm.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add desktop/gpu_llm.py tests/test_gpu_llm.py
git commit -m "feat: gpu_llm tunnel with health wait and remote progress"
```

---

### Task 8: `gpu_llm.py status` and `logs`

**Files:**
- Modify: `desktop/gpu_llm.py` (replace `cmd_status`, `cmd_logs` stubs)
- Test: `tests/test_gpu_llm.py` (append)

**Interfaces:**
- Consumes: `load_state`, `pid_alive`, `http_get`, `ssh_run`, `ssh_base`, `resolve_conn` (Tasks 6–7).
- Produces: `cmd_status` prints three lines: `tunnel: up (pid N) | down`, `models: <ids> | unreachable`, `gpu: <nvidia-smi line>`. `cmd_logs` execs `ssh ... tail [-f] -n 100 /var/log/vllm.log` with inherited stdio and returns its exit code.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_status_reports_tunnel_models_and_gpu(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "NVIDIA GeForce RTX 5090, 31000 MiB, 24000 MiB"))
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    gpu_llm.save_state({"pid": p.pid, "host": "h", "ssh_port": 22, "local_port": srv.server_port})
    try:
        rc = gpu_llm.main(["status"])
    finally:
        srv.shutdown(); p.kill()
    out = capsys.readouterr().out
    assert rc == 0
    assert f"tunnel: up (pid {p.pid})" in out
    assert "models: qwen" in out
    assert "gpu: NVIDIA GeForce RTX 5090" in out


def test_status_when_down(home, capsys):
    rc = gpu_llm.main(["status"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "tunnel: down" in out


def test_logs_invokes_ssh_tail(home, monkeypatch, capsys):
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "LOGLINE"))
    rc = gpu_llm.main(["logs", "--host", "h", "--port", "22"])
    assert rc == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v -k "status or logs"`
Expected: FAIL — stubs return 2 / print "not implemented".

- [ ] **Step 3: Implement `cmd_status` and `cmd_logs` (replace stubs)**

```python
def cmd_status(args) -> int:
    st = load_state()
    if not st or not pid_alive(st["pid"]):
        print("tunnel: down")
        return 1
    print(f"tunnel: up (pid {st['pid']}) 127.0.0.1:{st['local_port']} -> {st['host']}:{REMOTE_PORT}")
    code, body = http_get(f"http://127.0.0.1:{st['local_port']}/v1/models", timeout=3.0)
    if code == 200:
        try:
            ids = [m["id"] for m in json.loads(body).get("data", [])]
        except (ValueError, KeyError, TypeError):
            ids = []
        print(f"models: {', '.join(ids) or '(none)'}")
    else:
        print("models: unreachable")
    cfg = {"host": st["host"], "ssh_port": st["ssh_port"], **({"ssh_key": load_config()["ssh_key"]} if load_config().get("ssh_key") else {})}
    gpu = ssh_run(cfg, "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    print(f"gpu: {gpu}")
    return 0


def cmd_logs(args) -> int:
    cfg = resolve_conn(args)
    tail = ["tail", "-f" if args.follow else "", "-n", "100", LOG_FILE]
    remote = " ".join(x for x in tail if x)
    try:
        return subprocess.call(ssh_base(cfg) + [remote])
    except KeyboardInterrupt:
        return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -v`
Expected: all pass (28 + 2 smoke + 15 gpu_llm = 45)

- [ ] **Step 5: Commit**

```bash
git add desktop/gpu_llm.py tests/test_gpu_llm.py
git commit -m "feat: gpu_llm status and logs commands"
```

---

### Task 9: `vast/template.md`, `README.md`, acceptance checklist

**Files:**
- Create: `vast/template.md`, `README.md`
- Modify: `docs/superpowers/specs/2026-08-22-vllm-gpu-setup-design.md` — change `**Status:**` line to `Implemented (M1 code complete; acceptance run pending)`.

**Interfaces:** none (docs). The onstart line must match `bootstrap.sh` (Task 4) and the status/log paths in Global Constraints.

- [ ] **Step 1: Write `vast/template.md`**

```markdown
# Vast.ai rental settings (Milestone 1)

| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.27.1` |
| Launch mode | **SSH** (interactive shell; do NOT use the image's default entrypoint) |
| Disk | 60 GB minimum |
| GPU filter | RTX 5090 (1x). Also works: A100 80GB, H100 80GB, H200 |
| Offer filters | Download speed >= 500 Mbps, reliability >= 98%, CUDA >= 12.8 |
| Ports | none (vLLM binds 127.0.0.1 only; everything goes through the SSH tunnel) |

## Environment variables (template "Docker options" / env section)

```
-e HF_TOKEN=            # optional; only for gated repos
-e PROFILE=             # optional override: 5090 | a100-80 | h100-80 | h200
-e MAX_MODEL_LEN=       # optional; lower after an OOM
```

## On-start script

```
cd /root && git clone https://github.com/<you>/vllm_gpu_setup.git && cd vllm_gpu_setup && cp -n .env.example .env && nohup ./bootstrap.sh > /var/log/bootstrap.log 2>&1 &
```

Replace `<you>` with the GitHub owner after pushing this repo. For a private
repo, use a deploy token in the URL or bake an SSH deploy key into the
template's env and switch to the `git@` URL.

## What to expect

- `/var/log/bootstrap.log` — version check, detected profile.
- `/var/log/vllm.status` — `STARTING` → `READY`, or `FAILED: <reason>`.
- `/var/log/vllm.log` — vLLM output; weight download (~25 GB for NVFP4) shows here.
- Cold start on a fast host: 5–10 minutes to READY.

## Known 5090 settings

`profiles/5090.yaml` uses `Inferact/Qwen3.8-27B-NVFP4`, 32K context,
`--enforce-eager` (required to avoid CUDA-graph OOM on a single 32 GB card),
fp8 KV cache. Raise `MAX_MODEL_LEN` only after checking `nvidia-smi` headroom.
```

- [ ] **Step 2: Write `README.md`**

```markdown
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
Bootstrap exit codes: 2 unknown GPU, 3 `HF_TOKEN` required, 4 vLLM < 0.17.

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
| Time to READY | |
| smoke.sh tok/s | |
| Peak VRAM used | |
| Max stable max_model_len | |
| Cost of run | |

## Deferred (milestones 2–3)

FastAPI control plane, Vast API provisioning (`gpu-llm up`), 3090/4090
profiles (need patched vLLM), persistent weight volumes, auto-teardown.
```

- [ ] **Step 3: Update spec status line and run full suite**

Change `**Status:** Approved design, pending implementation plan` → `**Status:** Implemented (M1 code complete; acceptance run pending)`.

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add vast/template.md README.md docs/superpowers/specs/2026-08-22-vllm-gpu-setup-design.md
git commit -m "docs: Vast template, README, M1 acceptance checklist"
```

---

## Self-review notes

- **Spec coverage:** §4.1 bootstrap → Task 4; §4.2 detect → Task 2; §4.3 profiles + `profile.py` → Task 1; §4.4 serve → Task 3; §4.5 smoke → Task 5; §5 desktop CLI (`tunnel`/`status`/`logs`/`down`) → Tasks 6–8; §6 template → Task 9; §7 error table → Tasks 2–4, 7; §8 local tests → every task, acceptance run → Task 9 README.
- **Type consistency:** `profile_to_argv(profile, overrides)` used identically in Tasks 1 and 3 (via CLI); `write_status` from `lib/status.sh` used in Tasks 3–4; `ssh_base/ssh_run/http_get/load_state/save_state/clear_state/pid_alive/kill_pid/resolve_conn` defined in Task 6 and used unchanged in Tasks 7–8; status strings `STARTING`/`READY`/`FAILED: ...` identical across serve.sh, bootstrap.sh, wait_ready, and docs.
- **Test counts:** T1 9, T2 9, T3 4, T4 6, T5 2, T6 7, T7 5, T8 3 = 45.
