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


def test_serve_failure_after_ready_writes_failed(tmp_path, fakes):
    write_fake(fakes, "vllm", 'sleep 1; echo "CUDA out of memory" >&2; exit 3\n')
    write_fake(fakes, "curl", "exit 0\n")  # always healthy
    env, status, log = env_for(tmp_path, fakes)
    r = run_bash(SERVE, "5090", env=env)
    assert r.returncode == 3
    assert status.read_text().strip() == "FAILED: vllm exited 3"
    assert "out of memory" in r.stdout
    assert "MAX_MODEL_LEN" in r.stdout


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
