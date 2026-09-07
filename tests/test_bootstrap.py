import shutil
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
        "DETECT_ERR_FILE": str(tmp_path / "detect.err"),
        "SERVE": str(fakes / "serve"),
        "PROFILE": "",
        "MODEL": "",
        "HF_TOKEN": "",
    }
    return env, status


def test_happy_path_detects_and_serves(tmp_path, fakes):
    env, status = setup(tmp_path, fakes)
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "SERVE 5090 qwen3.8-27b-nvfp4" in r.stdout


def test_old_vllm_exits_4(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, version="0.16.2")
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 4
    assert status.read_text().startswith("FAILED: vllm 0.16.2 < 0.17")


def test_unknown_gpu_exits_2(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    write_fake(
        fakes, "detect",
        "echo 'detect_gpu: unknown GPU: NVIDIA GeForce RTX 4090, 24564 MiB' >&2\nexit 2\n",
    )
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 2
    assert status.read_text().startswith("FAILED: unknown gpu")
    assert "RTX 4090" in status.read_text()


def test_profile_env_override_skips_detection(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    env["PROFILE"] = "h200"
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "SERVE h200 qwen3.8-27b-bf16" in r.stdout


def test_requires_hf_token_exits_3(tmp_path, fakes):
    # make a temporary GPU dir whose default model requires a token
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


def test_unknown_gpu_dir_fails_2(tmp_path, fakes):
    env, status = setup(tmp_path, fakes)
    env["PROFILE"] = "zz-missing"
    r = run_bash(BOOT, env=env, cwd=tmp_path)
    assert r.returncode == 2
    assert status.read_text().strip() == "FAILED: no profile 'zz-missing'"


def test_gpu_dir_without_default_fails_2(tmp_path, fakes):
    d = ROOT / "profiles" / "zz-nodefault-test"
    d.mkdir()
    (d / "a.yaml").write_text("model: a" + chr(10) + "served_model_name: a" + chr(10) + "requires_hf_token: false" + chr(10))
    try:
        env, status = setup(tmp_path, fakes)
        env["PROFILE"] = "zz-nodefault-test"
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 2
        assert status.read_text().strip() == "FAILED: no default model for zz-nodefault-test (found 0)"
    finally:
        shutil.rmtree(d)


def test_dotenv_is_loaded_from_repo_root(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    dotenv = ROOT / ".env"
    assert not dotenv.exists(), "refusing to clobber a real .env"
    dotenv.write_text("PROFILE=a100-80\n")
    try:
        del env["PROFILE"]
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "SERVE a100-80 qwen3.8-27b-bf16" in r.stdout
    finally:
        dotenv.unlink()


def test_dotenv_malformed_line_is_ignored(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    dotenv = ROOT / ".env"
    assert not dotenv.exists(), "refusing to clobber a real .env"
    dotenv.write_text("BAD KEY = x\nPROFILE=a100-80\n")
    try:
        del env["PROFILE"]
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "SERVE a100-80 qwen3.8-27b-bf16" in r.stdout
        assert "malformed" in r.stderr
    finally:
        dotenv.unlink()


def test_dotenv_strips_quotes_comments_and_last_line(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    dotenv = ROOT / ".env"
    assert not dotenv.exists(), "refusing to clobber a real .env"
    # PROFILE is the last line with NO trailing newline, so the
    # "SERVE a100-80" assertion actually depends on the `|| [[ -n "$k" ]]`
    # guard reading a final, newline-less line. HF_TOKEN carries an inline
    # comment and single quotes; PROFILE carries double quotes.
    dotenv.write_bytes(b'HF_TOKEN=\'x\'  # tok\nPROFILE="a100-80"')
    try:
        del env["PROFILE"]
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "SERVE a100-80 qwen3.8-27b-bf16" in r.stdout
    finally:
        dotenv.unlink()
