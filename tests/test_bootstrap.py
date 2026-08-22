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


def test_dotenv_malformed_line_is_ignored(tmp_path, fakes):
    env, status = setup(tmp_path, fakes, detect_out="", detect_rc=2)
    dotenv = ROOT / ".env"
    assert not dotenv.exists(), "refusing to clobber a real .env"
    dotenv.write_text("BAD KEY = x\nPROFILE=a100-80\n")
    try:
        del env["PROFILE"]
        r = run_bash(BOOT, env=env, cwd=tmp_path)
        assert r.returncode == 0, r.stderr
        assert "SERVE a100-80" in r.stdout
        assert "malformed" in r.stderr
    finally:
        dotenv.unlink()
