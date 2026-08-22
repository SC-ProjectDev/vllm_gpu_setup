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
