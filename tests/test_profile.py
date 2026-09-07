from pathlib import Path

import pytest

from lib.profile import default_model, list_models, load_profile, profile_to_argv

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
GPUS = ("5090", "a100-80", "h100-80", "h200")
ALL_PROFILES = sorted(p for g in GPUS for p in (PROFILES / g).glob("*.yaml"))
P5090 = PROFILES / "5090" / "qwen3.8-27b-nvfp4.yaml"


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
    argv = profile_to_argv(load_profile(P5090), {})
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
    argv = profile_to_argv(load_profile(P5090), {"max_model_len": "16384"})
    assert argv[argv.index("--max-model-len") + 1] == "16384"


def test_empty_override_is_ignored():
    argv = profile_to_argv(load_profile(P5090), {"max_model_len": ""})
    assert argv[argv.index("--max-model-len") + 1] == "32768"


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
    argv = profile_to_argv(p, {})
    assert argv[1:4] == ["--served-model-name", "local", path.stem]


@pytest.mark.parametrize("gpu", GPUS)
def test_exactly_one_default_per_gpu(gpu):
    models = list_models(PROFILES, gpu)
    assert default_model(PROFILES, gpu) in {m["id"] for m in models}
    assert sum(m["default"] for m in models) == 1


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


def test_cli_prints_argv_one_per_line(tmp_path):
    import os, subprocess, sys
    env = dict(os.environ, MAX_MODEL_LEN="8192")
    out = subprocess.run(
        [sys.executable, str(ROOT / "lib" / "profile.py"), str(P5090)],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.splitlines()
    assert out[0] == "Inferact/Qwen3.8-27B-NVFP4"
    assert out[out.index("--max-model-len") + 1] == "8192"
