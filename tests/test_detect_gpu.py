import pytest

from tests.conftest import ROOT, run_bash, write_fake

SCRIPT = ROOT / "lib" / "detect_gpu.sh"


def detect(fakes, smi_output: str):
    fake = write_fake(fakes, "fake_smi", f"printf '%b\\n' {smi_output!r}\n")
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
