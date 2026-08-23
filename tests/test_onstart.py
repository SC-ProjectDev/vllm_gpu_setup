from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_onstart_script_exists_and_runs_bootstrap():
    text = (REPO_ROOT / "vast" / "onstart.sh").read_text(encoding="utf-8")
    assert "git clone https://github.com/SC-ProjectDev/vllm_gpu_setup.git" in text
    assert "bootstrap.sh" in text
    assert "/var/log/bootstrap.log" in text


def test_template_points_at_onstart_file():
    md = (REPO_ROOT / "vast" / "template.md").read_text(encoding="utf-8")
    assert "vast/onstart.sh" in md
