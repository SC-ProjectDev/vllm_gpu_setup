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


def test_template_onstart_command_matches_file_verbatim():
    """The one-line command in vast/onstart.sh must appear byte-identical inside
    a fenced code block in vast/template.md -- manual (console paste) and API
    (gpu-llm up) rentals read from two different places and must not drift."""
    onstart_line = (REPO_ROOT / "vast" / "onstart.sh").read_text(encoding="utf-8").strip()
    md = (REPO_ROOT / "vast" / "template.md").read_text(encoding="utf-8")
    fenced_blocks = md.split("```")[1::2]
    assert any(onstart_line in block for block in fenced_blocks), (
        "onstart.sh command not found verbatim in a fenced block of template.md"
    )
