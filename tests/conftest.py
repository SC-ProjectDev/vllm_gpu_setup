import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")


def write_fake(dir_: Path, name: str, body: str) -> Path:
    """Create an executable bash script `dir_/name` with `body` (no shebang needed)."""
    p = dir_ / name
    p.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _to_bash_path(p: Path) -> str:
    """Convert a Windows path to Git Bash format (e.g., F:/path -> /f/path)."""
    posix = p.as_posix()
    # Convert drive letter: F:/... -> /f/...
    if len(posix) >= 2 and posix[1] == ':':
        return '/' + posix[0].lower() + posix[2:]
    return posix


def run_bash(script: Path, *args: str, env: dict | None = None, cwd: Path | None = None):
    """Run a repo bash script through `bash`, returning CompletedProcess with text output."""
    full_env = dict(os.environ)
    if env:
        # Work on a copy to avoid mutating caller's dict
        env_copy = dict(env)
        # Convert only Windows absolute paths (e.g., C:\... or D:/) to bash paths
        for key, value in env_copy.items():
            if isinstance(value, str) and re.match(r"^[A-Za-z]:[/\\]", value):
                env_copy[key] = _to_bash_path(Path(value))
        full_env.update(env_copy)
    return subprocess.run(
        [BASH, _to_bash_path(script), *args],
        capture_output=True, text=True, env=full_env, cwd=cwd or ROOT,
    )


@pytest.fixture
def fakes(tmp_path):
    d = tmp_path / "fakes"
    d.mkdir()
    return d
