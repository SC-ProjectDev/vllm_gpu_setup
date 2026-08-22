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
