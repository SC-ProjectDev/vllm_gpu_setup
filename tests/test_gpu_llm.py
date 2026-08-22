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
    try:
        gpu_llm.save_state({"pid": p.pid, "host": "h", "ssh_port": 22, "local_port": 8000})
        rc = gpu_llm.main(["down"])
        p.wait(timeout=10)
        assert rc == 0
        assert gpu_llm.load_state() is None
        assert "still billing" in capsys.readouterr().out
    finally:
        try:
            p.kill()
        except OSError:
            pass
        p.wait(timeout=10)


def test_down_with_dead_pid_is_clean(home, capsys):
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22, "local_port": 8000})
    assert gpu_llm.main(["down"]) == 0
    assert gpu_llm.load_state() is None


def test_down_without_state(home, capsys):
    assert gpu_llm.main(["down"]) == 0
    assert "no tunnel" in capsys.readouterr().out.lower()


import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _health_server(ok_after: int):
    """HTTP server whose /health returns 503 for the first `ok_after` hits, then 200."""
    hits = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            if self.path == "/health" and hits["n"] > ok_after:
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            elif self.path == "/v1/models":
                body = b'{"data":[{"id":"qwen"}]}'
                self.send_response(200); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            else:
                self.send_response(503); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _fake_ssh(tmp_path, status_text: str):
    """Fake ssh: `-N` -> sleep; otherwise echo a canned status + log tail."""
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        "import sys, time\n"
        "if '-N' in sys.argv:\n"
        "    time.sleep(120)\n"
        "else:\n"
        f"    print({status_text!r}); print('log line 1'); print('log line 2')\n"
    )
    if sys.platform == "win32":
        bat = tmp_path / "fake_ssh.bat"
        bat.write_text(f'@"{sys.executable}" "{script}" %*\n')
        return str(bat)
    sh = tmp_path / "fake_ssh"
    sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    sh.chmod(0o755)
    return str(sh)


def test_wait_ready_returns_true_when_health_ok(home, monkeypatch):
    srv = _health_server(ok_after=2)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING"))
    cfg = {"host": "h", "ssh_port": 22}
    msgs = []
    try:
        ok, why = gpu_llm.wait_ready(cfg, srv.server_port, timeout=20, interval=0.2, progress=msgs.append)
    finally:
        srv.shutdown()
    assert ok and why == "READY"
    assert any("STARTING" in m for m in msgs)


def test_wait_ready_stops_early_on_remote_failed(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "FAILED: vllm exited 1"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=20, interval=0.2, progress=lambda m: None)
    finally:
        srv.shutdown()
    assert not ok and why.startswith("FAILED: vllm exited 1")


def test_wait_ready_times_out(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=1, interval=0.2, progress=lambda m: None)
    finally:
        srv.shutdown()
    assert (ok, why) == (False, "timeout")


def test_cmd_tunnel_spawns_ssh_saves_state_and_prints_env(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "READY"))
    try:
        rc = gpu_llm.main(["tunnel", "--host", "1.2.3.4", "--port", "2222",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert rc == 0, out
    st = gpu_llm.load_state()
    assert st["host"] == "1.2.3.4" and st["ssh_port"] == 2222 and st["local_port"] == srv.server_port
    assert gpu_llm.pid_alive(st["pid"])
    assert f"LLM_BASE_URL=http://127.0.0.1:{srv.server_port}" in out
    assert "LLM_MODEL=qwen" in out
    gpu_llm.main(["down"])


def test_cmd_tunnel_failure_kills_ssh_and_returns_1(home, monkeypatch, capsys):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "FAILED: vllm 0.16 < 0.17"))
    try:
        rc = gpu_llm.main(["tunnel", "--host", "h", "--port", "22",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        srv.shutdown()
    assert rc == 1
    assert gpu_llm.load_state() is None
    assert "FAILED: vllm 0.16 < 0.17" in capsys.readouterr().out
