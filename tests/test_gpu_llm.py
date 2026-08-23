import os
import subprocess
import sys
import time

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


@pytest.mark.skipif(sys.platform != "win32", reason="tasklist image matching is Windows-only")
def test_pid_alive_image_match_is_stem_and_case_insensitive():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert gpu_llm.pid_alive(p.pid, "python")
        assert gpu_llm.pid_alive(p.pid, "python.exe")
        assert gpu_llm.pid_alive(p.pid, "PYTHON.EXE")
        assert not gpu_llm.pid_alive(p.pid, "notepad")
    finally:
        p.kill()
        p.wait(timeout=10)


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


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def _fake_ssh(tmp_path, status_text: str, exit_code: int = 0, tunnel_exit: int | None = None):
    """Fake ssh: `-N` -> sleep (or, if `tunnel_exit` is set, sleep ~0.5s then exit with
    that code, simulating the tunnel dying mid-wait); otherwise echo a canned status +
    log tail (or, if `exit_code` is non-zero, write `status_text` to stderr and exit
    with that code, simulating a failed remote command).

    The status line is printed first (so callers that only look at the first
    line, like `wait_ready`'s FAILED check, are unaffected), then an
    `ARGV: ...` line with the full argv the fake ssh was invoked with, then
    the canned log lines.

    Exits 2 if `-N` appears after the destination (`root@...`) argument,
    which is what OpenSSH sees as "run this as a remote command" instead of
    "open a tunnel" -- catches a regression in ssh argument ordering.
    """
    if exit_code:
        non_n_body = (
            f"    sys.stderr.write({status_text!r} + '\\n')\n"
            f"    sys.exit({exit_code!r})\n"
        )
    else:
        non_n_body = (
            f"    print({status_text!r})\n"
            "    print('ARGV:', ' '.join(argv))\n"
            "    print('log line 1'); print('log line 2')\n"
        )
    if tunnel_exit is not None:
        n_body = f"    time.sleep(0.5)\n    sys.exit({tunnel_exit!r})\n"
    else:
        n_body = "    time.sleep(120)\n"
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        "import sys, time\n"
        "argv = sys.argv[1:]\n"
        "dest_idx = next((i for i, a in enumerate(argv) if '@' in a), None)\n"
        "if '-N' in argv and dest_idx is not None and argv.index('-N') > dest_idx:\n"
        "    sys.stderr.write('fake_ssh: -N appears after destination arg\\n')\n"
        "    sys.exit(2)\n"
        "if '-N' in argv:\n"
        + n_body
        + "else:\n"
        + non_n_body
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
        _stop(srv)
    assert ok and why == "READY"
    assert any("STARTING" in m for m in msgs)


def test_wait_ready_stops_early_on_remote_failed(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "FAILED: vllm exited 1"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=20, interval=0.2, progress=lambda m: None)
    finally:
        _stop(srv)
    assert not ok and why.startswith("FAILED: vllm exited 1")


def test_wait_ready_times_out(home, monkeypatch):
    srv = _health_server(ok_after=10_000)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING"))
    try:
        ok, why = gpu_llm.wait_ready({"host": "h", "ssh_port": 22}, srv.server_port,
                                     timeout=1, interval=0.2, progress=lambda m: None)
    finally:
        _stop(srv)
    assert (ok, why) == (False, "timeout")


def test_cmd_tunnel_spawns_ssh_saves_state_and_prints_env(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "READY"))
    try:
        rc = gpu_llm.main(["tunnel", "--host", "1.2.3.4", "--port", "2222",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        _stop(srv)
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
        _stop(srv)
    assert rc == 1
    assert gpu_llm.load_state() is None
    assert "FAILED: vllm 0.16 < 0.17" in capsys.readouterr().out


def test_cmd_tunnel_ssh_dying_mid_wait_returns_1_fast(home, monkeypatch, capsys):
    srv = _health_server(ok_after=10_000)  # never becomes healthy on its own
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "STARTING", tunnel_exit=7))
    start = time.monotonic()
    try:
        rc = gpu_llm.main(["tunnel", "--host", "h", "--port", "22",
                           "--local", str(srv.server_port), "--timeout", "20", "--interval", "0.2"])
    finally:
        _stop(srv)
    elapsed = time.monotonic() - start
    out = capsys.readouterr().out
    assert rc == 1
    assert elapsed < 10, f"took {elapsed}s, expected well under the 20s timeout"
    assert "ssh exited with code 7" in out
    assert gpu_llm.load_state() is None


def test_tunnel_command_puts_options_before_destination():
    cfg = {"host": "h", "ssh_port": 22, "ssh_key": "k"}
    cmd = gpu_llm.tunnel_cmd(cfg, 8000)
    assert cmd[-1] == "root@h"
    dest_idx = len(cmd) - 1
    for flag in ("-N", "-L", "-p", "-i"):
        assert flag in cmd
        assert cmd.index(flag) < dest_idx


def test_status_reports_tunnel_models_and_gpu(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "NVIDIA GeForce RTX 5090, 31000 MiB, 24000 MiB"))
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gpu_llm.save_state({"pid": p.pid, "host": "h", "ssh_port": 22, "local_port": srv.server_port})
        rc = gpu_llm.main(["status"])
    finally:
        _stop(srv)
        try:
            p.kill()
        except OSError:
            pass
        p.wait(timeout=10)
    out = capsys.readouterr().out
    assert rc == 0
    assert f"tunnel: up (pid {p.pid})" in out
    assert "models: qwen" in out
    assert "gpu: NVIDIA GeForce RTX 5090" in out


def test_status_when_down(home, capsys):
    rc = gpu_llm.main(["status"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "tunnel: down" in out


def test_status_gpu_line_shows_ssh_failure(home, monkeypatch, capsys):
    srv = _health_server(ok_after=0)
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "permission denied", exit_code=1))
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gpu_llm.save_state({"pid": p.pid, "host": "h", "ssh_port": 22, "local_port": srv.server_port})
        rc = gpu_llm.main(["status"])
    finally:
        _stop(srv)
        try:
            p.kill()
        except OSError:
            pass
        p.wait(timeout=10)
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpu: (ssh failed rc=1)" in out
    assert "permission denied" in out


def test_logs_invokes_ssh_tail(home, monkeypatch, capfd):
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "LOGLINE"))
    rc = gpu_llm.main(["logs", "--host", "h", "--port", "22"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "tail -n 100 /var/log/vllm.log" in out
    assert out.index("root@h") < out.index("tail")


def test_logs_follow_passes_dash_f(home, monkeypatch, capfd):
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "LOGLINE"))
    rc = gpu_llm.main(["logs", "-f", "--host", "h", "--port", "22"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "tail -f -n 100 /var/log/vllm.log" in out


def test_resolve_api_key_env_beats_config(home, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "ENVKEY")
    assert gpu_llm.resolve_api_key({"api_key": "CFGKEY"}) == "ENVKEY"
    monkeypatch.delenv("VAST_API_KEY")
    assert gpu_llm.resolve_api_key({"api_key": "CFGKEY"}) == "CFGKEY"
    assert gpu_llm.resolve_api_key({}) is None


def test_resolve_max_price_precedence():
    cfg = {"max_price": {"5090": 0.8}}
    assert gpu_llm.resolve_max_price(cfg, "5090", 0.7) == 0.7
    assert gpu_llm.resolve_max_price(cfg, "5090", None) == 0.8
    assert gpu_llm.resolve_max_price({}, "5090", None) == 1.00
    assert gpu_llm.resolve_max_price({}, "h200", None) == 3.50


def test_merge_state_preserves_existing_fields(home):
    gpu_llm.save_state({"instance_id": 42, "gpu": "5090", "dph": 0.59})
    gpu_llm.merge_state({"pid": 123, "host": "h"})
    st = gpu_llm.load_state()
    assert st["instance_id"] == 42 and st["pid"] == 123 and st["host"] == "h"


def test_resolve_conn_falls_back_to_state(home):
    gpu_llm.save_state({"instance_id": 42, "host": "9.9.9.9", "ssh_port": 1234, "pid": 0})
    args = type("A", (), {"host": None, "port": None, "local": None})()
    cfg = gpu_llm.resolve_conn(args)
    assert cfg["host"] == "9.9.9.9" and cfg["ssh_port"] == 1234


def test_resolve_conn_config_beats_state(home):
    (home / "config.toml").write_text('host = "1.2.3.4"\nssh_port = 22\n')
    gpu_llm.save_state({"host": "9.9.9.9", "ssh_port": 1234, "pid": 0})
    args = type("A", (), {"host": None, "port": None, "local": None})()
    assert gpu_llm.resolve_conn(args)["host"] == "1.2.3.4"


def test_status_shows_instance_line(home, monkeypatch, capsys):
    monkeypatch.setenv("GPU_LLM_SSH", _fake_ssh(home, "gpu-line"))
    gpu_llm.save_state({"pid": os.getpid(), "host": "h", "ssh_port": 22,
                       "local_port": 8000, "image": "python",
                       "instance_id": 4242, "dph": 0.592})
    gpu_llm.main(["status"])
    out = capsys.readouterr().out
    assert "instance: 4242 ($0.592/hr)" in out


@pytest.fixture
def up_env(home, monkeypatch):
    """cmd_up with all externals stubbed: api calls recorded, tunnel skipped."""
    calls = {"destroyed": [], "rented": [], "tunnel": 0}
    monkeypatch.setenv("VAST_API_KEY", "K")
    def _search(k, q):
        calls.setdefault("queries", []).append(q)
        return calls.get("offers", [])
    monkeypatch.setattr(gpu_llm.vast_api, "search_offers", _search)
    monkeypatch.setattr(gpu_llm.vast_api, "rent_offer",
                        lambda k, oid, image, disk, onstart: calls["rented"].append((oid, image, disk, onstart)) or 4242)
    monkeypatch.setattr(gpu_llm.vast_api, "list_instances",
                        lambda k: calls.get("instances", []))
    monkeypatch.setattr(gpu_llm.vast_api, "destroy_instance",
                        lambda k, iid: calls["destroyed"].append(iid) or True)
    monkeypatch.setattr(gpu_llm, "open_tunnel_and_wait",
                        lambda cfg, timeout, interval: calls.__setitem__("tunnel", calls["tunnel"] + 1) or 0)
    monkeypatch.setattr(gpu_llm, "confirm", lambda prompt: True)
    return calls


OFFER = {"id": 7, "dph_total": 0.592, "gpu_name": "RTX 5090",
         "inet_down": 812.0, "reliability2": 0.992, "geolocation": "US"}
RUNNING = {"id": 4242, "actual_status": "running", "ssh_host": "ssh9.vast.ai", "ssh_port": 41000}


def test_up_happy_path_rents_and_tunnels(up_env, capsys):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]
    rc = gpu_llm.main(["up", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0
    assert up_env["rented"][0][0] == 7
    assert up_env["rented"][0][1] == "vllm/vllm-openai:v0.27.1"
    assert up_env["rented"][0][2] == 60
    assert "git clone" in up_env["rented"][0][3]          # onstart.sh content
    assert up_env["tunnel"] == 1
    st = gpu_llm.load_state()
    assert st["instance_id"] == 4242 and st["gpu"] == "5090" and st["dph"] == 0.592
    assert st["host"] == "ssh9.vast.ai" and st["ssh_port"] == 41000
    assert "gpu-llm down" in out                           # cost reminder printed


def test_up_no_api_key_exits_1(up_env, monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY")
    rc = gpu_llm.main(["up", "--yes"])
    assert rc == 1
    assert "manage-keys" in capsys.readouterr().out
    assert up_env["rented"] == []


def test_up_no_offers_lists_over_cap_and_exits_1(up_env, monkeypatch, capsys):
    seen_queries = []
    def search(k, q):
        seen_queries.append(q)
        return [] if "dph_total" in q else [OFFER, OFFER, OFFER, OFFER]
    monkeypatch.setattr(gpu_llm.vast_api, "search_offers", search)
    rc = gpu_llm.main(["up", "--yes"])
    out = capsys.readouterr().out
    assert rc == 1
    assert up_env["rented"] == []
    assert out.count("RTX 5090") == 3                     # exactly 3 over-cap offers shown
    assert "dph_total" not in seen_queries[1]             # second query uncapped


def test_up_decline_rents_nothing(up_env, monkeypatch, capsys):
    up_env["offers"] = [OFFER]
    monkeypatch.setattr(gpu_llm, "confirm", lambda prompt: False)
    rc = gpu_llm.main(["up"])
    assert rc == 0
    assert up_env["rented"] == []
    assert "Nothing rented" in capsys.readouterr().out


def test_up_dead_on_arrival_asks_destroy(up_env, monkeypatch, capsys):
    up_env["offers"] = [OFFER]
    up_env["instances"] = [{"id": 4242, "actual_status": "exited"}]
    monkeypatch.setattr(gpu_llm, "confirm", lambda prompt: True)  # yes to destroy
    rc = gpu_llm.main(["up", "--yes"])
    assert rc == 1
    assert up_env["destroyed"] == [4242]
    assert gpu_llm.load_state() is None


def test_up_refuses_when_instance_recorded_and_live(up_env, capsys):
    gpu_llm.save_state({"pid": 0, "instance_id": 4242})
    up_env["instances"] = [RUNNING]
    rc = gpu_llm.main(["up", "--yes"])
    assert rc == 1
    assert "down" in capsys.readouterr().out
    assert up_env["rented"] == []


def test_up_clears_stale_state_and_proceeds(up_env, capsys):
    gpu_llm.save_state({"pid": 999999999, "instance_id": 1111})   # dead pid, gone instance
    up_env["offers"] = [OFFER]
    up_env["instances"] = [RUNNING]                                # 1111 not present
    rc = gpu_llm.main(["up", "--yes"])
    assert rc == 0
    assert gpu_llm.load_state()["instance_id"] == 4242


def test_wait_instance_running_timeout(up_env, monkeypatch):
    monkeypatch.setattr(gpu_llm.vast_api, "list_instances", lambda k: [])
    inst, status = gpu_llm.wait_instance_running("K", 4242, timeout=0.3, interval=0.1)
    assert inst is None and status == "timeout"


def test_confirm_yes_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert gpu_llm.confirm("rent? ")
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert not gpu_llm.confirm("rent? ")
