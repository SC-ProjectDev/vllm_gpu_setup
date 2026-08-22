#!/usr/bin/env python3
"""gpu-llm: desktop side of vllm_gpu_setup. Stdlib only. Windows-first (uses ssh.exe)."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_LOCAL_PORT = 8000
REMOTE_PORT = 8000
STATUS_FILE = "/var/log/vllm.status"
LOG_FILE = "/var/log/vllm.log"


# ---------- config / state ----------

def home() -> Path:
    p = Path(os.environ.get("GPU_LLM_HOME") or Path.home() / ".gpu-llm")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config() -> dict:
    cfg = {"local_port": DEFAULT_LOCAL_PORT}
    f = home() / "config.toml"
    if f.exists():
        with f.open("rb") as fh:
            cfg.update(tomllib.load(fh))
    return cfg


def _state_file() -> Path:
    return home() / "state.json"


def load_state() -> dict | None:
    f = _state_file()
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def save_state(d: dict) -> None:
    _state_file().write_text(json.dumps(d, indent=2), encoding="utf-8")


def clear_state() -> None:
    f = _state_file()
    if f.exists():
        f.unlink()


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


# ---------- ssh helpers ----------

def ssh_bin() -> str:
    return os.environ.get("GPU_LLM_SSH", "ssh")


def ssh_base(cfg: dict) -> list[str]:
    cmd = [ssh_bin(), "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
           "-p", str(cfg["ssh_port"])]
    if cfg.get("ssh_key"):
        cmd += ["-i", cfg["ssh_key"]]
    cmd.append(f"root@{cfg['host']}")
    return cmd


def ssh_run(cfg: dict, remote_cmd: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(ssh_base(cfg) + [remote_cmd], capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(ssh timed out)"


# ---------- http ----------

def http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError, ValueError):
        return 0, ""


# ---------- commands ----------

def resolve_conn(args) -> dict:
    cfg = load_config()
    if getattr(args, "host", None):
        cfg["host"] = args.host
    if getattr(args, "port", None):
        cfg["ssh_port"] = args.port
    if getattr(args, "local", None):
        cfg["local_port"] = args.local
    if not cfg.get("host") or not cfg.get("ssh_port"):
        raise SystemExit("host and ssh port required: pass --host/--port or set them in "
                         f"{home() / 'config.toml'}")
    return cfg


def cmd_down(args) -> int:
    st = load_state()
    if not st:
        print("No tunnel recorded (nothing to do).")
        return 0
    if pid_alive(st["pid"]):
        kill_pid(st["pid"])
        print(f"Tunnel pid {st['pid']} stopped.")
    else:
        print(f"Tunnel pid {st['pid']} was already gone.")
    clear_state()
    print("Reminder: the Vast instance is still billing — destroy it in the Vast console.")
    return 0


def cmd_tunnel(args) -> int:  # implemented in Task 7
    print("tunnel: not implemented", file=sys.stderr)
    return 2


def cmd_status(args) -> int:  # implemented in Task 8
    print("status: not implemented", file=sys.stderr)
    return 2


def cmd_logs(args) -> int:  # implemented in Task 8
    print("logs: not implemented", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu_llm", description="SSH tunnel + health for a Vast vLLM instance")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tunnel", help="open tunnel and wait for vLLM to be READY")
    t.add_argument("--host"); t.add_argument("--port", type=int); t.add_argument("--local", type=int)
    t.add_argument("--timeout", type=int, default=900, help="seconds to wait for READY (default 900)")
    t.add_argument("--interval", type=float, default=30.0, help="seconds between progress reports")
    t.set_defaults(fn=cmd_tunnel)

    s = sub.add_parser("status", help="tunnel + model + GPU status")
    s.add_argument("--host"); s.add_argument("--port", type=int); s.add_argument("--local", type=int)
    s.set_defaults(fn=cmd_status)

    l = sub.add_parser("logs", help="show instance vllm.log")
    l.add_argument("-f", "--follow", action="store_true")
    l.add_argument("--host"); l.add_argument("--port", type=int)
    l.set_defaults(fn=cmd_logs)

    d = sub.add_parser("down", help="close the tunnel")
    d.set_defaults(fn=cmd_down)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
