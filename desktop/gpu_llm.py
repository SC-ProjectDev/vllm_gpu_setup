#!/usr/bin/env python3
"""gpu-llm: desktop side of vllm_gpu_setup. Stdlib only. Windows-first (uses ssh.exe)."""
from __future__ import annotations

import argparse
import csv
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
BOOTSTRAP_LOG = "/var/log/bootstrap.log"


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


def pid_alive(pid: int, image: str | None = None) -> bool:
    """True if `pid` is a running process. On Windows, if `image` is given, the
    process's image name must match it -- or be `cmd.exe`, since a .bat wrapper
    (used by the test suite's fake ssh) runs as a cmd.exe child holding the pid
    we recorded -- otherwise a reused pid for an unrelated process is rejected.
    The match is on the stem, case-insensitively, so "ssh" (the CLI default),
    "ssh.exe" (what tasklist actually reports), and "SSH.EXE" all agree."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True, text=True,
        ).stdout
        for row in csv.reader(out.splitlines()):
            if len(row) < 2:
                continue
            row_image, row_pid = row[0], row[1]
            if row_pid != str(pid):
                continue
            if image is not None:
                row_stem = Path(row_image).stem.casefold()
                if row_stem not in (Path(image).stem.casefold(), "cmd"):
                    return False
            return True
        return False
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

# Popen handles for tunnels we intentionally leave running past cmd_tunnel's
# return (the OS process persists until `down` kills it by pid); kept here so
# Python doesn't finalize the handle mid-process and warn "still running".
_spawned: list[subprocess.Popen] = []


def ssh_bin() -> str:
    return os.environ.get("GPU_LLM_SSH", "ssh")


def ssh_opts(cfg: dict) -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-p", str(cfg["ssh_port"])]
    if cfg.get("ssh_key"):
        opts += ["-i", cfg["ssh_key"]]
    return opts


def ssh_dest(cfg: dict) -> str:
    return f"root@{cfg['host']}"


def ssh_base(cfg: dict) -> list[str]:
    return [ssh_bin(), *ssh_opts(cfg), ssh_dest(cfg)]


def tunnel_cmd(cfg: dict, local_port: int) -> list[str]:
    """ssh -N -L command with options before the destination, so OpenSSH parses
    -N/-L as flags instead of a remote command."""
    return [ssh_bin(), *ssh_opts(cfg), "-N", "-L", f"{local_port}:127.0.0.1:{REMOTE_PORT}", ssh_dest(cfg)]


def ssh_log_path() -> Path:
    return home() / "ssh.log"


def print_ssh_log_tail(n: int = 10) -> None:
    """Print the last `n` lines of the tunnel ssh's stderr log, if any."""
    path = ssh_log_path()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if not lines:
        return
    print(f"ssh.log tail ({path}):")
    for line in lines[-n:]:
        print(f"  {line}")


def ssh_run(cfg: dict, remote_cmd: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(ssh_base(cfg) + [remote_cmd], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return f"(ssh failed rc={r.returncode}) {(r.stderr or r.stdout).strip()}"
        return (r.stdout or r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(ssh timed out)"


# ---------- http ----------

def http_get(url: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        e.close()
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
    if pid_alive(st["pid"], st.get("image")):
        kill_pid(st["pid"])
        print(f"Tunnel pid {st['pid']} stopped.")
    else:
        print(f"Tunnel pid {st['pid']} was already gone.")
    clear_state()
    print("Reminder: the Vast instance is still billing — destroy it in the Vast console.")
    return 0


def wait_ready(cfg: dict, local_port: int, timeout: float, interval: float, progress=print,
               proc: subprocess.Popen | None = None) -> tuple[bool, str]:
    """Poll local /health; report remote status/log every `interval`; stop early on remote
    FAILED or (if `proc` is given) if the tunnel ssh process has died."""
    health = f"http://127.0.0.1:{local_port}/health"
    deadline = time.monotonic() + timeout
    next_report = 0.0
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False, f"ssh exited with code {proc.returncode}"
        code, _ = http_get(health, timeout=2.0)
        if code == 200:
            return True, "READY"
        now = time.monotonic()
        if now >= next_report:
            next_report = now + interval
            remote = ssh_run(
                cfg,
                f"cat {STATUS_FILE} 2>/dev/null; "
                f"tail -n 3 {BOOTSTRAP_LOG} 2>/dev/null; "
                f"tail -n 3 {LOG_FILE} 2>/dev/null",
            )
            first = remote.splitlines()[0] if remote else ""
            progress(f"[instance] {remote or '(no output yet)'}")
            if first.startswith("FAILED"):
                return False, first
        time.sleep(min(1.0, interval))
    return False, "timeout"


def cmd_tunnel(args) -> int:
    cfg = resolve_conn(args)
    local_port = int(cfg["local_port"])
    old = load_state()
    if old and pid_alive(old["pid"], old.get("image")):
        print(f"Tunnel already running (pid {old['pid']}); run `down` first.")
        return 1
    cmd = tunnel_cmd(cfg, local_port)
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    ssh_log = ssh_log_path()
    ssh_log_f = ssh_log.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ssh_log_f, creationflags=creation)
    finally:
        # The child has its own duplicated fd; our handle isn't needed once spawned.
        ssh_log_f.close()
    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"ssh exited with code {proc.returncode} immediately; check host/port/key.")
        print_ssh_log_tail()
        return 1
    image = Path(ssh_bin()).name
    save_state({"pid": proc.pid, "host": cfg["host"], "ssh_port": int(cfg["ssh_port"]), "local_port": local_port,
                "image": image})
    print(f"Tunnel pid {proc.pid}: 127.0.0.1:{local_port} -> {cfg['host']}:{REMOTE_PORT}. Waiting for vLLM...")
    ok, why = wait_ready(cfg, local_port, timeout=args.timeout, interval=args.interval, proc=proc)
    if not ok:
        print(f"Not ready: {why}")
        if why.startswith("ssh exited"):
            print_ssh_log_tail()
        kill_pid(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        clear_state()
        return 1
    _spawned.append(proc)
    print("vLLM is READY.\n")
    print(f"LLM_BASE_URL=http://127.0.0.1:{local_port}")
    print("LLM_MODEL=qwen")
    return 0


def state_cfg(st: dict) -> dict:
    """Build an ssh cfg (host/ssh_port/ssh_key) from saved tunnel state, pulling
    ssh_key from config.toml if one is set there."""
    cfg = {"host": st["host"], "ssh_port": st["ssh_port"]}
    ssh_key = load_config().get("ssh_key")
    if ssh_key:
        cfg["ssh_key"] = ssh_key
    return cfg


def cmd_status(args) -> int:
    st = load_state()
    if not st or not pid_alive(st["pid"], st.get("image")):
        print("tunnel: down")
        return 1
    print(f"tunnel: up (pid {st['pid']}) 127.0.0.1:{st['local_port']} -> {st['host']}:{REMOTE_PORT}")
    code, body = http_get(f"http://127.0.0.1:{st['local_port']}/v1/models", timeout=3.0)
    if code == 200:
        try:
            ids = [m["id"] for m in json.loads(body).get("data", [])]
        except (ValueError, KeyError, TypeError):
            ids = []
        print(f"models: {', '.join(ids) or '(none)'}")
    else:
        print("models: unreachable")
    gpu = ssh_run(state_cfg(st), "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
    print(f"gpu: {gpu}")
    return 0


def cmd_logs(args) -> int:
    cfg = resolve_conn(args)
    tail = ["tail", "-f" if args.follow else "", "-n", "100", LOG_FILE]
    remote = " ".join(x for x in tail if x)
    try:
        return subprocess.call(ssh_base(cfg) + [remote])
    except KeyboardInterrupt:
        return 0


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
