#!/usr/bin/env python3
"""gpu-llm: desktop side of vllm_gpu_setup. Stdlib only. Windows-first (uses ssh.exe)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

try:
    from desktop import vast_api
except ImportError:  # run as `python desktop/gpu_llm.py` from repo root
    import vast_api

DEFAULT_LOCAL_PORT = 8000
REMOTE_PORT = 8000
STATUS_FILE = "/var/log/vllm.status"
LOG_FILE = "/var/log/vllm.log"
BOOTSTRAP_LOG = "/var/log/bootstrap.log"

IMAGE = "vllm/vllm-openai:v0.27.1"
DISK_GB = 60
INSTANCE_WAIT_SECS = 600
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python desktop/gpu_llm.py` puts desktop/ first, not the repo
    sys.path.insert(0, str(REPO_ROOT))
from lib.profile import list_models  # noqa: E402

PROFILES_DIR = REPO_ROOT / "profiles"
MANAGE_KEYS_URL = "https://cloud.vast.ai/manage-keys/"
CONSOLE_INSTANCES_URL = "https://cloud.vast.ai/instances/"
# Model ids and API keys are single-quoted into the onstart shell text; keep them boring.
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


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


def merge_state(fields: dict) -> None:
    save_state({**(load_state() or {}), **fields})


def resolve_api_key(cfg: dict) -> str | None:
    return os.environ.get("VAST_API_KEY") or cfg.get("api_key")


def llm_api_key_path() -> Path:
    return home() / "llm_api_key"


def resolve_llm_api_key(cfg: dict) -> str:
    """The bearer token vLLM is started with and clients must send.
    LLM_API_KEY env > config `llm_api_key` > ~/.gpu-llm/llm_api_key > generate and save."""
    key = os.environ.get("LLM_API_KEY") or cfg.get("llm_api_key")
    f = llm_api_key_path()
    if not key and f.exists():
        key = f.read_text(encoding="utf-8").strip()
    if not key:
        key = secrets.token_urlsafe(24)
        try:
            f.write_text(key + "\n", encoding="utf-8")
            if sys.platform != "win32":
                f.chmod(0o600)
            print(f"Generated LLM API key -> {f}")
        except OSError as e:
            print(f"WARNING: could not save LLM API key to {f} ({e}); using it for this run only.")
    if not SAFE_TOKEN.match(key):
        raise SystemExit("llm_api_key must match [A-Za-z0-9._-]+")
    return key


def resolve_max_price(cfg: dict, gpu: str, cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    from_cfg = (cfg.get("max_price") or {}).get(gpu)
    if from_cfg is not None:
        return float(from_cfg)
    return vast_api.GPU_FILTERS[gpu]["max_price"]


def pid_alive(pid: int, image: str | None = None) -> bool:
    """True if `pid` is a running process. On Windows, if `image` is given, the
    process's image name must match it -- or be `cmd.exe`, since a .bat wrapper
    (used by the test suite's fake ssh) runs as a cmd.exe child holding the pid
    we recorded -- otherwise a reused pid for an unrelated process is rejected.
    The match is on the stem, case-insensitively, so "ssh" (the CLI default),
    "ssh.exe" (what tasklist actually reports), and "SSH.EXE" all agree."""
    if not pid:
        # pid 0 means "no tunnel recorded" (crash-safe rent state); on Windows
        # it matches the System Idle Process and on POSIX os.kill(0, 0) is a
        # no-op that succeeds, so it must never be treated as "alive".
        return False
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

def http_get(url: str, timeout: float = 3.0, headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
        st = load_state()
        if st and st.get("host") and st.get("ssh_port"):
            cfg.setdefault("host", st["host"])
            cfg.setdefault("ssh_port", st["ssh_port"])
    if not cfg.get("host") or not cfg.get("ssh_port"):
        raise SystemExit("host and ssh port required: pass --host/--port or set them in "
                         f"{home() / 'config.toml'}")
    return cfg


def cmd_down(args) -> int:
    st = load_state()
    if not st:
        print("No tunnel recorded (nothing to do).")
        return 0
    if st.get("pid") and pid_alive(st["pid"], st.get("image")):
        kill_pid(st["pid"])
        print(f"Tunnel pid {st['pid']} stopped.")
    elif st.get("pid"):
        print(f"Tunnel pid {st['pid']} was already gone.")
    iid = st.get("instance_id")
    if not iid:
        clear_state()
        print("Reminder: the Vast instance is still billing — destroy it in the Vast console.")
        return 0
    if getattr(args, "keep", False):
        merge_state({"pid": 0})
        print(f"instance {iid} kept (still billing); plain `down` destroys it later.")
        return 0
    api_key = resolve_api_key(load_config())
    if not api_key:
        print(f"WARNING: instance {iid} is recorded but no Vast API key is configured —")
        print(f"it may still be billing. Set VAST_API_KEY (see {MANAGE_KEYS_URL}) and re-run `down`,")
        print(f"or destroy it at {CONSOLE_INSTANCES_URL}")
        return 1
    try:
        gone = vast_api.destroy_instance(api_key, iid)
        print(f"instance {iid} {'destroyed' if gone else 'already gone'}.")
        clear_state()
        return 0
    except vast_api.VastError as e:
        print(f"WARNING: destroy failed ({e}). Instance {iid} may still be billing!")
        print(f"Check {CONSOLE_INSTANCES_URL} — state kept; re-run `down` to retry.")
        return 1


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


def open_tunnel_and_wait(cfg: dict, timeout: int, interval: float) -> int:
    local_port = int(cfg["local_port"])
    cmd = tunnel_cmd(cfg, local_port)
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    ssh_log_f = ssh_log_path().open("w", encoding="utf-8")
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
    merge_state({"pid": proc.pid, "host": cfg["host"], "ssh_port": int(cfg["ssh_port"]),
                 "local_port": local_port, "image": Path(ssh_bin()).name})
    print(f"Tunnel pid {proc.pid}: 127.0.0.1:{local_port} -> {cfg['host']}:{REMOTE_PORT}. Waiting for vLLM...")
    ok, why = wait_ready(cfg, local_port, timeout=timeout, interval=interval, proc=proc)
    if not ok:
        print(f"Not ready: {why}")
        if why.startswith("ssh exited"):
            print_ssh_log_tail()
        kill_pid(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        st = load_state() or {}
        if st.get("instance_id"):
            merge_state({"pid": 0})
        else:
            clear_state()
        return 1
    _spawned.append(proc)
    print("vLLM is READY.\n")
    print_client_settings(local_port)
    return 0


def state_context(st: dict) -> int | None:
    """max_model_len of the model recorded in state (needs gpu + model), else None."""
    if not (st.get("gpu") and st.get("model")):
        return None
    try:
        entry = next(m for m in list_models(PROFILES_DIR, st["gpu"]) if m["id"] == st["model"])
    except (ValueError, StopIteration):
        return None
    return entry.get("max_model_len")


def print_client_settings(local_port: int) -> None:
    """What llm-cli / Rider need: URL, the constant `local` model name, key, context."""
    st = load_state() or {}
    print(f"LLM_BASE_URL=http://127.0.0.1:{local_port}")
    print("LLM_MODEL=local")
    print(f"LLM_API_KEY={resolve_llm_api_key(load_config())}")
    ctx = state_context(st)
    if ctx:
        print(f"LLM_CONTEXT={ctx}")


def cmd_tunnel(args) -> int:
    cfg = resolve_conn(args)
    old = load_state()
    if old and pid_alive(old["pid"], old.get("image")):
        print(f"Tunnel already running (pid {old['pid']}); run `down` first.")
        return 1
    return open_tunnel_and_wait(cfg, args.timeout, args.interval)


def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def wait_instance_running(api_key: str, instance_id: int, timeout: float,
                          interval: float = 10.0, progress=print) -> tuple[dict | None, str]:
    deadline = time.monotonic() + timeout
    next_poll = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_poll:
            next_poll = now + interval
            try:
                inst = next((i for i in vast_api.list_instances(api_key)
                             if i.get("id") == instance_id), None)
            except vast_api.VastError as e:
                progress(f"[vast] {e} (retrying)")
                inst = None
            if inst:
                status = inst.get("actual_status")
                if status == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
                    return inst, "running"
                if status in ("exited", "offline", "unknown"):
                    return inst, status
                progress(f"[vast] instance {instance_id}: {status or 'starting'}...")
        time.sleep(min(1.0, interval))
    return None, "timeout"


def _destroy_and_report(api_key: str, instance_id: int) -> None:
    try:
        gone = vast_api.destroy_instance(api_key, instance_id)
        print(f"instance {instance_id} {'destroyed' if gone else 'already gone'}.")
        clear_state()
    except vast_api.VastError as e:
        print(f"WARNING: destroy failed ({e}). Instance {instance_id} may still be billing!")
        print(f"Check {CONSOLE_INSTANCES_URL} — state kept; re-run `down` to retry.")


def cmd_up(args) -> int:
    cfg = load_config()
    api_key = resolve_api_key(cfg)
    if not api_key:
        print(f"No Vast API key. Set VAST_API_KEY or api_key in {home() / 'config.toml'}")
        print(f"(create one at {MANAGE_KEYS_URL})")
        return 1
    try:
        return _cmd_up_body(args, cfg, api_key)
    except vast_api.VastError as e:
        print(f"Vast API error: {e}")
        if e.code == 401:
            print(f"Check your API key at {MANAGE_KEYS_URL}")
        return 1


def format_model(m: dict) -> str:
    parts = [m["id"], str(m.get("quant") or "?")]
    if m.get("size_gb") is not None:
        parts.append(f"{m['size_gb']} GB")
    if m.get("max_model_len"):
        parts.append(f"{m['max_model_len']} ctx")
    return "  ".join(parts) + ("  (default)" if m["default"] else "")


def print_models(models: list[dict]) -> None:
    for i, m in enumerate(models, 1):
        print(f"{i}. {format_model(m)}")


def choose_model(models: list[dict], requested: str | None, yes: bool) -> dict | None:
    """Pick a catalog entry: --model wins, then --yes / a single entry take the
    default, else a numbered prompt (Enter = default). None means abort."""
    default = next((m for m in models if m["default"]), models[0])
    if requested:
        return next((m for m in models if m["id"] == requested), None)
    if yes or len(models) == 1:
        return default
    print_models(models)
    try:
        raw = input(f"model? [1-{len(models)}, Enter = {default['id']}] ").strip()
    except EOFError:
        return None
    if raw == "":
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(models):
        return models[int(raw) - 1]
    return None


def onstart_text(model_id: str, api_key: str) -> str:
    """vast/onstart.sh with the desktop's choices exported in front of it; bootstrap
    reads MODEL / VLLM_API_KEY from the environment before .env."""
    for v in (model_id, api_key):
        if not SAFE_TOKEN.match(v):
            raise ValueError(f"unsafe value for onstart: {v!r}")
    base = (REPO_ROOT / "vast" / "onstart.sh").read_text(encoding="utf-8")
    return f"export MODEL='{model_id}' VLLM_API_KEY='{api_key}'; {base}"


def choose_offer(offers: list[dict]) -> dict | None:
    """Print a numbered offer list and return the chosen offer, or None to abort."""
    for i, o in enumerate(offers, 1):
        print(f"{i}. {vast_api.format_offer(o)} · offer {o.get('id', '?')}")
    try:
        raw = input(f"rent which? [1-{len(offers)}, Enter aborts] ").strip()
    except EOFError:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(offers):
        return offers[int(raw) - 1]
    return None


def _cmd_up_body(args, cfg: dict, api_key: str) -> int:
    """The rest of cmd_up, after the api-key check. Split out so cmd_up can wrap
    every Vast API interaction here (stale-check list_instances, both
    search_offers calls, rent_offer) in one try/except VastError -- state is
    never cleared by any of these paths, so a mid-call VastError just leaves
    whatever was on disk before the call."""
    st = load_state()
    if st:
        # pid 0 means "no tunnel recorded" (crash-safe rent state); pid_alive(0)
        # must not be consulted — on Windows it matches the System Idle Process.
        pid_live = bool(st.get("pid")) and pid_alive(st["pid"], st.get("image"))
        inst_live = st.get("instance_id") and any(
            i.get("id") == st["instance_id"] for i in vast_api.list_instances(api_key))
        if pid_live or inst_live:
            print("Already up (tunnel or instance recorded); run `down` first.")
            return 1
        clear_state()  # stale: dead pid, instance gone
    models = list_models(PROFILES_DIR, args.gpu)
    if args.model and args.model not in {m["id"] for m in models}:
        print(f"Unknown model '{args.model}' for {args.gpu}. Available:")
        print_models(models)
        return 1
    filters = cfg.get("filters") or {}
    max_price = resolve_max_price(cfg, args.gpu, args.max_price)
    if args.offer:
        # Rent a specific offer id (e.g. spotted in the Vast console). The
        # uncapped search is only a details lookup — renting proceeds either way.
        # Retry without the config [filters] so an offer outside them still
        # gets its price recorded.
        found = []
        for lookup_filters in (filters, {}):
            found = [o for o in vast_api.search_offers(
                         api_key, vast_api.build_offer_query(args.gpu, None, lookup_filters))
                     if o.get("id") == args.offer]
            if found or not filters:
                break
        offer = found[0] if found else {"id": args.offer}
        if found:
            print(f"{vast_api.format_offer(offer)} · offer {offer['id']}")
        else:
            print(f"offer {args.offer} (details not found in search; renting by id)")
        if not args.yes and not confirm("rent? [y/N] "):
            print("Nothing rented.")
            return 0
    else:
        offers = vast_api.search_offers(
            api_key, vast_api.build_offer_query(args.gpu, max_price, filters))
        if not offers:
            print(f"No {args.gpu} offers under ${max_price:.2f}/hr. Cheapest above the cap:")
            over = vast_api.search_offers(
                api_key, vast_api.build_offer_query(args.gpu, None, filters))
            for o in sorted(over, key=lambda o: o.get("dph_total", float("inf")))[:3]:
                print(f"  {vast_api.format_offer(o)}")
            return 1
        offers = sorted(offers, key=lambda o: o.get("dph_total", float("inf")))[:max(1, args.list)]
        if args.yes:
            offer = offers[0]
            print(f"{vast_api.format_offer(offer)} · offer {offer.get('id', '?')}")
        else:
            offer = choose_offer(offers)
            if offer is None:
                print("Nothing rented.")
                return 0
    model = choose_model(models, args.model, args.yes)
    if model is None:
        print("Nothing rented.")
        return 0
    llm_key = resolve_llm_api_key(cfg)
    disk = int(model.get("disk_gb") or DISK_GB)
    # Computed once, before renting, so a `dph_total`-less offer never loses
    # the instance id to a KeyError after a successful (billing) rent.
    dph = offer.get("dph_total", 0.0)
    onstart = onstart_text(model["id"], llm_key)
    iid = vast_api.rent_offer(api_key, offer["id"], IMAGE, disk, onstart)
    save_state({"pid": 0, "instance_id": iid, "gpu": args.gpu, "model": model["id"], "dph": dph})
    print(f"Rented instance {iid} ({model['id']}, {disk} GB disk) at ${dph:.3f}/hr. Waiting for SSH info...")
    inst, status = wait_instance_running(api_key, iid, INSTANCE_WAIT_SECS)
    if status != "running":
        print(f"Instance {iid} did not reach running ({status}); it is still rented.")
        if status != "timeout" and confirm(f"Destroy instance {iid}? [y/N] "):
            _destroy_and_report(api_key, iid)
        else:
            print(f"`gpu-llm down` destroys it; {CONSOLE_INSTANCES_URL} to inspect.")
        return 1
    merge_state({"host": inst["ssh_host"], "ssh_port": int(inst["ssh_port"])})
    cfg = {**cfg, "host": inst["ssh_host"], "ssh_port": int(inst["ssh_port"]),
           "local_port": cfg.get("local_port", DEFAULT_LOCAL_PORT)}
    rc = open_tunnel_and_wait(cfg, args.timeout, args.interval)
    if rc == 0:
        print(f"instance {iid} at ${dph:.3f}/hr — `gpu-llm down` destroys it.")
    else:
        print(f"Instance {iid} is still rented at ${dph:.3f}/hr:")
        print("  `gpu-llm down`   destroys it")
        print("  `gpu-llm tunnel` retries the tunnel (host/port already recorded)")
    return rc


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
    if st and st.get("instance_id"):
        print(f"instance: {st['instance_id']} (${st.get('dph', 0):.3f}/hr)")
    if not st or not pid_alive(st.get("pid"), st.get("image")):
        print("tunnel: down")
        return 1
    print(f"tunnel: up (pid {st['pid']}) 127.0.0.1:{st.get('local_port')} -> {st.get('host')}:{REMOTE_PORT}")
    key = resolve_llm_api_key(load_config())
    code, body = http_get(f"http://127.0.0.1:{st['local_port']}/v1/models", timeout=3.0,
                          headers={"Authorization": f"Bearer {key}"})
    if code == 200:
        try:
            ids = [m["id"] for m in json.loads(body).get("data", [])]
        except (ValueError, KeyError, TypeError):
            ids = []
        print(f"models: {', '.join(ids) or '(none)'}")
    elif code == 401:
        print("models: unauthorized (check LLM_API_KEY)")
    else:
        print("models: unreachable")
    if st.get("model"):
        ctx = state_context(st)
        print(f"model: {st['model']}" + (f" (context {ctx})" if ctx else ""))
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

    u = sub.add_parser("up", help="rent a Vast GPU, wait for READY, open the tunnel")
    u.add_argument("--gpu", choices=sorted(vast_api.GPU_FILTERS), default="5090")
    u.add_argument("--model", default=None,
                   help="model id under profiles/<gpu>/ (default: that GPU's default; see profiles/README.md)")
    u.add_argument("--max-price", type=float, default=None, help="max $/hr (default per GPU)")
    u.add_argument("--yes", action="store_true", help="skip the prompt and rent the cheapest offer")
    u.add_argument("--list", type=int, default=5, help="how many offers to choose from (default 5)")
    u.add_argument("--offer", type=int, default=None, help="rent this specific offer id directly")
    u.add_argument("--timeout", type=int, default=900)
    u.add_argument("--interval", type=float, default=30.0)
    u.set_defaults(fn=cmd_up)

    d = sub.add_parser("down", help="close the tunnel")
    d.add_argument("--keep", action="store_true", help="kill the tunnel but keep the instance")
    d.set_defaults(fn=cmd_down)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
