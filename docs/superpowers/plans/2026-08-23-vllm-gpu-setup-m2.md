# vllm_gpu_setup M2 — `gpu-llm up` / auto-teardown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `gpu-llm up` rents a Vast.ai GPU via the REST API and takes it all the way to a serving model; `gpu-llm down` destroys the instance so the meter always stops.

**Architecture:** A new stdlib-only module `desktop/vast_api.py` holds the four Vast REST calls plus the pure offer-query/pick/format functions. `desktop/gpu_llm.py` gains `cmd_up` (search → confirm → rent → poll instance → reuse the existing tunnel/READY flow) and extends `cmd_down` to destroy the recorded instance. The onstart script moves to `vast/onstart.sh` so manual and API rentals share one source.

**Tech Stack:** Python 3.11+ stdlib only (`urllib`, `json`, `tomllib`), pytest with local fake HTTP servers, Git Bash for shell tests. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-23-vllm-gpu-setup-m2-design.md`

## Global Constraints

- `desktop/*.py` is **stdlib only** — no pip installs (spec §2).
- Windows-first: everything must work under Windows Python + Git Bash; tests run offline, no GPU, no live Vast API (spec §9).
- Image pin `vllm/vllm-openai:v0.27.1`, disk 60 GB, `runtype: "ssh"` (spec §6).
- API base URL `https://console.vast.ai`, overridable via `VAST_API_BASE` env for tests.
- API key: `VAST_API_KEY` env wins over `api_key` in `~/.gpu-llm/config.toml`; never committed (spec §2).
- All existing M1 tests must keep passing unchanged (spec §9).
- Run the suite with `python -m pytest` from the repo root (`F:\Coding\VirtualTrashcan47\llm\vllm_gpu_setup`).
- Commit after every task; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Single-source onstart script (`vast/onstart.sh`)

**Files:**
- Create: `vast/onstart.sh`
- Modify: `vast/template.md` (the "On-start script" section)
- Test: `tests/test_onstart.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `vast/onstart.sh` — a one-line bash script; Task 5's `cmd_up` reads this file's text verbatim and sends it as the `onstart` field when renting.

- [ ] **Step 1: Write the failing test**

Create `tests/test_onstart.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_onstart.py -v`
Expected: both FAIL (`FileNotFoundError` for onstart.sh; template has no pointer yet).

- [ ] **Step 3: Create `vast/onstart.sh`**

Exact content (this is the command currently inlined in template.md):

```bash
cd /root && git clone https://github.com/SC-ProjectDev/vllm_gpu_setup.git && cd vllm_gpu_setup && cp -n .env.example .env && nohup bash bootstrap.sh > /var/log/bootstrap.log 2>&1 &
```

- [ ] **Step 4: Update `vast/template.md`**

Replace the fenced command in the "On-start script" section with:

````markdown
## On-start script

Paste the contents of [`vast/onstart.sh`](onstart.sh) (one line). `gpu-llm up`
sends the same file automatically, so manual and API rentals cannot drift.

```
cd /root && git clone https://github.com/SC-ProjectDev/vllm_gpu_setup.git && cd vllm_gpu_setup && cp -n .env.example .env && nohup bash bootstrap.sh > /var/log/bootstrap.log 2>&1 &
```
````

(The command stays visible in the doc for copy-paste; the test only pins that the doc names `vast/onstart.sh` as the source of truth.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_onstart.py -v` → PASS. Then the full suite: `python -m pytest` → all pass.

- [ ] **Step 6: Commit**

```bash
git add vast/onstart.sh vast/template.md tests/test_onstart.py
git commit -m "feat: single-source onstart script in vast/onstart.sh"
```

---

### Task 2: `vast_api.py` pure functions (query build, pick, format)

**Files:**
- Create: `desktop/vast_api.py`
- Test: `tests/test_vast_api.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by Tasks 3–6):
  - `GPU_FILTERS: dict[str, dict]` — keys `"5090" | "a100-80" | "h100-80" | "h200"`, values `{"gpu_names": list[str], "max_price": float}`.
  - `build_offer_query(gpu: str, max_price: float | None) -> dict` — `None` omits the price filter (used for the over-cap listing).
  - `pick_offer(offers: list[dict]) -> dict | None` — cheapest by `dph_total`, `None` on empty.
  - `format_offer(offer: dict) -> str` — one-line human summary.
  - `class VastError(Exception)` with attribute `code: int | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vast_api.py`:

```python
import pytest

from desktop import vast_api
from desktop.vast_api import (
    GPU_FILTERS,
    VastError,
    build_offer_query,
    format_offer,
    pick_offer,
)


def test_gpu_filters_cover_all_profiles():
    assert set(GPU_FILTERS) == {"5090", "a100-80", "h100-80", "h200"}
    for v in GPU_FILTERS.values():
        assert v["gpu_names"] and v["max_price"] > 0


def test_build_offer_query_baked_in_filters():
    q = build_offer_query("5090", 1.0)
    assert q["gpu_name"] == {"in": ["RTX 5090"]}
    assert q["dph_total"] == {"lte": 1.0}
    assert q["reliability"] == {"gte": 0.98}
    assert q["inet_down"] == {"gte": 500}
    assert q["cuda_max_good"] == {"gte": 12.8}
    assert q["num_gpus"] == {"eq": 1}
    assert q["rentable"] == {"eq": True}
    assert q["type"] == "ondemand"
    assert q["order"] == [["dph_total", "asc"]]


def test_build_offer_query_none_cap_omits_price():
    q = build_offer_query("h200", None)
    assert "dph_total" not in q


def test_build_offer_query_unknown_gpu_raises():
    with pytest.raises(KeyError):
        build_offer_query("3090", 1.0)


def test_pick_offer_cheapest_and_empty():
    offers = [{"id": 1, "dph_total": 0.9}, {"id": 2, "dph_total": 0.5}]
    assert pick_offer(offers)["id"] == 2
    assert pick_offer([]) is None


def test_format_offer_line():
    line = format_offer({"gpu_name": "RTX 5090", "dph_total": 0.592,
                         "inet_down": 812.0, "reliability2": 0.992,
                         "geolocation": "US, TX"})
    assert "RTX 5090" in line
    assert "$0.592/hr" in line
    assert "812 Mbps" in line
    assert "99.2%" in line
    assert "US, TX" in line


def test_format_offer_tolerates_missing_fields():
    line = format_offer({"gpu_name": "H200", "dph_total": 2.5})
    assert "H200" in line and "$2.500/hr" in line


def test_vast_error_carries_code():
    e = VastError("boom", code=404)
    assert e.code == 404
    assert VastError("plain").code is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vast_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'desktop.vast_api'`.

- [ ] **Step 3: Write `desktop/vast_api.py` (pure part)**

```python
"""Vast.ai REST client for gpu_llm. Stdlib only. Spec: docs/superpowers/specs/2026-08-23-vllm-gpu-setup-m2-design.md"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://console.vast.ai"

# --gpu id -> Vast gpu_name filter values + default price cap ($/hr).
# gpu_names must match Vast's canonical spellings (verify against live
# search results during the acceptance run; adjust here if they differ).
GPU_FILTERS = {
    "5090": {"gpu_names": ["RTX 5090"], "max_price": 1.00},
    "a100-80": {"gpu_names": ["A100 SXM4", "A100 PCIE"], "max_price": 1.60},
    "h100-80": {"gpu_names": ["H100 SXM", "H100 PCIE", "H100 NVL"], "max_price": 2.50},
    "h200": {"gpu_names": ["H200", "H200 NVL"], "max_price": 3.50},
}


class VastError(Exception):
    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


def build_offer_query(gpu: str, max_price: float | None) -> dict:
    q = {
        "limit": 20,
        "type": "ondemand",
        "rentable": {"eq": True},
        "verified": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_name": {"in": list(GPU_FILTERS[gpu]["gpu_names"])},
        "reliability": {"gte": 0.98},
        "inet_down": {"gte": 500},
        "cuda_max_good": {"gte": 12.8},
        "order": [["dph_total", "asc"]],
    }
    if max_price is not None:
        q["dph_total"] = {"lte": max_price}
    return q


def pick_offer(offers: list[dict]) -> dict | None:
    if not offers:
        return None
    return min(offers, key=lambda o: o.get("dph_total", float("inf")))


def format_offer(offer: dict) -> str:
    parts = [str(offer.get("gpu_name", "?")), f"${offer.get('dph_total', 0):.3f}/hr"]
    if offer.get("inet_down") is not None:
        parts.append(f"{offer['inet_down']:.0f} Mbps")
    rel = offer.get("reliability2", offer.get("reliability"))
    if rel is not None:
        parts.append(f"{rel * 100:.1f}%")
    if offer.get("geolocation"):
        parts.append(str(offer["geolocation"]))
    return " · ".join(parts)
```

(The `json`/`os`/`urllib` imports are used by Task 3; leaving them now avoids an import-shuffle commit.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vast_api.py -v` → PASS. Full suite: `python -m pytest` → all pass.

- [ ] **Step 5: Commit**

```bash
git add desktop/vast_api.py tests/test_vast_api.py
git commit -m "feat: vast_api offer query/pick/format pure functions"
```

---

### Task 3: `vast_api.py` HTTP layer against a fake Vast server

**Files:**
- Modify: `desktop/vast_api.py` (append)
- Test: `tests/test_vast_api.py` (append)

**Interfaces:**
- Consumes: `VastError`, `DEFAULT_API_BASE` from Task 2.
- Produces (used by Tasks 4–6):
  - `api_request(method: str, path: str, api_key: str, body: dict | None = None, timeout: float = 30.0) -> dict` — raises `VastError` (with `.code` for HTTP errors). Base URL from `VAST_API_BASE` env, else `DEFAULT_API_BASE`.
  - `search_offers(api_key: str, query: dict) -> list[dict]`
  - `rent_offer(api_key: str, offer_id: int, image: str, disk: int, onstart: str) -> int` — returns the new instance id (`new_contract`).
  - `list_instances(api_key: str) -> list[dict]`
  - `destroy_instance(api_key: str, instance_id: int) -> bool` — `True` destroyed, `False` already gone (HTTP 404); raises `VastError` otherwise.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vast_api.py` (fake server pattern mirrors `tests/test_gpu_llm.py`):

```python
import json as _json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeVast:
    """Local stand-in for console.vast.ai; scripts responses per (method, path prefix)."""

    def __init__(self, monkeypatch):
        self.requests = []          # (method, path, body_dict_or_None, auth_header)
        self.responses = {}         # (method, path_prefix) -> (status, payload_dict)
        outer = self

        class H(BaseHTTPRequestHandler):
            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = _json.loads(self.rfile.read(length)) if length else None
                outer.requests.append((method, self.path, body,
                                       self.headers.get("Authorization")))
                for (m, prefix), (status, payload) in outer.responses.items():
                    if m == method and self.path.startswith(prefix):
                        data = _json.dumps(payload).encode()
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                self.send_response(404)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_GET(self): self._handle("GET")
            def do_POST(self): self._handle("POST")
            def do_PUT(self): self._handle("PUT")
            def do_DELETE(self): self._handle("DELETE")
            def log_message(self, *a): pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        monkeypatch.setenv("VAST_API_BASE", f"http://127.0.0.1:{self.srv.server_port}")

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def fake_vast(monkeypatch):
    fv = FakeVast(monkeypatch)
    yield fv
    fv.stop()


def test_search_offers_posts_query_with_bearer_auth(fake_vast):
    fake_vast.responses[("POST", "/api/v0/bundles")] = (200, {"offers": [{"id": 7, "dph_total": 0.5}]})
    offers = vast_api.search_offers("KEY", build_offer_query("5090", 1.0))
    assert offers == [{"id": 7, "dph_total": 0.5}]
    method, path, body, auth = fake_vast.requests[0]
    assert (method, path) == ("POST", "/api/v0/bundles")
    assert auth == "Bearer KEY"
    assert body["gpu_name"] == {"in": ["RTX 5090"]}


def test_rent_offer_puts_ask_and_returns_contract(fake_vast):
    fake_vast.responses[("PUT", "/api/v0/asks/7")] = (200, {"success": True, "new_contract": 4242})
    iid = vast_api.rent_offer("KEY", 7, "vllm/vllm-openai:v0.27.1", 60, "echo hi")
    assert iid == 4242
    _, _, body, _ = fake_vast.requests[0]
    assert body == {"image": "vllm/vllm-openai:v0.27.1", "disk": 60,
                    "env": "", "onstart": "echo hi", "runtype": "ssh"}


def test_rent_offer_without_contract_raises(fake_vast):
    fake_vast.responses[("PUT", "/api/v0/asks/7")] = (200, {"success": False})
    with pytest.raises(VastError):
        vast_api.rent_offer("KEY", 7, "img", 60, "x")


def test_list_instances(fake_vast):
    fake_vast.responses[("GET", "/api/v1/instances")] = (200, {"instances": [{"id": 4242, "actual_status": "running"}]})
    assert vast_api.list_instances("KEY")[0]["id"] == 4242


def test_destroy_instance_true_false_and_error(fake_vast):
    fake_vast.responses[("DELETE", "/api/v0/instances/1")] = (200, {"success": True})
    fake_vast.responses[("DELETE", "/api/v0/instances/2")] = (404, {"error": "no such"})
    fake_vast.responses[("DELETE", "/api/v0/instances/3")] = (429, {"error": "rate"})
    assert vast_api.destroy_instance("KEY", 1) is True
    assert vast_api.destroy_instance("KEY", 2) is False
    with pytest.raises(VastError) as exc:
        vast_api.destroy_instance("KEY", 3)
    assert exc.value.code == 429


def test_api_request_http_error_carries_code_and_body(fake_vast):
    fake_vast.responses[("POST", "/api/v0/bundles")] = (401, {"error": "bad key"})
    with pytest.raises(VastError) as exc:
        vast_api.search_offers("KEY", build_offer_query("5090", 1.0))
    assert exc.value.code == 401
    assert "bad key" in str(exc.value)


def test_api_request_connection_refused_raises_vasterror(monkeypatch):
    monkeypatch.setenv("VAST_API_BASE", "http://127.0.0.1:1")  # nothing listens
    with pytest.raises(VastError) as exc:
        vast_api.list_instances("KEY")
    assert exc.value.code is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vast_api.py -v`
Expected: new tests FAIL with `AttributeError` (`search_offers` etc. not defined); Task 2 tests still PASS.

- [ ] **Step 3: Append the HTTP layer to `desktop/vast_api.py`**

```python
def _api_base() -> str:
    return os.environ.get("VAST_API_BASE", DEFAULT_API_BASE)


def api_request(method: str, path: str, api_key: str, body: dict | None = None,
                timeout: float = 30.0) -> dict:
    url = f"{_api_base()}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        e.close()
        raise VastError(f"{method} {path} -> HTTP {e.code}: {detail}", code=e.code) from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise VastError(f"{method} {path} failed: {e}") from e


def search_offers(api_key: str, query: dict) -> list[dict]:
    return api_request("POST", "/api/v0/bundles", api_key, body=query).get("offers", [])


def rent_offer(api_key: str, offer_id: int, image: str, disk: int, onstart: str) -> int:
    body = {"image": image, "disk": disk, "env": "", "onstart": onstart, "runtype": "ssh"}
    resp = api_request("PUT", f"/api/v0/asks/{offer_id}", api_key, body=body)
    iid = resp.get("new_contract")
    if not iid:
        raise VastError(f"rent succeeded without new_contract: {resp}")
    return int(iid)


def list_instances(api_key: str) -> list[dict]:
    return api_request("GET", "/api/v1/instances", api_key).get("instances", [])


def destroy_instance(api_key: str, instance_id: int) -> bool:
    try:
        api_request("DELETE", f"/api/v0/instances/{instance_id}", api_key)
        return True
    except VastError as e:
        if e.code == 404:
            return False
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vast_api.py -v` → PASS. Full suite: `python -m pytest` → all pass.

- [ ] **Step 5: Commit**

```bash
git add desktop/vast_api.py tests/test_vast_api.py
git commit -m "feat: vast_api HTTP layer (search/rent/list/destroy)"
```

---

### Task 4: `gpu_llm.py` plumbing — api key, price, state merge, tunnel extraction

**Files:**
- Modify: `desktop/gpu_llm.py`
- Test: `tests/test_gpu_llm.py` (append)

**Interfaces:**
- Consumes: `vast_api.GPU_FILTERS` (Task 2).
- Produces (used by Tasks 5–6):
  - `resolve_api_key(cfg: dict) -> str | None` — `VAST_API_KEY` env, else `cfg["api_key"]`, else `None`.
  - `resolve_max_price(cfg: dict, gpu: str, cli_value: float | None) -> float` — CLI > `cfg["max_price"][gpu]` > `GPU_FILTERS[gpu]["max_price"]`.
  - `merge_state(fields: dict) -> None` — `save_state({**(load_state() or {}), **fields})`.
  - `open_tunnel_and_wait(cfg: dict, timeout: int, interval: float) -> int` — the body of today's `cmd_tunnel` after its guard (spawn ssh, merge state, `wait_ready`, epilogue/teardown). `cmd_tunnel` becomes guard + this call. Uses `merge_state` so `instance_id`/`gpu`/`dph` written by `cmd_up` survive.
  - `resolve_conn` falls back to `state.json`'s `host`/`ssh_port` when config and CLI have neither (so `gpu-llm tunnel` retries an API rental without editing config.toml).
  - `cmd_status` prints `instance: <id> ($<dph>/hr)` when state has `instance_id`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_llm.py`:

```python
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
```

Note for the implementer: `_fake_ssh` and the `home` fixture already exist in this file — reuse them, do not redefine. The status test's tunnel pid must look alive, hence `os.getpid()` + image `"python"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v -k "resolve_api or max_price or merge_state or falls_back or beats_state or instance_line"`
Expected: FAIL with `AttributeError` (functions missing) / assertion on missing instance line.

- [ ] **Step 3: Implement in `desktop/gpu_llm.py`**

Add near the config helpers (after `load_config`); `from desktop import vast_api` fails when run as a script from repo root, so import defensively at the top:

```python
try:
    from desktop import vast_api
except ImportError:  # run as `python desktop/gpu_llm.py` from repo root
    import vast_api
```

```python
def resolve_api_key(cfg: dict) -> str | None:
    return os.environ.get("VAST_API_KEY") or cfg.get("api_key")


def resolve_max_price(cfg: dict, gpu: str, cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    from_cfg = (cfg.get("max_price") or {}).get(gpu)
    if from_cfg is not None:
        return float(from_cfg)
    return vast_api.GPU_FILTERS[gpu]["max_price"]


def merge_state(fields: dict) -> None:
    save_state({**(load_state() or {}), **fields})
```

In `resolve_conn`, before the `raise SystemExit`, add the state fallback:

```python
    if not cfg.get("host") or not cfg.get("ssh_port"):
        st = load_state()
        if st and st.get("host") and st.get("ssh_port"):
            cfg.setdefault("host", st["host"])
            cfg.setdefault("ssh_port", st["ssh_port"])
    if not cfg.get("host") or not cfg.get("ssh_port"):
        raise SystemExit(...)  # unchanged
```

Extract `open_tunnel_and_wait(cfg, timeout, interval) -> int` from `cmd_tunnel`: everything from `cmd = tunnel_cmd(...)` through the final `return 0`, with two changes — `save_state({...})` becomes `merge_state({...})`, and the `wait_ready` failure path calls `merge_state({"pid": 0})` instead of `clear_state()` when state holds an `instance_id` (a rented instance must stay recorded; plain-tunnel behavior with no instance_id keeps `clear_state()`):

```python
def open_tunnel_and_wait(cfg: dict, timeout: int, interval: float) -> int:
    local_port = int(cfg["local_port"])
    cmd = tunnel_cmd(cfg, local_port)
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    ssh_log_f = ssh_log_path().open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ssh_log_f, creationflags=creation)
    finally:
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
    print(f"LLM_BASE_URL=http://127.0.0.1:{local_port}")
    print("LLM_MODEL=qwen")
    return 0


def cmd_tunnel(args) -> int:
    cfg = resolve_conn(args)
    old = load_state()
    if old and pid_alive(old["pid"], old.get("image")):
        print(f"Tunnel already running (pid {old['pid']}); run `down` first.")
        return 1
    return open_tunnel_and_wait(cfg, args.timeout, args.interval)
```

In `cmd_status`, after the `tunnel:` line, add:

```python
    if st.get("instance_id"):
        print(f"instance: {st['instance_id']} (${st.get('dph', 0):.3f}/hr)")
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest` → all pass (existing tunnel tests exercise the extraction; if any fail, fix `open_tunnel_and_wait` — do not modify the M1 tests).

- [ ] **Step 5: Commit**

```bash
git add desktop/gpu_llm.py tests/test_gpu_llm.py
git commit -m "feat: api-key/price/state plumbing; extract open_tunnel_and_wait"
```

---

### Task 5: `cmd_up` — search, confirm, rent, poll, tunnel

**Files:**
- Modify: `desktop/gpu_llm.py`
- Test: `tests/test_gpu_llm.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 2–4; `vast/onstart.sh` (Task 1).
- Produces:
  - `IMAGE = "vllm/vllm-openai:v0.27.1"`, `DISK_GB = 60`, `INSTANCE_WAIT_SECS = 600` module constants.
  - `confirm(prompt: str) -> bool` — `input()` wrapper returning True only for `y`/`yes` (case-insensitive).
  - `wait_instance_running(api_key, instance_id, timeout, interval=10.0, progress=print) -> tuple[dict | None, str]` — returns `(instance, "running")`, `(instance, "exited"|"offline"|"unknown")`, or `(None, "timeout")`. `VastError` during a poll is reported via `progress` and retried until timeout.
  - `cmd_up(args) -> int` and the `up` subparser (`--gpu`, `--max-price`, `--yes`, `--timeout`, `--interval`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_llm.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v -k "up_ or wait_instance or confirm_"`
Expected: FAIL — `up` is not a known subcommand (argparse SystemExit) / missing attributes.

- [ ] **Step 3: Implement `cmd_up` in `desktop/gpu_llm.py`**

Module constants next to `DEFAULT_LOCAL_PORT`:

```python
IMAGE = "vllm/vllm-openai:v0.27.1"
DISK_GB = 60
INSTANCE_WAIT_SECS = 600
REPO_ROOT = Path(__file__).resolve().parent.parent
MANAGE_KEYS_URL = "https://cloud.vast.ai/manage-keys/"
CONSOLE_INSTANCES_URL = "https://cloud.vast.ai/instances/"
```

```python
def confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def wait_instance_running(api_key: str, instance_id: int, timeout: float,
                          interval: float = 10.0, progress=print) -> tuple[dict | None, str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    max_price = resolve_max_price(cfg, args.gpu, args.max_price)
    offers = vast_api.search_offers(api_key, vast_api.build_offer_query(args.gpu, max_price))
    if not offers:
        print(f"No {args.gpu} offers under ${max_price:.2f}/hr. Cheapest above the cap:")
        over = vast_api.search_offers(api_key, vast_api.build_offer_query(args.gpu, None))
        for o in sorted(over, key=lambda o: o.get("dph_total", float("inf")))[:3]:
            print(f"  {vast_api.format_offer(o)}")
        return 1
    offer = vast_api.pick_offer(offers)
    print(vast_api.format_offer(offer))
    if not args.yes and not confirm("rent? [y/N] "):
        print("Nothing rented.")
        return 0
    onstart = (REPO_ROOT / "vast" / "onstart.sh").read_text(encoding="utf-8")
    iid = vast_api.rent_offer(api_key, offer["id"], IMAGE, DISK_GB, onstart)
    save_state({"pid": 0, "instance_id": iid, "gpu": args.gpu, "dph": offer["dph_total"]})
    print(f"Rented instance {iid} at ${offer['dph_total']:.3f}/hr. Waiting for SSH info...")
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
    dph = offer["dph_total"]
    if rc == 0:
        print(f"instance {iid} at ${dph:.3f}/hr — `gpu-llm down` destroys it.")
    else:
        print(f"Instance {iid} is still rented at ${dph:.3f}/hr:")
        print("  `gpu-llm down`   destroys it")
        print("  `gpu-llm tunnel` retries the tunnel (host/port already recorded)")
    return rc
```

In `build_parser()` add before the `down` subparser:

```python
    u = sub.add_parser("up", help="rent a Vast GPU, wait for READY, open the tunnel")
    u.add_argument("--gpu", choices=sorted(vast_api.GPU_FILTERS), default="5090")
    u.add_argument("--max-price", type=float, default=None, help="max $/hr (default per GPU)")
    u.add_argument("--yes", action="store_true", help="skip the rent confirmation")
    u.add_argument("--timeout", type=int, default=900)
    u.add_argument("--interval", type=float, default=30.0)
    u.set_defaults(fn=cmd_up)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest` → all pass.

- [ ] **Step 5: Commit**

```bash
git add desktop/gpu_llm.py tests/test_gpu_llm.py
git commit -m "feat: gpu-llm up — search, confirm, rent, poll, tunnel"
```

---

### Task 6: `cmd_down` destroys the instance; docs + acceptance checklist

**Files:**
- Modify: `desktop/gpu_llm.py` (`cmd_down`, `down` subparser)
- Modify: `README.md` (Daily loop, script table intro, new M2 acceptance section)
- Test: `tests/test_gpu_llm.py` (append)

**Interfaces:**
- Consumes: `vast_api.destroy_instance`, `resolve_api_key`, `_destroy_and_report` (Tasks 3–5).
- Produces: `down [--keep]` final behavior; M2 acceptance checklist in README.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gpu_llm.py`:

```python
def test_down_destroys_recorded_instance(home, monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", "K")
    destroyed = []
    monkeypatch.setattr(gpu_llm.vast_api, "destroy_instance",
                        lambda k, iid: destroyed.append(iid) or True)
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22,
                       "local_port": 8000, "instance_id": 4242, "dph": 0.592})
    rc = gpu_llm.main(["down"])
    out = capsys.readouterr().out
    assert rc == 0
    assert destroyed == [4242]
    assert "destroyed" in out
    assert gpu_llm.load_state() is None


def test_down_keep_skips_destroy_and_keeps_state(home, monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", "K")
    monkeypatch.setattr(gpu_llm.vast_api, "destroy_instance",
                        lambda k, iid: pytest.fail("destroy called despite --keep"))
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22,
                       "local_port": 8000, "instance_id": 4242})
    rc = gpu_llm.main(["down", "--keep"])
    assert rc == 0
    assert gpu_llm.load_state()["instance_id"] == 4242
    assert "kept" in capsys.readouterr().out.lower()


def test_down_destroy_failure_warns_and_keeps_state(home, monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", "K")
    def boom(k, iid):
        raise gpu_llm.vast_api.VastError("rate limited", code=429)
    monkeypatch.setattr(gpu_llm.vast_api, "destroy_instance", boom)
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22,
                       "local_port": 8000, "instance_id": 4242})
    rc = gpu_llm.main(["down"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "WARNING" in out and "billing" in out
    assert gpu_llm.load_state()["instance_id"] == 4242


def test_down_instance_already_gone_clears_state(home, monkeypatch, capsys):
    monkeypatch.setenv("VAST_API_KEY", "K")
    monkeypatch.setattr(gpu_llm.vast_api, "destroy_instance", lambda k, iid: False)
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22,
                       "local_port": 8000, "instance_id": 4242})
    rc = gpu_llm.main(["down"])
    assert rc == 0
    assert "already gone" in capsys.readouterr().out
    assert gpu_llm.load_state() is None


def test_down_manual_rental_keeps_m1_reminder(home, capsys):
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22, "local_port": 8000})
    rc = gpu_llm.main(["down"])
    assert rc == 0
    assert "still billing" in capsys.readouterr().out
    assert gpu_llm.load_state() is None


def test_down_with_instance_but_no_api_key_warns(home, monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    gpu_llm.save_state({"pid": 999999999, "host": "h", "ssh_port": 22,
                       "local_port": 8000, "instance_id": 4242})
    rc = gpu_llm.main(["down"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "4242" in out and "key" in out.lower()
    assert gpu_llm.load_state()["instance_id"] == 4242
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gpu_llm.py -v -k "down_"`
Expected: the six new tests FAIL (`--keep` unknown; no destroy call; M1 `clear_state` wipes instance state). The three existing M1 `down` tests must still pass at the end of this task **unchanged** — the manual-rental path keeps its exact behavior.

- [ ] **Step 3: Rewrite `cmd_down`**

```python
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
```

In `build_parser()`, the `down` subparser gains:

```python
    d.add_argument("--keep", action="store_true", help="kill the tunnel but keep the instance")
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest` → all pass, including the untouched M1 down tests.

- [ ] **Step 5: Update `README.md`**

- "Daily loop" becomes: 1. `python desktop/gpu_llm.py up` (mention `--gpu`, confirm prompt, `--yes`); 2. use `llm-cli`; 3. `python desktop/gpu_llm.py down` — destroys the instance; `--keep` to leave it running. Keep the manual-rental loop as a fallback subsection pointing at `vast/template.md`.
- "One-time desktop config" gains `api_key = "..."` (or `VAST_API_KEY` env) with the manage-keys URL.
- Add after the M1 acceptance section:

````markdown
## Milestone 2 acceptance run (paid, ~30 min)

- [ ] `python desktop/gpu_llm.py up` — offer shown with $/hr, confirm, READY,
      `llm-cli health` + `chat` work.
- [ ] `python desktop/gpu_llm.py status` shows `instance: <id> ($/hr)`.
- [ ] `python desktop/gpu_llm.py down` — instance destroyed; verify gone in
      the console.
- [ ] `python desktop/gpu_llm.py up --yes --max-price 0.05` exits 1 with the
      over-cap listing; rents nothing.
- [ ] Verify `GPU_FILTERS` gpu_name spellings against the live offers seen
      above; fix the table if Vast spells any differently.
- [ ] Record: time from `up` to READY, total cost.
````

- [ ] **Step 6: Full suite one last time, then commit**

Run: `python -m pytest` → all pass.

```bash
git add desktop/gpu_llm.py tests/test_gpu_llm.py README.md
git commit -m "feat: gpu-llm down destroys the instance; M2 docs + acceptance"
```

---

## Self-review notes (already applied)

- Spec coverage: §2 decisions → Tasks 1–6; §3 API surface → Task 3; §4 CLI → Tasks 5–6; §5 config/state (incl. crash-safe rent recording and stale-state cleanup) → Tasks 4–5; §6 up flow → Task 5; §7 down flow → Task 6; §8 error table → Tasks 5–6 tests; §9 testing → every task; §10 acceptance → Task 6 README.
- Type consistency: `wait_instance_running` returns `(dict | None, str)` everywhere; `destroy_instance` bool contract used by both `_destroy_and_report` and `cmd_down`; `merge_state` used by `open_tunnel_and_wait`, `cmd_up`, `cmd_down --keep`.
- The M1 behavior contract is pinned by leaving existing tests untouched (Tasks 4 and 6 both call this out).
