import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.conftest import ROOT, run_bash

SMOKE = ROOT / "scripts" / "smoke.sh"
HEALTH = ROOT / "scripts" / "health.sh"


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def serve_json(payload: dict, seen: dict | None = None):
    """Fake vLLM: POST -> `payload`; GET /v1/models -> a model list; GET anything -> ok.
    When `seen` is given, records the POST's Authorization header and JSON body and
    every GET as (path, Authorization)."""
    class H(BaseHTTPRequestHandler):
        def _reply(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            if seen is not None:
                seen["auth"] = self.headers.get("Authorization")
                seen["body"] = json.loads(raw)
            self._reply(json.dumps(payload).encode())

        def do_GET(self):
            if seen is not None:
                seen.setdefault("gets", []).append((self.path, self.headers.get("Authorization")))
            self._reply(b'{"data":[{"id":"local"}]}' if self.path == "/v1/models" else b"ok")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def completion(reasoning, content, tokens=42):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}],
        "usage": {"completion_tokens": tokens},
    }


def test_smoke_passes_and_prints_toks():
    srv, base = serve_json(completion("thinking...", "4"))
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tok/s:" in r.stdout
    assert "reasoning: ok" in r.stdout


def test_smoke_fails_without_reasoning():
    srv, base = serve_json(completion("", "4"))
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base})
    finally:
        _stop(srv)
    assert r.returncode == 1
    assert "reasoning_content empty" in r.stdout + r.stderr


def test_smoke_handles_zero_elapsed():
    srv, base = serve_json(completion("thinking...", "4"))
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base, "SMOKE_START": "5", "SMOKE_END": "5"})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tok/s: n/a" in r.stdout


def test_smoke_sends_bearer_and_uses_local_model():
    seen = {}
    srv, base = serve_json(completion("thinking...", "4"), seen)
    try:
        r = run_bash(SMOKE, env={"BASE_URL": base, "VLLM_API_KEY": "sekrit"})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["model"] == "local"


def test_smoke_without_key_sends_no_auth_header():
    seen = {}
    srv, base = serve_json(completion("thinking...", "4"), seen)
    try:
        run_bash(SMOKE, env={"BASE_URL": base, "VLLM_API_KEY": ""})
    finally:
        _stop(srv)
    assert seen["auth"] is None


def test_health_sends_bearer_only_to_v1():
    seen = {}
    srv, base = serve_json({}, seen)
    try:
        r = run_bash(HEALTH, env={"BASE_URL": base, "VLLM_API_KEY": "sekrit"})
    finally:
        _stop(srv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("/health", None) in seen["gets"]
    assert ("/v1/models", "Bearer sekrit") in seen["gets"]
