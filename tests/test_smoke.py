import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.conftest import ROOT, run_bash

SMOKE = ROOT / "scripts" / "smoke.sh"


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def serve_json(payload: dict):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
