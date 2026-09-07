"""desktop/shim.py: local HTTP shim that strips h2c upgrade headers before vLLM.

JetBrains' Ktor client sends `Connection: Upgrade, HTTP2-Settings` + `Upgrade: h2c`
on every request; vLLM's uvicorn then delivers an empty body to the app and the
IDE gets `400 body: Field required` (seen live 2026-09-07). The shim sits on the
port clients use and forwards to the raw ssh tunnel with those headers removed.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from desktop import shim

H2C = {"Connection": "Upgrade, HTTP2-Settings", "Upgrade": "h2c",
       "HTTP2-Settings": "AAEAAEAAAAIAAAAAAAMAAAAAAAQBAAAAAAUAAEAAAAYABgAA"}


def _upstream(seen: list):
    """Fake vLLM: records (method, path, headers, body); POST streams 3 SSE chunks,
    GET /v1/models needs a bearer (401 otherwise), GET /health -> ok."""
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _record(self, body: bytes):
            seen.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}, body))

        def do_GET(self):
            self._record(b"")
            if self.path == "/v1/models" and self.headers.get("Authorization") != "Bearer k":
                self.send_response(401); self.send_header("Content-Length", "2"); self.end_headers()
                self.wfile.write(b"{}"); return
            body = b'{"data":[{"id":"local"}]}' if self.path == "/v1/models" else b"ok"
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            self._record(body)
            if not body:
                err = b'{"error":"body missing"}'
                self.send_response(400); self.send_header("Content-Length", str(len(err))); self.end_headers()
                self.wfile.write(err); return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                chunk = f"data: {i}\n\n".encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"); self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def stack():
    seen = []
    up = _upstream(seen)
    sh = shim.make_server(0, up.server_port)
    threading.Thread(target=sh.serve_forever, daemon=True).start()
    yield seen, f"http://127.0.0.1:{sh.server_port}"
    sh.shutdown(); sh.server_close()
    up.shutdown(); up.server_close()


def _post(url, body: dict, headers: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    return urllib.request.urlopen(req, timeout=10)


def test_strips_h2c_upgrade_headers_and_forwards_body(stack):
    seen, base = stack
    resp = _post(f"{base}/v1/chat/completions", {"model": "local", "stream": True}, {**H2C, "Authorization": "Bearer k"})
    assert resp.status == 200
    method, path, hdrs, body = seen[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert json.loads(body) == {"model": "local", "stream": True}
    assert "upgrade" not in hdrs and "http2-settings" not in hdrs
    assert hdrs.get("connection", "").lower() != "upgrade, http2-settings"
    assert hdrs["authorization"] == "Bearer k"
    assert hdrs["content-type"] == "application/json"


def test_streams_sse_chunks_incrementally(stack):
    seen, base = stack
    resp = _post(f"{base}/v1/chat/completions", {"model": "local"}, H2C)
    assert resp.headers.get("Content-Type") == "text/event-stream"
    t0 = time.monotonic()
    first = resp.readline()
    t_first = time.monotonic() - t0
    rest = resp.read()
    assert first == b"data: 0\n"
    assert rest.count(b"data: ") == 2
    assert t_first < 0.5   # first chunk arrived before the upstream finished


def test_get_passthrough_and_status_codes(stack):
    seen, base = stack
    with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
        assert r.status == 200 and r.read() == b"ok"
    req = urllib.request.Request(f"{base}/v1/models", headers={"Authorization": "Bearer k", **H2C})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.loads(r.read())["data"][0]["id"] == "local"
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(f"{base}/v1/models", timeout=5)
    assert ei.value.code == 401


def test_upstream_down_returns_502(stack):
    seen, base = stack
    dead = shim.make_server(0, 1)   # nothing listens on port 1
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(f"http://127.0.0.1:{dead.server_port}/health", timeout=5)
        assert ei.value.code == 502
    finally:
        dead.shutdown(); dead.server_close()


def test_raw_port_helper():
    assert shim.raw_port(8000) == 18000
    with pytest.raises(ValueError):
        shim.raw_port(60000)
