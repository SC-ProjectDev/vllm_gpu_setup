#!/usr/bin/env python3
"""Local HTTP shim between IDE clients and the ssh tunnel to vLLM. Stdlib only.

Why: JetBrains' Ktor HTTP client sends `Connection: Upgrade, HTTP2-Settings` and
`Upgrade: h2c` on every request. vLLM's uvicorn does not upgrade, and on such a
request it hands the app an *empty* body, so chat completions fail with
`400 body: Field required` while `GET /v1/models` works. Stripping those three
headers is the whole fix; everything else is forwarded byte-for-byte, streaming
included, so llm-cli / curl / Prometheus scrapes are unaffected.

Layout (see gpu_llm.py): ssh tunnel listens on raw_port(local_port); this shim
listens on local_port and forwards to the tunnel.

Usage: shim.py --listen 8000 --target 18000
"""
from __future__ import annotations

import argparse
import http.client
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RAW_PORT_OFFSET = 10000
# Hop-by-hop / framing headers we regenerate ourselves, plus the h2c upgrade trio.
STRIP_REQUEST = {"host", "connection", "upgrade", "http2-settings", "transfer-encoding",
                 "content-length", "keep-alive", "proxy-connection"}
STRIP_RESPONSE = {"connection", "transfer-encoding", "content-length", "keep-alive"}


def raw_port(local_port: int) -> int:
    """Port the ssh tunnel binds when the shim owns `local_port`."""
    p = int(local_port) + RAW_PORT_OFFSET
    if not 1 <= p <= 65535:
        raise ValueError(f"local_port {local_port} + {RAW_PORT_OFFSET} is out of range")
    return p


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
        chunks = []
        while True:
            line = handler.rfile.readline().strip()
            size = int(line.split(b";")[0] or b"0", 16)
            if size == 0:
                handler.rfile.readline()
                return b"".join(chunks)
            chunks.append(handler.rfile.read(size))
            handler.rfile.readline()
    n = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(n) if n else b""


def make_handler(target_host: str, target_port: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self):
            body = _read_body(self)
            fwd = {k: v for k, v in self.headers.items() if k.lower() not in STRIP_REQUEST}
            fwd["Content-Length"] = str(len(body))
            fwd["Connection"] = "close"
            try:
                conn = http.client.HTTPConnection(target_host, target_port, timeout=600)
                conn.request(self.command, self.path, body=body, headers=fwd)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                msg = f'{{"error":"shim: upstream 127.0.0.1:{target_port} unreachable ({e})"}}'.encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(msg)
                return
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in STRIP_RESPONSE:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read1(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, socket.timeout):
                pass
            finally:
                conn.close()
                self.close_connection = True

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = _forward

        def log_message(self, *args):
            pass

    return Handler


def make_server(listen_port: int, target_port: int, target_host: str = "127.0.0.1",
                listen_host: str = "127.0.0.1") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((listen_host, listen_port), make_handler(target_host, target_port))
    srv.daemon_threads = True
    return srv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="strip h2c upgrade headers and forward to the vLLM tunnel")
    p.add_argument("--listen", type=int, required=True, help="port clients use (the banner URL)")
    p.add_argument("--target", type=int, required=True, help="raw ssh tunnel port")
    args = p.parse_args(argv)
    srv = make_server(args.listen, args.target)
    print(f"shim: 127.0.0.1:{args.listen} -> 127.0.0.1:{args.target}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
