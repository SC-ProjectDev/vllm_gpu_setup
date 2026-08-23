import json as _json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
