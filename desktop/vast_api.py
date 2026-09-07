"""Vast.ai REST client for gpu_llm. Stdlib only. Spec: docs/superpowers/specs/2026-08-23-vllm-gpu-setup-m2-design.md"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://console.vast.ai"

# --gpu id -> Vast gpu_name filter values, minimum VRAM (MiB, Vast's `gpu_ram`),
# and default price cap ($/hr). gpu_names match Vast's canonical spellings
# (verified live 2026-08-23 / 2026-09-07). min_gpu_ram matters because Vast
# names the 40 GB and 80 GB A100 identically ("A100 SXM4") -- the thresholds
# mirror lib/detect_gpu.sh's VRAM fallback so a rented card always maps to
# the profile dir we asked for.
GPU_FILTERS = {
    "5090": {"gpu_names": ["RTX 5090"], "min_gpu_ram": 30000, "max_price": 1.00},
    "a100-80": {"gpu_names": ["A100 SXM4", "A100 PCIE"], "min_gpu_ram": 80000, "max_price": 1.60},
    "h100-80": {"gpu_names": ["H100 SXM", "H100 PCIE", "H100 NVL"], "min_gpu_ram": 80000, "max_price": 2.50},
    "h200": {"gpu_names": ["H200", "H200 NVL"], "min_gpu_ram": 130000, "max_price": 3.50},
}


class VastError(Exception):
    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


def build_offer_query(gpu: str, max_price: float | None, filters: dict | None = None) -> dict:
    """filters: optional [filters] table from config.toml — inet_down/reliability
    raise the default floors; country (list of codes) adds a geolocation filter."""
    f = filters or {}
    q = {
        "limit": 20,
        "type": "ondemand",
        "rentable": {"eq": True},
        "verified": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_name": {"in": list(GPU_FILTERS[gpu]["gpu_names"])},
        "gpu_ram": {"gte": GPU_FILTERS[gpu]["min_gpu_ram"]},
        "reliability": {"gte": float(f["reliability"]) if "reliability" in f else 0.98},
        "inet_down": {"gte": float(f["inet_down"]) if "inet_down" in f else 500},
        "cuda_max_good": {"gte": 12.8},
        "order": [["dph_total", "asc"]],
    }
    if f.get("country"):
        q["geolocation"] = {"in": [str(c) for c in f["country"]]}
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
