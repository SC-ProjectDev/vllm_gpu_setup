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
