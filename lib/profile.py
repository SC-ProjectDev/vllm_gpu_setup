"""Flat-YAML profile loader and vllm serve argv builder. Stdlib only."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _scalar(raw: str):
    v = raw.split("#", 1)[0].strip() if not raw.strip().startswith(("'", '"')) else raw.strip()
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.startswith(("'", '"')) and v.endswith(v[0]) and len(v) >= 2:
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_profile(path: str | Path) -> dict:
    """Parse a flat YAML file: `key: scalar` lines and `key:` followed by `- item` lines."""
    data: dict = {}
    current_list: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list is not None:
            data[current_list].append(stripped[2:].split("  #", 1)[0].strip())
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}: cannot parse line: {line!r}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        if rest.strip() == "":
            data[key] = []
            current_list = key
        else:
            data[key] = _scalar(rest)
            current_list = None
    return data


def profile_to_argv(profile: dict, overrides: dict[str, str]) -> list[str]:
    """Return the argv that follows `vllm serve` for this profile.

    `overrides` maps profile keys to string values; empty strings are ignored.
    Only `max_model_len` is honoured in M1.
    """
    max_len = profile["max_model_len"]
    if overrides.get("max_model_len"):
        max_len = int(overrides["max_model_len"])
    argv = [
        str(profile["model"]),
        "--served-model-name", str(profile["served_model_name"]),
        "--max-model-len", str(max_len),
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
        "--kv-cache-dtype", str(profile["kv_cache_dtype"]),
        "--tensor-parallel-size", str(profile["tensor_parallel_size"]),
    ]
    if profile.get("enforce_eager") is True:
        argv.append("--enforce-eager")
    for extra in profile.get("extra_args", []):
        argv.extend(extra.split())
    return argv


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: profile.py <profile.yaml>", file=sys.stderr)
        return 1
    profile = load_profile(argv[1])
    for item in profile_to_argv(profile, {"max_model_len": os.environ.get("MAX_MODEL_LEN", "")}):
        print(item)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
