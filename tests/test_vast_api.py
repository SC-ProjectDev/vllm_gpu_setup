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
