"""Independent failure scenarios for degraded serving and experiment units."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as api
from experiments.ab_analysis import analyze, load_counts, report


@pytest.fixture
def fallback_client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "TELEMETRY", tmp_path / "events.jsonl")
    with TestClient(api.app) as client:
        api.store["recs"] = pd.DataFrame()
        api.store["recs_b"] = None
        api.store["customers"] = pd.DataFrame([
            {"customer_id": "CUST-001", "customer_name": "Synthetic buyer", "region": "West"},
        ])
        api.store["sales"] = pd.DataFrame([
            {"customer_id": "CUST-001", "region": "West", "sku": "SKU-001",
             "protein": "Beef", "description": "Owned", "quantity_lb": 1},
            {"customer_id": "CUST-002", "region": "West", "sku": "SKU-002",
             "protein": "Beef", "description": "Candidate", "quantity_lb": 10},
            {"customer_id": "CUST-002", "region": "West", "sku": "SKU-003",
             "protein": "Beef", "description": "Next", "quantity_lb": 5},
        ])
        yield client


def test_purchase_remains_suppressed_when_personalized_artifacts_are_missing(fallback_client):
    url = "/customers/CUST-001/recommendations?limit=1"
    assert fallback_client.get(url).json()["recommendations"][0]["sku"] == "SKU-002"
    assert fallback_client.post("/events", json={
        "customer_id": "CUST-001", "sku": "SKU-002", "event_type": "purchase",
    }).status_code == 202
    after = fallback_client.get(url).json()
    assert after["serving_mode"] == "degraded_fallback"
    assert after["suppressed_skus"] == ["SKU-002"]
    assert [(r["rank"], r["sku"]) for r in after["recommendations"]] == [(1, "SKU-003")]
    events = [json.loads(line) for line in api.TELEMETRY.read_text().splitlines()]
    impressions = [e for e in events if e["event_type"] == "impression"]
    assert len(impressions) == 2
    assert all(e["serving_mode"] == "degraded_fallback" for e in impressions)
    assert load_counts(api.TELEMETRY) == {
        "A": {"impressions": 0, "purchases": 0}, "B": {"impressions": 0, "purchases": 0},
    }


@pytest.mark.parametrize("sku", ["", "SKU-999", "NOT-A-PRODUCT", "=formula"])
def test_unknown_product_feedback_cannot_change_session_or_telemetry(fallback_client, sku):
    response = fallback_client.post("/events", json={
        "customer_id": "CUST-001", "sku": sku, "event_type": "purchase",
    })
    assert response.status_code == 404
    assert api.store["session_purchases"] == {}
    assert not api.TELEMETRY.exists()


def test_replayed_requests_cannot_manufacture_independent_experiment_units(tmp_path):
    path = tmp_path / "events.jsonl"
    events = []
    for variant, purchases in (("A", 50), ("B", 200)):
        customer = "CUST-" + variant
        events += [{"event_type": "impression", "variant": variant,
                    "customer_id": customer, "skus": ["SKU-001"]}] * 1000
        events += [{"event_type": "purchase", "variant": variant,
                    "customer_id": customer, "sku": "SKU-001"}] * purchases
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    counts = load_counts(path)
    assert counts == {"A": {"impressions": 1, "purchases": 1},
                      "B": {"impressions": 1, "purchases": 1}}
    assert analyze(counts)["decision"] == "insufficient_data"


def test_only_preceding_personalized_exposure_can_support_a_conversion(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {"event_type": "purchase", "variant": "A", "customer_id": "C1", "sku": "P1"},
        {"event_type": "impression", "variant": "A", "customer_id": "C1", "skus": ["P1"]},
        {"event_type": "purchase", "variant": "A", "customer_id": "C1", "sku": "P2"},
        {"event_type": "impression", "variant": "A", "customer_id": "C1", "skus": ["P1"],
         "serving_mode": "degraded_fallback"},
        {"event_type": "purchase", "variant": "A", "customer_id": "C1", "sku": "P1"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    assert load_counts(path)["A"] == {"impressions": 1, "purchases": 0}


def test_a_customer_crossing_variants_blocks_the_readout(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps({"event_type": "impression", "variant": v,
                                         "customer_id": "C1", "skus": ["P1"]})
                               for v in ("A", "B")), encoding="utf-8")
    with pytest.raises(ValueError, match="changed experiment variants"):
        load_counts(path)


@pytest.mark.parametrize("purchases", [0, 100])
def test_identical_constant_outcomes_remain_finite_and_cannot_promote(purchases):
    result = report({v: {"impressions": 100, "purchases": purchases} for v in ("A", "B")}, "synthetic boundary")
    assert result["p_value"] == 1
    assert result["lift"] == 0
    assert result["decision"] == "keep_A_keep_testing"
    assert result["promotion_authorized"] is False
    json.dumps(result, allow_nan=False)


def test_zero_control_conversion_does_not_produce_json_infinity():
    result = report({"A": {"impressions": 1000, "purchases": 0},
                     "B": {"impressions": 1000, "purchases": 20}}, "synthetic boundary")
    assert result["lift"] is None
    assert result["decision"] == "review_B"
    assert result["promotion_authorized"] is False
    json.dumps(result, allow_nan=False)


def test_assignment_imbalance_blocks_outcome_interpretation():
    result = analyze({"A": {"impressions": 100, "purchases": 1},
                      "B": {"impressions": 1000, "purchases": 500}})
    assert result["decision"] == "assignment_quality_review"
    assert result["promotion_authorized"] is False


@pytest.mark.parametrize("arm", [
    {"impressions": 100, "purchases": 101}, {"impressions": 100, "purchases": -1},
    {"impressions": float("nan"), "purchases": 0}, {"impressions": 100.5, "purchases": 1},
    {"impressions": True, "purchases": 0}, {"impressions": 2**54, "purchases": 1},
])
def test_invalid_aggregates_never_receive_a_favorable_readout(arm):
    with pytest.raises(ValueError):
        analyze({"A": arm, "B": {"impressions": 100, "purchases": 10}})
