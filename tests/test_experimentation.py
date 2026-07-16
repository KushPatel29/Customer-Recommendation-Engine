"""Online serving behaviors: A/B assignment, suppression, telemetry, stats."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import ab_variant, app  # noqa: E402
from experiments.ab_analysis import analyze, simulate  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def all_customers():
    return pd.read_csv(ROOT / "output" / "customer_analytics.csv")["customer_id"].tolist()


# ---------------- A/B assignment ----------------

def test_assignment_is_deterministic_and_split(client, all_customers):
    variants = {cid: ab_variant(cid) for cid in all_customers}
    assert variants == {cid: ab_variant(cid) for cid in all_customers}, "must be sticky"
    share_b = sum(v == "B" for v in variants.values()) / len(variants)
    assert 0.3 < share_b < 0.7, f"hash split badly skewed: B={share_b:.0%}"


def test_variant_b_serves_the_challenger(client, all_customers):
    b_cust = next(c for c in all_customers if ab_variant(c) == "B")
    r = client.get(f"/customers/{b_cust}/recommendations")
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_variant"] == "B"
    assert "two-stage" in body["model"]


# ---------------- serve-time suppressor ----------------

def test_purchase_suppresses_sku_immediately(client, all_customers):
    cust = next(c for c in all_customers
                if client.get(f"/customers/{c}/recommendations").json()["recommendations"])
    before = client.get(f"/customers/{cust}/recommendations").json()
    top_sku = before["recommendations"][0]["sku"]

    r = client.post("/events", json={"customer_id": cust, "sku": top_sku,
                                     "event_type": "purchase"})
    assert r.status_code == 202

    after = client.get(f"/customers/{cust}/recommendations").json()
    served = [rec["sku"] for rec in after["recommendations"]]
    assert top_sku not in served, "a just-purchased SKU must vanish from the next call"
    assert top_sku in after["suppressed_skus"]
    # ranks re-close over the gap
    assert [rec["rank"] for rec in after["recommendations"]] == list(
        range(1, len(served) + 1))


def test_telemetry_is_written(client, all_customers):
    telemetry = ROOT / "output" / "telemetry_events.jsonl"
    assert telemetry.exists()
    text = telemetry.read_text(encoding="utf-8")
    assert '"event_type": "impression"' in text
    assert '"event_type": "purchase"' in text


# ---------------- the statistics ----------------

def test_chi_square_detects_planted_difference():
    result = analyze(simulate(5000, cvr_a=0.08, cvr_b=0.11, seed=42))
    assert result["decision"] == "promote_B"
    assert result["p_value"] < 0.05


def test_chi_square_does_not_reward_noise():
    result = analyze(simulate(5000, cvr_a=0.08, cvr_b=0.08, seed=42))
    assert result["decision"] == "keep_A_keep_testing"


def test_insufficient_data_refuses_to_decide():
    result = analyze({"A": {"impressions": 5, "purchases": 1},
                      "B": {"impressions": 5, "purchases": 4}})
    assert result["decision"] == "insufficient_data"
