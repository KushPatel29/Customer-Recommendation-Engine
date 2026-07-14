"""API contract tests — the service must serve exactly what the engine scored."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402

OUT = ROOT / "output"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # context manager triggers the lifespan loader
        yield c


@pytest.fixture(scope="module")
def known_customer():
    recs = pd.read_csv(OUT / "cross_sell_recommendations.csv")
    return recs.iloc[0]["customer_id"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["customers_scored"] > 0
    assert body["recommendations_available"] > 0


def test_list_customers_sorted_and_limited(client):
    r = client.get("/customers", params={"limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 5
    revenues = [c["total_revenue"] for c in rows]
    assert revenues == sorted(revenues, reverse=True)


def test_customer_profile_and_404(client, known_customer):
    ok = client.get(f"/customers/{known_customer}")
    assert ok.status_code == 200
    assert ok.json()["customer_id"] == known_customer
    missing = client.get("/customers/CUST-99999")
    assert missing.status_code == 404


def test_recommendations_match_engine_output(client, known_customer):
    r = client.get(f"/customers/{known_customer}/recommendations")
    assert r.status_code == 200
    body = r.json()
    recs = body["recommendations"]
    assert 1 <= len(recs) <= 10
    assert [x["rank"] for x in recs] == sorted(x["rank"] for x in recs)
    # served rows must equal the batch-scored file, verbatim
    disk = pd.read_csv(OUT / "cross_sell_recommendations.csv")
    disk_top = disk[disk["customer_id"] == known_customer].sort_values("rank")
    assert recs[0]["sku"] == disk_top.iloc[0]["sku"]
    assert recs[0]["score"] == pytest.approx(disk_top.iloc[0]["score"])


def test_recommendations_unknown_customer_404(client):
    r = client.get("/customers/CUST-99999/recommendations")
    assert r.status_code == 404


def test_cold_start_live_inference(client):
    sales = pd.read_csv(ROOT / "data" / "sales_lines.csv")
    region = sales["region"].iloc[0]
    r = client.get(f"/recommendations/cold-start/{region}")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 10
    scores = [x["blend_score"] for x in rows]
    assert scores == sorted(scores, reverse=True)
    bad = client.get("/recommendations/cold-start/Atlantis")
    assert bad.status_code == 404


def test_affinity_respects_min_lift(client):
    r = client.get("/affinity", params={"min_lift": 1.5, "limit": 10})
    assert r.status_code == 200
    assert all(row["lift"] >= 1.5 for row in r.json())
