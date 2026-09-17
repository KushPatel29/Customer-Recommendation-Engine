"""Release evidence for product quality, serving safety and online learning."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT))

from api.main import app, store  # noqa: E402


def test_release_packet_is_conditionally_approved_not_called_realized_lift():
    packet = json.loads((OUT / "recommendation_release_packet.json").read_text())
    assert packet["release_status"] == "CONDITIONALLY APPROVE PILOT"
    assert "synthetic" in packet["evidence_boundary"].lower()
    assert "not incremental sales" in packet["evidence_boundary"].lower()
    assert "experiment" in packet["required_next_action"].lower() or "pilot" in packet["required_next_action"].lower()


def test_release_packet_digest_recomputes_from_canonical_payload():
    packet = json.loads((OUT / "recommendation_release_packet.json").read_text())
    claimed = packet.pop("sha256")
    canonical = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == claimed


def test_objective_hierarchy_covers_relevance_discovery_choice_trust_and_safety():
    scorecard = pd.read_csv(OUT / "recommendation_quality_scorecard.csv")
    assert {"relevance", "reach", "discovery", "choice", "trust", "safety"} <= set(scorecard["objective"])
    assert not (scorecard["status"] == "BLOCK").any()
    assert scorecard["measurement_basis"].str.strip().ne("").all()


def test_slice_report_covers_customer_history_geography_and_product_exposure():
    slices = pd.read_csv(OUT / "recommendation_slice_evaluation.csv")
    assert {"history_density", "region", "rfm_segment", "popularity_band", "protein"} <= set(slices["dimension"])
    assert slices["recall_at_10"].between(0, 1).all()
    assert set(slices["evidence_status"]) <= {"REPORT", "DIRECTIONAL_ONLY"}


def test_every_served_item_passes_all_eligibility_controls():
    audit = pd.read_csv(OUT / "recommendation_eligibility_audit.csv")
    controls = ["catalog_present", "not_previously_purchased", "reason_present", "opportunity_positive"]
    assert audit[controls].all().all()
    assert audit["eligible"].all()
    assert set(audit["eligibility_decision"]) == {"SERVE"}


def test_experiment_contract_has_srm_guardrails_and_a_causal_boundary():
    plan = json.loads((OUT / "recommendation_experiment_plan.json").read_text())
    assert "sample_ratio_mismatch" in plan["quality_checks"]
    assert len(plan["guardrails"]) >= 4
    assert "randomized" in plan["business_boundary"].lower()
    assert "p < 0.05" in plan["promotion_rule"]


def test_serving_benchmark_reports_ordered_percentiles_inside_the_contract():
    slo = json.loads((OUT / "serving_slo.json").read_text())
    measured = slo["latest_benchmark"]
    assert measured["p50_ms"] <= measured["p95_ms"] <= measured["p99_ms"]
    assert measured["all_targets_met"] is True
    assert measured["requests"] >= 50


def test_health_exposes_primary_artifact_and_fallback_readiness():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["champion_artifact_ready"] is True
        assert body["degraded_fallback_ready"] is True


def test_missing_personalized_artifact_uses_labelled_safe_default():
    with TestClient(app) as client:
        customer = client.get("/customers", params={"limit": 1}).json()[0]["customer_id"]
        original_a, original_b = store["recs"], store["recs_b"]
        try:
            store["recs"] = pd.DataFrame()
            store["recs_b"] = None
            body = client.get(f"/customers/{customer}/recommendations").json()
        finally:
            store["recs"], store["recs_b"] = original_a, original_b
        owned = set(store["sales"].loc[store["sales"]["customer_id"] == customer, "sku"])
        assert body["serving_mode"] == "degraded_fallback"
        assert "regional-popularity" in body["model"]
        assert all(row["sku"] not in owned for row in body["recommendations"])
        assert all(row["est_revenue_opportunity"] == 0 for row in body["recommendations"])


def test_unknown_customer_still_fails_closed_instead_of_falling_back():
    with TestClient(app) as client:
        response = client.get("/customers/CUST-DOES-NOT-EXIST/recommendations")
    assert response.status_code == 404


def test_decision_room_source_keeps_release_experiment_and_reliability_separate():
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    for label in ("Release decision", "Rep assist", "Experiment & reliability"):
        assert label in source
    assert "Only the pre-registered customer-level experiment" in source


def test_release_memo_names_human_review_and_no_sales_lift_claim():
    memo = (OUT / "recommendation_release_memo.md").read_text(encoding="utf-8")
    assert "human-reviewed" in memo
    assert "not a revenue-lift claim" in memo
    assert "REC-PILOT-2026-001" in memo
