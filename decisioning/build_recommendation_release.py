"""Build the evidence pack for a controlled recommendation-service release.

This module deliberately separates three questions:

1. Does the model recover held-out items for warm customers?
2. Is the served slate useful, varied, eligible and operationally safe?
3. Does it improve customer or commercial outcomes online?

Only the first two can be answered by this synthetic offline project. The third
is expressed as a pre-registered experiment contract, never as claimed lift.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

QUALITY_FILE = OUT / "recommendation_quality_scorecard.csv"
SLICE_FILE = OUT / "recommendation_slice_evaluation.csv"
ELIGIBILITY_FILE = OUT / "recommendation_eligibility_audit.csv"
EXPERIMENT_FILE = OUT / "recommendation_experiment_plan.json"
SLO_FILE = OUT / "serving_slo.json"
PACKET_FILE = OUT / "recommendation_release_packet.json"
MEMO_FILE = OUT / "recommendation_release_memo.md"


def _split_set(value: object) -> set[str]:
    if pd.isna(value) or not str(value):
        return set()
    return {item for item in str(value).split(";") if item}


def _mean_intra_list_diversity(recs: pd.DataFrame) -> float:
    """Mean share of recommendation pairs from different protein families."""
    values: list[float] = []
    for _, group in recs.groupby("customer_id"):
        proteins = group.sort_values("rank")["protein"].astype(str).tolist()
        pairs = list(combinations(proteins, 2))
        if pairs:
            values.append(sum(a != b for a, b in pairs) / len(pairs))
    return float(np.mean(values)) if values else 0.0


def _mean_protein_calibration_error(sales: pd.DataFrame, recs: pd.DataFrame) -> float:
    """Average total-variation distance between bought and recommended proteins."""
    errors: list[float] = []
    proteins = sorted(set(sales["protein"]) | set(recs["protein"]))
    for customer_id, served in recs.groupby("customer_id"):
        history = sales[sales["customer_id"] == customer_id]
        if history.empty:
            continue
        historical = history.groupby("protein")["quantity_lb"].sum()
        historical = historical / historical.sum()
        recommended = served.groupby("protein").size()
        recommended = recommended / recommended.sum()
        error = 0.5 * sum(
            abs(float(historical.get(p, 0.0)) - float(recommended.get(p, 0.0)))
            for p in proteins
        )
        errors.append(error)
    return float(np.mean(errors)) if errors else 0.0


def build_eligibility_audit(
    sales: pd.DataFrame, recs: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    owned = sales.groupby("customer_id")["sku"].apply(set).to_dict()
    catalog_skus = set(catalog["sku"])
    audit = recs.copy()
    audit["catalog_present"] = audit["sku"].isin(catalog_skus)
    audit["not_previously_purchased"] = [
        sku not in owned.get(customer_id, set())
        for customer_id, sku in zip(audit["customer_id"], audit["sku"], strict=True)
    ]
    audit["reason_present"] = audit["because_similar_to"].fillna("").str.strip().ne("")
    audit["opportunity_positive"] = audit["est_revenue_opportunity"].gt(0)
    checks = [
        "catalog_present",
        "not_previously_purchased",
        "reason_present",
        "opportunity_positive",
    ]
    audit["eligible"] = audit[checks].all(axis=1)
    audit["eligibility_decision"] = np.where(audit["eligible"], "SERVE", "SUPPRESS")
    return audit[
        [
            "customer_id",
            "rank",
            "sku",
            "description",
            "protein",
            "because_similar_to",
            "est_revenue_opportunity",
            *checks,
            "eligible",
            "eligibility_decision",
        ]
    ]


def build_quality_scorecard(
    sales: pd.DataFrame,
    recs: pd.DataFrame,
    evaluation: pd.DataFrame,
    catalog: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    item_share = sales.groupby("sku")["quantity_lb"].sum()
    item_share = item_share / item_share.sum()
    novelty = recs["sku"].map(lambda sku: -math.log2(max(float(item_share.get(sku, 0)), 1e-12)))
    hidden = float(evaluation["hidden"].sum())
    cf_recall = float(evaluation["cf_hits"].sum() / hidden)
    pop_recall = float(evaluation["pop_hits"].sum() / hidden)
    rows = [
        ("relevance", "Recall@10", cf_recall, ">= 0.80", "PASS" if cf_recall >= 0.80 else "REVIEW", "warm-customer basket holdout"),
        ("relevance", "Lift over popularity", cf_recall / pop_recall, "> 1.00x", "PASS" if cf_recall > pop_recall else "BLOCK", "same holdout and denominator"),
        ("reach", "Catalog coverage", recs["sku"].nunique() / catalog["sku"].nunique(), ">= 0.80", "PASS" if recs["sku"].nunique() / catalog["sku"].nunique() >= 0.80 else "REVIEW", "share of catalog served at least once"),
        ("discovery", "Mean novelty (bits)", float(novelty.mean()), ">= 4.00", "PASS" if novelty.mean() >= 4 else "REVIEW", "self-information under purchase-volume popularity"),
        ("choice", "Intra-list diversity", _mean_intra_list_diversity(recs), ">= 0.55", "PASS" if _mean_intra_list_diversity(recs) >= 0.55 else "REVIEW", "share of slate pairs from different protein families"),
        ("trust", "Protein calibration error", _mean_protein_calibration_error(sales, recs), "<= 0.45", "PASS" if _mean_protein_calibration_error(sales, recs) <= 0.45 else "REVIEW", "lower is closer to the customer's historical mix"),
        ("safety", "Eligibility pass rate", float(eligibility["eligible"].mean()), "= 1.00", "PASS" if eligibility["eligible"].all() else "BLOCK", "catalog, ownership, reason and opportunity controls"),
    ]
    return pd.DataFrame(
        rows,
        columns=["objective", "metric", "value", "release_threshold", "status", "measurement_basis"],
    )


def _slice_row(
    dimension: str, slice_name: str, group: pd.DataFrame, minimum_n: int = 20
) -> dict[str, Any]:
    hidden = int(group["hidden"].sum())
    recall = float(group["cf_hits"].sum() / hidden) if hidden else 0.0
    return {
        "dimension": dimension,
        "slice": str(slice_name),
        "customers": int(group["customer_id"].nunique()),
        "hidden_items": hidden,
        "recall_at_10": round(recall, 6),
        "evidence_status": "REPORT" if hidden >= minimum_n else "DIRECTIONAL_ONLY",
    }


def build_slice_evaluation(
    sales: pd.DataFrame,
    evaluation: pd.DataFrame,
    customers: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    joined = evaluation.merge(
        customers[["customer_id", "region", "rfm_segment", "distinct_skus"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    joined["history_density"] = pd.qcut(
        joined["distinct_skus"].rank(method="first"),
        q=3,
        labels=["sparse warm", "medium warm", "dense warm"],
    )
    rows: list[dict[str, Any]] = []
    for dimension in ("history_density", "region", "rfm_segment"):
        for name, group in joined.groupby(dimension, observed=True):
            rows.append(_slice_row(dimension, str(name), group))

    popularity = sales.groupby("sku")["quantity_lb"].sum().rank(pct=True, method="average")
    popularity_band = pd.cut(
        popularity,
        bins=[0, 0.33, 0.67, 1],
        labels=["tail", "mid", "head"],
        include_lowest=True,
    ).astype(str)
    protein = catalog.set_index("sku")["protein"]
    exploded: list[dict[str, Any]] = []
    for row in evaluation.itertuples(index=False):
        recommended = _split_set(row.cf_recs)
        for sku in _split_set(row.hidden_skus):
            exploded.append(
                {
                    "customer_id": row.customer_id,
                    "sku": sku,
                    "hit": int(sku in recommended),
                    "popularity_band": popularity_band.get(sku, "unknown"),
                    "protein": protein.get(sku, "unknown"),
                }
            )
    product = pd.DataFrame(exploded)
    for dimension in ("popularity_band", "protein"):
        for name, group in product.groupby(dimension, observed=True):
            rows.append(
                {
                    "dimension": dimension,
                    "slice": str(name),
                    "customers": int(group["customer_id"].nunique()),
                    "hidden_items": int(len(group)),
                    "recall_at_10": round(float(group["hit"].mean()), 6),
                    "evidence_status": "REPORT" if len(group) >= 20 else "DIRECTIONAL_ONLY",
                }
            )
    return pd.DataFrame(rows).sort_values(["dimension", "slice"]).reset_index(drop=True)


def build_experiment_plan() -> dict[str, Any]:
    return {
        "experiment_id": "REC-PILOT-2026-001",
        "decision": "Does the challenger improve incremental purchased opportunity without harming trust?",
        "unit_of_randomization": "customer_id",
        "assignment": "deterministic 50/50 hash; sticky across sessions",
        "primary_metric": "purchase conversion per eligible recommendation impression",
        "secondary_metrics": [
            "realized gross margin per impression",
            "catalog coverage",
            "new-category adoption rate",
        ],
        "guardrails": [
            "unavailable-item impression rate = 0",
            "already-owned-item impression rate = 0",
            "opt-out and complaint rate non-inferior",
            "p95 recommendation response <= 150 ms",
        ],
        "quality_checks": {
            "sample_ratio_mismatch": "chi-square p >= 0.01 before outcome readout",
            "minimum_exposure": "at least 1,000 eligible impressions per variant and powered analysis",
            "novelty_effect": "report new-category adoption separately from repeat-category purchases",
            "multiple_looks": "one pre-registered final read; no peeking-based promotion",
        },
        "promotion_rule": "Positive primary lift with p < 0.05, all guardrails pass, and no material slice harm.",
        "business_boundary": "Offline recall authorizes a controlled pilot; only randomized online evidence can support a commercial-lift claim.",
        "owner": "Product Analytics Lead",
        "approvers": ["Sales Operations Owner", "Commercial Finance Partner"],
    }


def serving_slo_contract() -> dict[str, Any]:
    return {
        "service": "customer-recommendation-api",
        "scope": "single-process portfolio deployment with precomputed champion scores",
        "targets_ms": {"p50": 50, "p95": 150, "p99": 300},
        "timeout_ms": 500,
        "availability_target": 0.995,
        "degraded_mode": "regional-popularity safe default, exclude owned SKUs, preserve reason code",
        "fail_closed_conditions": [
            "customer identity is unknown",
            "catalog eligibility cannot be established",
            "all candidate items are owned or suppressed",
        ],
    }


def benchmark_serving(iterations: int = 60) -> dict[str, Any]:
    """Measure in-process API latency; result is environment-labelled evidence."""
    from fastapi.testclient import TestClient

    from api.main import app

    timings: list[float] = []
    with TestClient(app) as client:
        customer_ids = [row["customer_id"] for row in client.get("/customers", params={"limit": 20}).json()]
        for index in range(iterations):
            started = time.perf_counter_ns()
            response = client.get(f"/customers/{customer_ids[index % len(customer_ids)]}/recommendations")
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
            if response.status_code != 200:
                raise RuntimeError(f"benchmark request failed: {response.status_code}")
    percentiles = np.percentile(timings, [50, 95, 99])
    targets = serving_slo_contract()["targets_ms"]
    measured = {f"p{p}_ms": round(float(value), 3) for p, value in zip((50, 95, 99), percentiles, strict=True)}
    measured.update(
        {
            "requests": iterations,
            "environment": "FastAPI TestClient, local in-process, warm artifacts",
            "all_targets_met": all(measured[f"p{p}_ms"] <= targets[f"p{p}"] for p in (50, 95, 99)),
        }
    )
    return measured


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_release_packet(run_benchmark: bool = True) -> dict[str, Any]:
    sales = pd.read_csv(DATA / "sales_lines.csv")
    catalog = pd.read_csv(DATA / "catalog.csv")
    customers = pd.read_csv(OUT / "customer_analytics.csv")
    recs = pd.read_csv(OUT / "cross_sell_recommendations.csv")
    evaluation = pd.read_csv(OUT / "holdout_evaluation.csv")

    required = {"hidden_skus", "cf_recs", "cf_hits", "pop_hits", "hidden"}
    missing = sorted(required - set(evaluation.columns))
    if missing:
        raise ValueError(f"holdout evaluation is missing release fields: {missing}")

    eligibility = build_eligibility_audit(sales, recs, catalog)
    quality = build_quality_scorecard(sales, recs, evaluation, catalog, eligibility)
    slices = build_slice_evaluation(sales, evaluation, customers, catalog)
    experiment = build_experiment_plan()
    slo = serving_slo_contract()
    benchmark = benchmark_serving() if run_benchmark else {"status": "not_run"}

    QUALITY_FILE.parent.mkdir(exist_ok=True)
    quality.to_csv(QUALITY_FILE, index=False)
    slices.to_csv(SLICE_FILE, index=False)
    eligibility.to_csv(ELIGIBILITY_FILE, index=False)
    EXPERIMENT_FILE.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
    SLO_FILE.write_text(json.dumps({**slo, "latest_benchmark": benchmark}, indent=2) + "\n", encoding="utf-8")

    blockers = quality.loc[quality["status"] == "BLOCK", "metric"].tolist()
    reportable_slices = slices[slices["evidence_status"] == "REPORT"]
    min_slice_recall = float(reportable_slices["recall_at_10"].min())
    release_status = "BLOCK" if blockers or not benchmark.get("all_targets_met", False) else "CONDITIONALLY APPROVE PILOT"
    packet: dict[str, Any] = {
        "release_id": "REC-REL-2026-001",
        "release_status": release_status,
        "approved_scope": "synthetic-data rep-assist pilot; human review required before outreach",
        "model": "item-neighborhood collaborative filtering champion",
        "offline_recall_at_10": round(float(evaluation["cf_hits"].sum() / evaluation["hidden"].sum()), 6),
        "popularity_recall_at_10": round(float(evaluation["pop_hits"].sum() / evaluation["hidden"].sum()), 6),
        "catalog_coverage": round(float(recs["sku"].nunique() / catalog["sku"].nunique()), 6),
        "eligibility_pass_rate": round(float(eligibility["eligible"].mean()), 6),
        "minimum_reportable_slice_recall": round(min_slice_recall, 6),
        "latency_evidence": benchmark,
        "blockers": blockers,
        "decision_owner": "Product Analytics Lead",
        "approval_authorities": ["Sales Operations Owner", "Commercial Finance Partner"],
        "required_next_action": "Run REC-PILOT-2026-001 before making any sales-lift claim.",
        "evidence_boundary": "All data and traffic in this repository are synthetic. Offline metrics measure warm-customer basket recovery, not incremental sales.",
        "artifacts": [
            QUALITY_FILE.name,
            SLICE_FILE.name,
            ELIGIBILITY_FILE.name,
            EXPERIMENT_FILE.name,
            SLO_FILE.name,
        ],
    }
    packet["sha256"] = _digest(packet)
    PACKET_FILE.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    MEMO_FILE.write_text(_memo(packet, quality, slices), encoding="utf-8")
    return packet


def _memo(packet: dict[str, Any], quality: pd.DataFrame, slices: pd.DataFrame) -> str:
    score_lines = "\n".join(
        f"- **{row.metric}:** {row.value:.3f} — {row.status} ({row.measurement_basis})"
        for row in quality.itertuples(index=False)
    )
    weakest = slices[slices["evidence_status"] == "REPORT"].nsmallest(3, "recall_at_10")
    slice_lines = "\n".join(
        f"- **{row.dimension} / {row.slice}:** recall@10 {row.recall_at_10:.1%} across {row.hidden_items} hidden items"
        for row in weakest.itertuples(index=False)
    )
    return f"""# Recommendation release memo — {packet['release_id']}

## Decision

**{packet['release_status']}** for a synthetic-data rep-assist pilot. The collaborative-filtering champion remains simpler and stronger on the defined warm-customer holdout. This authorizes controlled evaluation, not automated outreach and not a revenue-lift claim.

## Objective hierarchy and release evidence

{score_lines}

## Weakest reportable slices

{slice_lines}

These slices are diagnostic, not proof of fairness or causal impact. New customers are excluded from the warm-customer holdout and use the separately labelled regional-popularity fallback.

## Operating controls

- Every served item must exist in the catalog, be unowned, carry a recommendation reason and retain a positive indicative opportunity.
- The service target is p95 <= 150 ms, with p50/p95/p99 recorded in `serving_slo.json`.
- Degraded mode serves an explicitly labelled regional-popularity default and still excludes owned products.
- Every recommendation is human-reviewed before a sales representative acts.

## Experiment before promotion

Run `REC-PILOT-2026-001` with sticky customer-level assignment, a sample-ratio-mismatch check, one pre-registered final read and commercial, trust and latency guardrails. Promotion requires positive purchase-conversion lift at p < 0.05, every guardrail passing and no material slice harm.

## Evidence boundary

{packet['evidence_boundary']}

Packet SHA-256: `{packet['sha256']}`
"""


def main() -> None:
    packet = build_release_packet(run_benchmark=True)
    print(f"{packet['release_status']}: {PACKET_FILE}")
    print(f"sha256: {packet['sha256']}")


if __name__ == "__main__":
    main()
