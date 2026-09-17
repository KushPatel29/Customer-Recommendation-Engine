"""
Recommendation service — FastAPI wrapper around the engine outputs.

Serving pattern (deliberate, and worth saying in an interview):
  * Cross-sell scores are computed in BATCH by engine/recommend.py and served
    from the precomputed table — the standard production shape for
    collaborative filtering, where scoring all customers is cheap offline and
    latency matters online.
  * Cold-start recommendations are computed LIVE per request by calling the
    engine's region-blend function — cheap enough to run inline, and it shows
    the same code path working as a real-time inference service.

Run locally:
    uvicorn api.main:app --reload
Docs (OpenAPI/Swagger) are auto-generated at http://127.0.0.1:8000/docs

The Dockerfile at the repo root packages generator -> engine -> API into a
self-contained image: `docker build -t rec-api . && docker run -p 8000:8000 rec-api`.
"""

import json
import sys
import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT))
from engine.recommend import recommend_for_new_customer  # noqa: E402

# In-memory store loaded once at startup (all tables are small; a warehouse
# or feature store would replace this layer at real scale).
store: dict[str, pd.DataFrame] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    rec_file = OUT / "cross_sell_recommendations.csv"
    store["recs"] = pd.read_csv(rec_file) if rec_file.exists() else pd.DataFrame()
    store["customers"] = pd.read_csv(OUT / "customer_analytics.csv")
    store["affinity"] = pd.read_csv(OUT / "sku_affinity.csv")
    store["actions"] = pd.read_csv(OUT / "action_list.csv")
    store["sales"] = pd.read_csv(DATA / "sales_lines.csv", parse_dates=["order_date"])
    # variant B (the A/B challenger): two-stage re-ranked recs, if built
    ts_file = OUT / "two_stage_recommendations.csv"
    store["recs_b"] = pd.read_csv(ts_file) if ts_file.exists() else None
    # online state: session purchases per customer (the serve-time suppressor)
    store["session_purchases"] = {}
    yield
    store.clear()


app = FastAPI(
    title="Customer Cross-Sell Recommendation API",
    description="Serves collaborative-filtering cross-sell recommendations, "
                "customer analytics, and basket-affinity insights for a B2B "
                "protein distributor. Batch-scored, holdout-evaluated "
                "(CF recall@10 = 84.9% vs 75.4% popularity baseline).",
    version="2.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- schemas

class CustomerSummary(BaseModel):
    customer_id: str
    customer_name: str
    region: str
    rep: str
    rfm_segment: str
    churn_risk: str
    total_revenue: float
    clv_12m_runrate: float


class Recommendation(BaseModel):
    rank: int
    sku: str
    protein: str
    description: str
    score: float
    est_revenue_opportunity: float = Field(description="Indicative $ if adopted at neighbor-implied volume")
    because_similar_to: str = Field(description="Most similar customer who buys this SKU")


class CustomerRecommendations(BaseModel):
    customer_id: str
    customer_name: str
    model: str = "item-based collaborative filtering (batch-scored)"
    experiment_variant: str = "A"
    serving_mode: str = "personalized_batch"
    fallback_reason: str | None = None
    suppressed_skus: list[str] = []
    recommendations: list[Recommendation]


class TrackedEvent(BaseModel):
    customer_id: str
    sku: str
    event_type: str = Field(pattern="^(click|purchase)$")


TELEMETRY = OUT / "telemetry_events.jsonl"


def ab_variant(customer_id: str) -> str:
    """Deterministic, sticky assignment: same customer, same variant, every
    request, no state — a hash split, the standard first tool of online
    experimentation. B serves the two-stage challenger when it's built."""
    if store.get("recs_b") is None:
        return "A"
    return "B" if zlib.crc32(customer_id.encode()) % 2 else "A"


def log_telemetry(event: dict) -> None:
    event["ts"] = datetime.now(UTC).isoformat()
    with open(TELEMETRY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


class ColdStartRecommendation(BaseModel):
    rank: int
    sku: str
    protein: str
    description: str
    blend_score: float


class AffinityPair(BaseModel):
    sku_a: str
    description_a: str
    sku_b: str
    description_b: str
    orders_together: int
    confidence_a_to_b: float
    lift: float


class Health(BaseModel):
    status: str
    customers_scored: int
    recommendations_available: int
    affinity_pairs: int
    champion_artifact_ready: bool
    degraded_fallback_ready: bool


# ---------------------------------------------------------------- endpoints

@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    return Health(
        status="ok",
        customers_scored=int(store["customers"]["customer_id"].nunique()),
        recommendations_available=len(store["recs"]),
        affinity_pairs=len(store["affinity"]),
        champion_artifact_ready=not store["recs"].empty,
        degraded_fallback_ready=not store["sales"].empty,
    )


@app.get("/customers", response_model=list[CustomerSummary], tags=["customers"])
def list_customers(
    region: str | None = None,
    rep: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> list[CustomerSummary]:
    df = store["customers"]
    if region:
        df = df[df["region"] == region]
    if rep:
        df = df[df["rep"] == rep]
    df = df.sort_values("total_revenue", ascending=False).head(limit)
    return [CustomerSummary(**row) for row in
            df[["customer_id", "customer_name", "region", "rep", "rfm_segment",
                "churn_risk", "total_revenue", "clv_12m_runrate"]].to_dict("records")]


@app.get("/customers/{customer_id}", response_model=CustomerSummary, tags=["customers"])
def get_customer(customer_id: str) -> CustomerSummary:
    df = store["customers"]
    match = df[df["customer_id"] == customer_id]
    if match.empty:
        raise HTTPException(404, f"unknown customer_id {customer_id!r}")
    row = match.iloc[0]
    return CustomerSummary(**row[["customer_id", "customer_name", "region", "rep",
                                  "rfm_segment", "churn_risk", "total_revenue",
                                  "clv_12m_runrate"]].to_dict())


@app.get("/customers/{customer_id}/recommendations",
         response_model=CustomerRecommendations, tags=["recommendations"])
def get_recommendations(
    customer_id: str,
    limit: int = Query(10, ge=1, le=10),
) -> CustomerRecommendations:
    variant = ab_variant(customer_id)
    recs = store["recs"] if variant == "A" else store["recs_b"]
    match = (recs[recs["customer_id"] == customer_id].sort_values("rank")
             if recs is not None and not recs.empty else pd.DataFrame())
    if match.empty:
        # distinguish "unknown customer" from "known but fully penetrated"
        if customer_id not in set(store["customers"]["customer_id"]):
            raise HTTPException(404, f"unknown customer_id {customer_id!r}")
        fallback = _safe_default(customer_id, limit)
        return CustomerRecommendations(
            customer_id=customer_id,
            customer_name=_name_of(customer_id),
            model="regional-popularity safe default",
            experiment_variant=variant,
            serving_mode="degraded_fallback",
            fallback_reason=("personalized artifact unavailable or no eligible personalized "
                             "candidates remained"),
            recommendations=fallback,
        )

    # serve-time suppressor: anything bought THIS session leaves the list
    # immediately — the batch scores don't know about it yet, the API does
    suppressed = sorted(store["session_purchases"].get(customer_id, set()))
    match = match[~match["sku"].isin(suppressed)].head(limit).copy()
    match["rank"] = range(1, len(match) + 1)

    served = [
        Recommendation(**row) for row in
        match[["rank", "sku", "protein", "description", "score",
               "est_revenue_opportunity", "because_similar_to"]]
        .fillna({"because_similar_to": ""}).to_dict("records")
    ]
    log_telemetry({"event_type": "impression", "customer_id": customer_id,
                   "variant": variant, "skus": [r.sku for r in served]})
    return CustomerRecommendations(
        customer_id=customer_id,
        customer_name=str(match.iloc[0]["customer_name"]) if len(match) else _name_of(customer_id),
        model=("item-based CF (batch-scored)" if variant == "A"
               else "two-stage CF -> gradient-boosted ranker"),
        experiment_variant=variant,
        suppressed_skus=suppressed,
        recommendations=served,
    )


@app.post("/events", status_code=202, tags=["experimentation"])
def track_event(event: TrackedEvent) -> dict:
    """Online feedback loop: clicks and purchases land in the telemetry log
    (the A/B evidence), and purchases update the session store so the very
    next recommendation call already excludes the bought SKU."""
    if event.customer_id not in set(store["customers"]["customer_id"]):
        raise HTTPException(404, f"unknown customer_id {event.customer_id!r}")
    log_telemetry({"event_type": event.event_type,
                   "customer_id": event.customer_id,
                   "variant": ab_variant(event.customer_id),
                   "sku": event.sku})
    if event.event_type == "purchase":
        store["session_purchases"].setdefault(event.customer_id, set()).add(event.sku)
    return {"accepted": True, "variant": ab_variant(event.customer_id)}


@app.get("/recommendations/cold-start/{region}",
         response_model=list[ColdStartRecommendation], tags=["recommendations"])
def cold_start(region: str) -> list[ColdStartRecommendation]:
    """LIVE inference: computes the region-weighted popularity blend on
    request via the engine — the fallback surface for a brand-new customer."""
    sales = store["sales"]
    if region not in set(sales["region"]):
        known = sorted(sales["region"].unique())
        raise HTTPException(404, f"unknown region {region!r}; known regions: {known}")
    out = recommend_for_new_customer(sales, region)
    return [ColdStartRecommendation(**row) for row in
            out[["rank", "sku", "protein", "description", "blend_score"]]
            .to_dict("records")]


@app.get("/affinity", response_model=list[AffinityPair], tags=["insights"])
def affinity(
    min_lift: float = Query(1.2, ge=1.0),
    limit: int = Query(20, ge=1, le=100),
) -> list[AffinityPair]:
    df = store["affinity"]
    df = df[df["lift"] >= min_lift].sort_values("lift", ascending=False).head(limit)
    return [AffinityPair(**row) for row in df.to_dict("records")]


def _name_of(customer_id: str) -> str:
    df = store["customers"]
    m = df[df["customer_id"] == customer_id]
    return str(m.iloc[0]["customer_name"]) if not m.empty else customer_id


def _safe_default(customer_id: str, limit: int) -> list[Recommendation]:
    """Regional-popularity fallback with the same ownership safety boundary.

    The fallback is deliberately conservative: it excludes already-purchased
    products, carries an explicit reason code and assigns no invented dollar
    opportunity. If identity or eligibility cannot be established, the normal
    endpoint path fails closed before reaching this function.
    """
    customer = store["customers"].loc[
        store["customers"]["customer_id"] == customer_id
    ].iloc[0]
    sales = store["sales"]
    history = sales[sales["customer_id"] == customer_id]
    owned = set(history["sku"])
    regional = sales[sales["region"] == customer["region"]]
    ranked = (
        regional.groupby(["sku", "protein", "description"], as_index=False)["quantity_lb"]
        .sum()
        .sort_values(["quantity_lb", "sku"], ascending=[False, True])
    )
    ranked = ranked[~ranked["sku"].isin(owned)].head(limit)
    if ranked.empty:
        return []
    max_volume = float(ranked["quantity_lb"].max()) or 1.0
    return [
        Recommendation(
            rank=index,
            sku=str(row.sku),
            protein=str(row.protein),
            description=str(row.description),
            score=round(float(row.quantity_lb) / max_volume, 6),
            est_revenue_opportunity=0.0,
            because_similar_to="Safe default: regional demand; personalized artifact unavailable",
        )
        for index, row in enumerate(ranked.itertuples(index=False), start=1)
    ]
