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

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
    store["recs"] = pd.read_csv(OUT / "cross_sell_recommendations.csv")
    store["customers"] = pd.read_csv(OUT / "customer_analytics.csv")
    store["affinity"] = pd.read_csv(OUT / "sku_affinity.csv")
    store["actions"] = pd.read_csv(OUT / "action_list.csv")
    store["sales"] = pd.read_csv(DATA / "sales_lines.csv", parse_dates=["order_date"])
    yield
    store.clear()


app = FastAPI(
    title="Customer Cross-Sell Recommendation API",
    description="Serves collaborative-filtering cross-sell recommendations, "
                "customer analytics, and basket-affinity insights for a B2B "
                "protein distributor. Batch-scored, holdout-evaluated "
                "(CF hit-rate@10 = 84.9% vs 75.4% popularity baseline).",
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
    recommendations: list[Recommendation]


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


# ---------------------------------------------------------------- endpoints

@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    return Health(
        status="ok",
        customers_scored=int(store["customers"]["customer_id"].nunique()),
        recommendations_available=len(store["recs"]),
        affinity_pairs=len(store["affinity"]),
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
    recs = store["recs"]
    match = recs[recs["customer_id"] == customer_id].sort_values("rank").head(limit)
    if match.empty:
        # distinguish "unknown customer" from "known but fully penetrated"
        if customer_id not in set(store["customers"]["customer_id"]):
            raise HTTPException(404, f"unknown customer_id {customer_id!r}")
        return CustomerRecommendations(
            customer_id=customer_id,
            customer_name=_name_of(customer_id),
            recommendations=[],
        )
    return CustomerRecommendations(
        customer_id=customer_id,
        customer_name=str(match.iloc[0]["customer_name"]),
        recommendations=[
            Recommendation(**row) for row in
            match[["rank", "sku", "protein", "description", "score",
                   "est_revenue_opportunity", "because_similar_to"]]
            .fillna({"because_similar_to": ""}).to_dict("records")
        ],
    )


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
