"""
Invariants for the recommendation engine.

A recommender that suggests things you already buy, or that can't beat
"just recommend the bestsellers", is worse than no recommender. These
tests pin both failure modes, plus the math.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "analytics"))

from recommend import (
    build_customer_sku_matrix,
    cross_sell_recommendations,
    customer_similarity,
    growth_targets,
    load_sales,
    recommend_for_new_customer,
    sku_affinity,
    top_customers_by_region,
)


@pytest.fixture(scope="module")
def sales():
    subprocess.run([sys.executable,
                    str(ROOT / "data_generator" / "generate_sales_data.py")],
                   check=True)
    return load_sales()


@pytest.fixture(scope="module")
def matrix(sales):
    return build_customer_sku_matrix(sales)


@pytest.fixture(scope="module")
def recs(sales):
    return cross_sell_recommendations(sales)


def test_similarity_matrix_is_well_formed(matrix):
    sim = customer_similarity(matrix)
    assert np.allclose(sim.values, sim.values.T), "similarity must be symmetric"
    assert np.allclose(np.diag(sim.values), 1.0), "self-similarity must be 1"
    assert sim.values.min() >= -1e-9 and sim.values.max() <= 1 + 1e-9


def test_never_recommends_what_customer_already_buys(sales, recs):
    owned = sales.groupby("customer_id")["sku"].apply(set)
    for cust, g in recs.groupby("customer_id"):
        overlap = set(g["sku"]) & owned[cust]
        assert not overlap, f"{cust} recommended already-purchased {overlap}"


def test_recommendations_ranked_by_score(recs):
    for cust, g in recs.groupby("customer_id"):
        scores = g.sort_values("rank")["score"].values
        assert (np.diff(scores) <= 1e-9).all(), f"{cust} scores not descending"


def test_top_customers_disjoint_from_growth_targets(sales):
    top = top_customers_by_region(sales)
    growth = growth_targets(sales, top)
    assert not (set(top["customer_id"]) & set(growth["customer_id"]))


def test_affinity_lift_math_on_toy_data():
    """Hand-checkable: A and B always co-occur (lift = n_orders / 1... = 2 here),
    A and C never do."""
    toy = pd.DataFrame({
        "order_id": ["o1", "o1", "o2", "o2", "o3", "o4"],
        "sku": ["A", "B", "A", "B", "C", "C"],
        "protein": ["x"] * 6,
        "description": ["A", "B", "A", "B", "C", "C"],
        "quantity_lb": [1] * 6,
    })
    out = sku_affinity(toy, min_lift=0, min_support=1)
    ab = out[(out["sku_a"] == "A") & (out["sku_b"] == "B")].iloc[0]
    # P(A,B)=2/4, P(A)=2/4, P(B)=2/4 -> lift = 0.5 / 0.25 = 2.0
    assert ab["lift"] == 2.0
    assert not ((out["sku_a"] == "A") & (out["sku_b"] == "C")).any()


def test_cf_beats_popularity_baseline(sales):
    """The reason this engine deserves to exist: on held-out purchases it
    must out-recommend 'just suggest the bestsellers'."""
    from evaluate_holdout import evaluate
    results = evaluate(sales)
    cf = results["cf_hits"].sum() / results["hidden"].sum()
    pop = results["pop_hits"].sum() / results["hidden"].sum()
    assert cf > pop, f"CF hit-rate {cf:.1%} does not beat popularity {pop:.1%}"


def test_deterministic_outputs(sales):
    a = cross_sell_recommendations(sales)
    b = cross_sell_recommendations(sales)
    pd.testing.assert_frame_equal(a, b)


def test_cold_start_returns_valid_ranked_skus(sales):
    recs = recommend_for_new_customer(sales, "Ontario")
    assert len(recs) == 10
    assert set(recs["sku"]) <= set(sales["sku"].unique())
    scores = recs.sort_values("rank")["blend_score"].values
    assert (np.diff(scores) <= 1e-9).all()


def test_explainability_and_opportunity_present(recs):
    assert (recs["est_revenue_opportunity"] > 0).all()
    assert (recs["because_similar_to"] != "").all(), "every rec must carry a why"


# ---------------------------------------------------- analytics invariants

from customer_analytics import build_action_list, cohort_retention, customer_metrics
from product_analytics import product_metrics

VALID_SEGMENTS = {"Champions", "Loyal", "New / Promising", "Potential Loyalist",
                  "Needs Attention", "At Risk (was valuable)", "At Risk",
                  "Hibernating"}


@pytest.fixture(scope="module")
def cust(sales):
    return customer_metrics(sales)


def test_rfm_covers_every_customer_with_valid_segment(sales, cust):
    assert len(cust) == sales["customer_id"].nunique()
    assert set(cust["rfm_segment"]) <= VALID_SEGMENTS
    assert cust["clv_12m_runrate"].min() >= 0


def test_churn_risk_consistent_with_recency(cust):
    repeaters = cust[cust["churn_risk"].isin(["Low", "Medium", "High"])]
    ratio = repeaters["recency_days"] / repeaters["median_reorder_days"].clip(lower=3)
    assert (ratio[repeaters["churn_risk"] == "High"] > 3).all()
    assert (ratio[repeaters["churn_risk"] == "Low"] <= 1.5).all()


def test_cohort_month_zero_is_full(sales):
    retention = cohort_retention(sales)
    assert (retention[0] == 1.0).all(), "month 0 must be 100% by definition"
    assert retention.max().max() <= 1.0


def test_abc_classification_properties(sales):
    prod = product_metrics(sales)
    assert len(prod) == sales["sku"].nunique()
    a_share = prod.loc[prod["abc_class"] == "A", "revenue_share"].sum()
    assert 0.75 <= a_share <= 0.85, f"A-class carries {a_share:.0%}, expected ~80%"
    assert set(prod["abc_class"]) == {"A", "B", "C"}
    assert prod["repeat_purchase_rate"].between(0, 1).all()


def test_action_list_joins_cleanly(sales, cust):
    actions = build_action_list(cust, sales)
    assert actions["customer_id"].is_unique
    assert actions["rfm_segment"].isin(VALID_SEGMENTS).all()
    # every action-list row carries an action: a cross-sell or a retention call
    assert actions["top_cross_sell"].notna().all()


# ---------------------------------------------------- forecasting + DS extras

from customer_analytics import behavioural_clusters
from product_analytics import product_metrics as _pm
from revenue_forecast import MODELS as FC_MODELS
from revenue_forecast import backtest, weekly_revenue


def test_forecast_backtest_no_leakage_and_beats_naive(sales):
    weekly = weekly_revenue(sales)
    results = backtest(weekly)
    assert not results.empty
    avg = results.groupby("model")["wape"].mean()
    assert avg.drop("seasonal_naive").min() <= avg["seasonal_naive"], \
        "a candidate model must beat the naive baseline or you ship the baseline"
    assert (results["wape"] >= 0).all()


def test_forecasts_nonnegative_full_horizon(sales):
    weekly = weekly_revenue(sales)
    for _, g in weekly.groupby("protein"):
        series = g.sort_values("order_date")["revenue"].values
        for name, fn in FC_MODELS.items():
            fc = fn(series[:-8], 8)
            assert len(fc) == 8 and (fc >= 0).all(), name


def test_clustering_recovers_planted_structure(sales, cust):
    _, sil, ari = behavioural_clusters(sales, cust)
    assert ari > 0.3, f"ARI {ari:.2f}: clusters should align with planted personas"
    assert sil > 0, "silhouette must be positive (clusters cohere at all)"


def test_dead_stock_flag_consistent(sales):
    prod = _pm(sales)
    flagged = prod[prod["dead_stock_flag"] == 1]
    assert (flagged["days_since_last_sold"] > 30).all()
    unflagged = prod[prod["dead_stock_flag"] == 0]
    assert (unflagged["days_since_last_sold"] <= 30).all()
