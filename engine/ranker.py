"""
Stage 2 of the two-stage pipeline: learned re-ranking of retrieved candidates.

The industry-standard production RecSys shape is retrieval -> ranking:

  Stage 1 (retrieval)  — collaborative filtering fetches a generous candidate
                         pool (top 30) per customer. Cheap, recall-oriented.
  Stage 2 (ranking)    — a gradient-boosted ranker re-scores those candidates
                         with features CF cannot see: product economics
                         (margin %, ABC class, repeat-purchase rate) and
                         customer context (RFM, churn risk, CLV run-rate).
                         Precision-oriented, business-aware.

At this repo's scale a single stage is computationally fine — the point of
building both stages is that the *evaluation* can then say whether the extra
machinery earns its keep. The ranker trains on the same leakage-safe holdout
protocol as everything else (candidates from a model that never saw the
hidden SKUs; hidden SKUs are the positives), and evaluate_holdout.py scores
the full two-stage system against single-stage CF. The winner ships; the
loser stays in the bake-off table as a receipt.

Usage:
    python engine/ranker.py     # trains, evaluates alongside CF, writes recs
"""

import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.recommend import (  # noqa: E402
    build_customer_sku_matrix,
    cross_sell_recommendations,
    load_sales,
)

OUT = ROOT / "output"
N_CANDIDATES = 30      # stage-1 pool size
N_FINAL = 10           # stage-2 output size
RANDOM_STATE = 42

PRODUCT_FEATURES = ["margin_pct", "repeat_purchase_rate", "revenue_share",
                    "velocity_lb_per_week"]
CUSTOMER_FEATURES = ["R", "F", "M", "clv_12m_runrate", "recency_days",
                     "protein_breadth"]


def _feature_frame(candidates: pd.DataFrame,
                   products: pd.DataFrame,
                   customers: pd.DataFrame) -> pd.DataFrame:
    """(customer, candidate SKU) pairs -> model features."""
    f = candidates.merge(products[["sku", "abc_class"] + PRODUCT_FEATURES],
                         on="sku", how="left")
    f = f.merge(customers[["customer_id"] + CUSTOMER_FEATURES],
                on="customer_id", how="left")
    f["abc_rank"] = f["abc_class"].map({"A": 0, "B": 1, "C": 2}).fillna(2)
    feature_cols = (["score", "abc_rank"] + PRODUCT_FEATURES + CUSTOMER_FEATURES)
    return f, feature_cols


def build_training_pairs(sales: pd.DataFrame, products: pd.DataFrame,
                         customers: pd.DataFrame,
                         min_skus: int = 8, holdout_frac: float = 0.25,
                         seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Leakage-safe supervision: per eligible customer, hide a slice of their
    SKUs, retrieve candidates from a matrix built WITHOUT them, and label a
    candidate 1 iff it is one of the hidden (truly re-purchased) SKUs."""
    rng = random.Random(seed)
    sku_counts = sales.groupby("customer_id")["sku"].nunique()
    eligible = sku_counts[sku_counts >= min_skus].index.tolist()

    rows = []
    for cust in eligible:
        cust_skus = sorted(sales.loc[sales["customer_id"] == cust, "sku"].unique())
        hidden = set(rng.sample(cust_skus, max(2, int(len(cust_skus) * holdout_frac))))
        train_sales = sales[~((sales["customer_id"] == cust)
                              & (sales["sku"].isin(hidden)))]
        cands = cross_sell_recommendations(
            train_sales, matrix=build_customer_sku_matrix(train_sales),
            n_recs=N_CANDIDATES)
        cands = cands[cands["customer_id"] == cust][["customer_id", "sku", "score"]]
        cands["label"] = cands["sku"].isin(hidden).astype(int)
        rows.append(cands)
    return pd.concat(rows, ignore_index=True)


def train_ranker(pairs: pd.DataFrame, products: pd.DataFrame,
                 customers: pd.DataFrame) -> tuple:
    f, cols = _feature_frame(pairs, products, customers)
    model = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, random_state=RANDOM_STATE)
    model.fit(f[cols], f["label"])
    return model, cols


def rerank(model, feature_cols: list[str], candidates: pd.DataFrame,
           products: pd.DataFrame, customers: pd.DataFrame,
           n_final: int = N_FINAL) -> pd.DataFrame:
    """Stage 2: score the stage-1 pool, keep the top n_final per customer."""
    f, cols = _feature_frame(candidates, products, customers)
    f["rank_score"] = model.predict_proba(f[cols])[:, 1]
    out = (f.sort_values(["customer_id", "rank_score"], ascending=[True, False])
           .groupby("customer_id").head(n_final).copy())
    out["rank"] = out.groupby("customer_id").cumcount() + 1
    return out


def main() -> None:
    sales = load_sales()
    products = pd.read_csv(OUT / "product_analytics.csv")
    customers = pd.read_csv(OUT / "customer_analytics.csv")

    print("stage 2: building leakage-safe training pairs "
          f"(top-{N_CANDIDATES} CF candidates per customer) ...")
    pairs = build_training_pairs(sales, products, customers)
    pos_rate = pairs["label"].mean()
    print(f"  {len(pairs):,} pairs, positive rate {pos_rate:.1%}")

    model, cols = train_ranker(pairs, products, customers)

    # production pass: full-history candidates, re-ranked
    full_candidates = cross_sell_recommendations(sales, n_recs=N_CANDIDATES)
    reranked = rerank(model, cols, full_candidates, products, customers)
    keep = ["customer_id", "customer_name", "rank", "sku", "protein",
            "description", "score", "rank_score", "est_revenue_opportunity",
            "because_similar_to"]
    reranked[keep].to_csv(OUT / "two_stage_recommendations.csv", index=False)
    print(f"wrote two_stage_recommendations.csv "
          f"({reranked['customer_id'].nunique()} customers x top {N_FINAL})")


if __name__ == "__main__":
    main()
