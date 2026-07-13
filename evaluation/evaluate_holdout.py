"""
Holdout evaluation: does the collaborative filter actually work?

Protocol (per eligible customer, one with >= 8 distinct SKUs):
  1. Hide 25% of their SKUs (all purchase lines for those SKUs removed).
  2. Rebuild the customer x SKU matrix on the remaining data only.
  3. Ask the engine for top-10 cross-sell recommendations.
  4. Score hit-rate@10: how many hidden SKUs the engine re-discovered.

Compared against a popularity baseline (recommend the globally best-selling
SKUs the customer doesn't own). If CF can't beat popularity, ship
popularity — same discipline as the demand-forecast backtest in the
supply-chain repo.

Usage:
    python evaluation/evaluate_holdout.py
"""

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from recommend import (build_customer_sku_matrix, cross_sell_recommendations,
                       load_sales)

random.seed(42)

MIN_SKUS = 8
HOLDOUT_FRAC = 0.25
K = 10


def popularity_baseline(train: pd.DataFrame, owned: set, k=K) -> list:
    pop = (train.groupby("sku")["quantity_lb"].sum()
           .sort_values(ascending=False).index)
    return [s for s in pop if s not in owned][:k]


def evaluate(sales: pd.DataFrame) -> pd.DataFrame:
    sku_counts = sales.groupby("customer_id")["sku"].nunique()
    eligible = sku_counts[sku_counts >= MIN_SKUS].index.tolist()

    rows = []
    for cust in eligible:
        cust_skus = sorted(sales.loc[sales["customer_id"] == cust, "sku"].unique())
        n_hide = max(2, int(len(cust_skus) * HOLDOUT_FRAC))
        hidden = set(random.sample(cust_skus, n_hide))

        train = sales[~((sales["customer_id"] == cust) & (sales["sku"].isin(hidden)))]
        owned = set(train.loc[train["customer_id"] == cust, "sku"])

        recs = cross_sell_recommendations(train, n_recs=K)
        cf_recs = set(recs.loc[recs["customer_id"] == cust, "sku"])
        pop_recs = set(popularity_baseline(train, owned))

        rows.append({
            "customer_id": cust,
            "hidden": len(hidden),
            "cf_hits": len(cf_recs & hidden),
            "pop_hits": len(pop_recs & hidden),
        })
    return pd.DataFrame(rows)


def main():
    sales = load_sales()
    results = evaluate(sales)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "holdout_evaluation.csv", index=False)

    cf = results["cf_hits"].sum() / results["hidden"].sum()
    pop = results["pop_hits"].sum() / results["hidden"].sum()
    print(f"customers evaluated: {len(results)}")
    print(f"hit-rate@{K}  collaborative filtering: {cf:.1%}")
    print(f"hit-rate@{K}  popularity baseline:     {pop:.1%}")
    print(f"CF lift over popularity: {cf / pop:.2f}x" if pop else "n/a")


if __name__ == "__main__":
    main()
