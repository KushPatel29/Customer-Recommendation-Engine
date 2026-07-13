"""
Customer recommendation engine — collaborative filtering + basket affinity.

Three recommendation surfaces a B2B sales team actually uses:

1. Growth targets  — customers outside the regional top-N whose revenue and
                     order cadence say they're ready to be grown (v1 logic,
                     kept and cleaned up).
2. Cross-sell      — item-based collaborative filtering: for each customer,
                     "white-space" SKUs they don't buy but their most similar
                     customers do, scored by similarity-weighted spend.
3. SKU affinity    — market-basket lift on order-level co-occurrence:
                     "orders with striploin are N.Nx more likely to include
                     short ribs" — the talk track a rep can actually say.

Pure pandas/numpy/scikit-learn. Deterministic. No network calls — v1's Bing
scraping was removed deliberately (fragile, untestable, and unnecessary once
the purchase history itself carries the signal).

Usage:
    python engine/recommend.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

TOP_N_PER_REGION = 10
N_SIMILAR = 8          # neighbors used for cross-sell scoring
N_RECS = 10            # cross-sell SKUs per customer
MIN_ORDERS_GROWTH = 3  # growth targets must have real cadence
MIN_LIFT = 1.2         # affinity pairs below this aren't worth a rep's breath
MIN_PAIR_SUPPORT = 12  # co-occurrence count floor (kills noise pairs)


def load_sales(path=None):
    return pd.read_csv(path or ROOT / "data" / "sales_lines.csv",
                       parse_dates=["order_date"])


# ------------------------------------------------------------ 1. top & growth

def top_customers_by_region(sales: pd.DataFrame, n=TOP_N_PER_REGION) -> pd.DataFrame:
    agg = (sales.groupby(["region", "customer_id", "customer_name"])
           .agg(total_revenue=("revenue", "sum"),
                order_count=("order_id", "nunique"))
           .reset_index())
    agg["rank_in_region"] = (agg.groupby("region")["total_revenue"]
                             .rank(method="first", ascending=False).astype(int))
    return (agg[agg["rank_in_region"] <= n]
            .sort_values(["region", "rank_in_region"])
            .reset_index(drop=True))


def growth_targets(sales: pd.DataFrame, top_df: pd.DataFrame,
                   min_orders=MIN_ORDERS_GROWTH) -> pd.DataFrame:
    """Non-top customers with real cadence, ranked by revenue momentum:
    second-half revenue vs first-half revenue of the window."""
    top_ids = set(top_df["customer_id"])
    pool = sales[~sales["customer_id"].isin(top_ids)].copy()

    midpoint = pool["order_date"].min() + (pool["order_date"].max()
                                           - pool["order_date"].min()) / 2
    pool["half"] = np.where(pool["order_date"] <= midpoint, "h1", "h2")

    agg = (pool.groupby(["region", "customer_id", "customer_name"])
           .agg(total_revenue=("revenue", "sum"),
                order_count=("order_id", "nunique"))
           .reset_index())
    halves = (pool.pivot_table(index="customer_id", columns="half",
                               values="revenue", aggfunc="sum", fill_value=0.0)
              .reindex(columns=["h1", "h2"], fill_value=0.0))
    agg = agg.merge(halves, on="customer_id", how="left").fillna({"h1": 0, "h2": 0})
    agg["momentum"] = (agg["h2"] - agg["h1"]) / agg[["h1", "h2"]].max(axis=1).replace(0, np.nan)
    agg = agg[agg["order_count"] >= min_orders].copy()
    agg["momentum"] = agg["momentum"].fillna(0).round(3)
    return (agg.sort_values(["momentum", "total_revenue"], ascending=False)
            .reset_index(drop=True)
            .rename(columns={"h1": "revenue_h1", "h2": "revenue_h2"}))


# ------------------------------------------------------ 2. CF cross-sell

def build_customer_sku_matrix(sales: pd.DataFrame) -> pd.DataFrame:
    """Customer x SKU matrix of log-damped quantities. Log damping stops one
    giant standing order from defining a customer's whole profile."""
    qty = sales.pivot_table(index="customer_id", columns="sku",
                            values="quantity_lb", aggfunc="sum", fill_value=0.0)
    return np.log1p(qty)


def customer_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(matrix.values)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def cross_sell_recommendations(sales: pd.DataFrame, matrix=None, sim=None,
                               n_similar=N_SIMILAR, n_recs=N_RECS) -> pd.DataFrame:
    """For each customer: SKUs they have never bought, scored by how heavily
    their nearest neighbors buy them (similarity-weighted)."""
    if matrix is None:
        matrix = build_customer_sku_matrix(sales)
    if sim is None:
        sim = customer_similarity(matrix)

    sku_info = (sales.groupby("sku")
                .agg(protein=("protein", "first"), description=("description", "first"),
                     avg_price=("unit_price", "mean"))
                .to_dict("index"))
    names = sales.drop_duplicates("customer_id").set_index("customer_id")["customer_name"]

    rows = []
    for cust in matrix.index:
        owned = set(matrix.columns[matrix.loc[cust] > 0])
        neighbors = sim[cust].drop(cust).nlargest(n_similar)
        if neighbors.sum() <= 0:
            continue
        # similarity-weighted neighbor basket
        neighbor_profile = (matrix.loc[neighbors.index]
                            .mul(neighbors.values, axis=0).sum() / neighbors.sum())
        candidates = neighbor_profile.drop(labels=list(owned)).nlargest(n_recs)
        for rank, (sku, score) in enumerate(candidates.items(), 1):
            if score <= 0:
                continue
            # explainability: the most similar neighbor who actually buys this SKU
            buyers = [n for n in neighbors.index if matrix.at[n, sku] > 0]
            because = names.get(buyers[0], buyers[0]) if buyers else ""
            # indicative $ opportunity: implied lb volume x average street price
            est_lb = float(np.expm1(score))
            opportunity = round(est_lb * sku_info[sku]["avg_price"], 2)
            rows.append({
                "customer_id": cust,
                "customer_name": names.get(cust, cust),
                "rank": rank,
                "sku": sku,
                "protein": sku_info[sku]["protein"],
                "description": sku_info[sku]["description"],
                "score": round(float(score), 4),
                "est_revenue_opportunity": opportunity,
                "because_similar_to": because,
            })
    return pd.DataFrame(rows)


def recommend_for_new_customer(sales: pd.DataFrame, region: str,
                               n_recs=N_RECS) -> pd.DataFrame:
    """Cold start: a brand-new customer has no purchase history, so fall back
    to a region-weighted popularity blend (70% their region, 30% global) —
    the honest default until they have orders to learn from."""
    regional = (sales[sales["region"] == region].groupby("sku")["revenue"].sum())
    overall = sales.groupby("sku")["revenue"].sum()
    blend = (0.7 * regional / regional.max()).add(
        0.3 * overall / overall.max(), fill_value=0.0)
    desc = sales.drop_duplicates("sku").set_index("sku")[["protein", "description"]]
    top = blend.nlargest(n_recs)
    out = desc.loc[top.index].reset_index()
    out.insert(0, "region", region)
    out["blend_score"] = top.values.round(4)
    out["rank"] = range(1, len(out) + 1)
    return out


# ------------------------------------------------------ 3. basket affinity

def sku_affinity(sales: pd.DataFrame, min_lift=MIN_LIFT,
                 min_support=MIN_PAIR_SUPPORT) -> pd.DataFrame:
    """Order-level co-occurrence lift between SKU pairs."""
    baskets = sales.groupby("order_id")["sku"].apply(set)
    n_orders = len(baskets)
    counts, pair_counts = {}, {}
    for basket in baskets:
        for a in basket:
            counts[a] = counts.get(a, 0) + 1
        basket = sorted(basket)
        for i, a in enumerate(basket):
            for b in basket[i + 1:]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    desc = sales.drop_duplicates("sku").set_index("sku")["description"].to_dict()
    rows = []
    for (a, b), both in pair_counts.items():
        if both < min_support:
            continue
        support = both / n_orders
        lift = support / ((counts[a] / n_orders) * (counts[b] / n_orders))
        if lift >= min_lift:
            rows.append({
                "sku_a": a, "description_a": desc[a],
                "sku_b": b, "description_b": desc[b],
                "orders_together": both,
                "confidence_a_to_b": round(both / counts[a], 3),
                "lift": round(lift, 3),
            })
    return (pd.DataFrame(rows).sort_values("lift", ascending=False)
            .reset_index(drop=True))


# ------------------------------------------------------------ pipeline

def main():
    sales = load_sales()
    top_df = top_customers_by_region(sales)
    growth = growth_targets(sales, top_df)
    cross = cross_sell_recommendations(sales)
    affinity = sku_affinity(sales)

    top_df.to_csv(OUT / "top_customers.csv", index=False)
    growth.to_csv(OUT / "growth_targets.csv", index=False)
    cross.to_csv(OUT / "cross_sell_recommendations.csv", index=False)
    affinity.to_csv(OUT / "sku_affinity.csv", index=False)

    print(f"top customers:   {len(top_df):5d} rows ({TOP_N_PER_REGION}/region)")
    print(f"growth targets:  {len(growth):5d} customers")
    print(f"cross-sell recs: {len(cross):5d} rows "
          f"({cross['customer_id'].nunique()} customers x top {N_RECS})")
    print(f"affinity pairs:  {len(affinity):5d} (lift >= {MIN_LIFT}, "
          f"support >= {MIN_PAIR_SUPPORT} orders)")
    if not affinity.empty:
        t = affinity.iloc[0]
        print(f"\nstrongest pair: {t['description_a']} + {t['description_b']} "
              f"(lift {t['lift']}, together in {t['orders_together']} orders)")


if __name__ == "__main__":
    main()
