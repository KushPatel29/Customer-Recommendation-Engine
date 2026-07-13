"""
Customer analytics — the KPI suite a commercial team standardizes on.

- RFM segmentation (quintile-scored, mapped to the standard named segments:
  Champions, Loyal, Potential Loyalist, At Risk, Hibernating, ...)
- Run-rate CLV: margin run-rate annualized, damped by churn risk. Labeled
  a heuristic on purpose — no BG/NBD black box, every number is auditable.
- Churn risk: each customer's days-since-last-order measured against their
  OWN median reorder interval (a weekly buyer 3 weeks dark is at risk; a
  quarterly buyer isn't).
- Cohort retention: monthly first-purchase cohorts x months-since-first,
  the classic retention triangle.

Outputs: output/customer_analytics.csv, output/rfm_segment_summary.csv,
         output/cohort_retention.csv, output/action_list.csv
Visuals: docs/rfm_segments.png, docs/cohort_retention.png

Usage:
    python analytics/customer_analytics.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from recommend import cross_sell_recommendations, load_sales

OUT = ROOT / "output"
DOCS = ROOT / "docs"
OUT.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

NAVY, TEAL, ORANGE, PLUM, RED = "#12436D", "#28A197", "#F46A25", "#801650", "#C0392B"


# ------------------------------------------------------------------ RFM

def rfm_segment(r: int, fm: float) -> str:
    """Standard R x FM segment map (R = recency quintile, FM = mean of
    frequency and monetary quintiles). 5 = best."""
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 4 and fm >= 2.5:
        return "Loyal"
    if r >= 4:
        return "New / Promising"
    if r >= 3 and fm >= 3:
        return "Potential Loyalist"
    if r >= 3:
        return "Needs Attention"
    if fm >= 4:
        return "At Risk (was valuable)"
    if fm >= 2.5:
        return "At Risk"
    return "Hibernating"


def customer_metrics(sales: pd.DataFrame) -> pd.DataFrame:
    as_of = sales["order_date"].max() + pd.Timedelta(days=1)
    sales = sales.assign(margin=sales["revenue"] - sales["cost"])

    per_order = (sales.groupby(["customer_id", "order_id"])
                 .agg(order_date=("order_date", "first"),
                      order_revenue=("revenue", "sum"))
                 .reset_index())

    # median reorder interval per customer (their own cadence)
    intervals = (per_order.sort_values("order_date")
                 .groupby("customer_id")["order_date"]
                 .apply(lambda d: d.diff().dt.days.median()))

    cust = (sales.groupby(["customer_id", "customer_name", "region", "rep"])
            .agg(total_revenue=("revenue", "sum"),
                 total_margin=("margin", "sum"),
                 orders=("order_id", "nunique"),
                 distinct_skus=("sku", "nunique"),
                 first_order=("order_date", "min"),
                 last_order=("order_date", "max"))
            .reset_index())
    cust["avg_order_value"] = (cust["total_revenue"] / cust["orders"]).round(2)
    cust["tenure_days"] = (as_of - cust["first_order"]).dt.days
    cust["recency_days"] = (as_of - cust["last_order"]).dt.days
    cust["median_reorder_days"] = cust["customer_id"].map(intervals)

    # churn risk: recency measured in units of the customer's own cadence
    ratio = cust["recency_days"] / cust["median_reorder_days"].clip(lower=3)
    cust["churn_risk"] = np.select(
        [ratio > 3, ratio > 1.5], ["High", "Medium"], default="Low")
    cust.loc[cust["median_reorder_days"].isna(), "churn_risk"] = "One-time buyer"

    # RFM quintiles (R inverted: low recency_days = high score)
    cust["R"] = pd.qcut(cust["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    cust["F"] = pd.qcut(cust["orders"].rank(method="first"), 5,
                        labels=[1, 2, 3, 4, 5]).astype(int)
    cust["M"] = pd.qcut(cust["total_revenue"].rank(method="first"), 5,
                        labels=[1, 2, 3, 4, 5]).astype(int)
    cust["rfm_segment"] = [rfm_segment(r, (f + m) / 2)
                           for r, f, m in zip(cust["R"], cust["F"], cust["M"])]

    # run-rate CLV: annualized margin run-rate, damped by churn risk
    monthly_margin = cust["total_margin"] / (cust["tenure_days"].clip(lower=30) / 30.4)
    damp = cust["churn_risk"].map(
        {"Low": 1.0, "Medium": 0.6, "High": 0.25, "One-time buyer": 0.1})
    cust["clv_12m_runrate"] = (monthly_margin * 12 * damp).round(0)

    return cust.round({"total_revenue": 2, "total_margin": 2,
                       "median_reorder_days": 1})


# ------------------------------------------------------------ cohorts

def cohort_retention(sales: pd.DataFrame) -> pd.DataFrame:
    orders = (sales.groupby(["customer_id", "order_id"])
              .agg(order_date=("order_date", "first")).reset_index())
    orders["order_month"] = orders["order_date"].dt.to_period("M")
    first = orders.groupby("customer_id")["order_month"].min().rename("cohort")
    orders = orders.join(first, on="customer_id")
    orders["months_since"] = ((orders["order_month"] - orders["cohort"])
                              .apply(lambda p: p.n))
    active = (orders.groupby(["cohort", "months_since"])["customer_id"]
              .nunique().unstack(fill_value=0))
    sizes = active[0]
    retention = active.div(sizes, axis=0)
    # mask cells beyond the observable window
    max_month = orders["order_month"].max()
    for cohort in retention.index:
        horizon = (max_month - cohort).n
        retention.loc[cohort, retention.columns > horizon] = np.nan
    retention.index = retention.index.astype(str)
    return retention.round(3)


# ------------------------------------------------------------ visuals

def plot_rfm(cust: pd.DataFrame):
    order = (cust.groupby("rfm_segment")
             .agg(customers=("customer_id", "count"),
                  revenue=("total_revenue", "sum"))
             .sort_values("revenue", ascending=True))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(order.index, order["revenue"] / 1000, color=NAVY)
    for bar, (seg, row) in zip(bars, order.iterrows()):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                f"{row['customers']} customers", va="center", fontsize=8.5,
                color="#5A6570")
    ax.set_xlabel("revenue ($k)")
    ax.set_title("Revenue by RFM segment — where the money and the risk sit",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "rfm_segments.png", dpi=130)


def plot_cohorts(retention: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(retention.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(retention.shape[1]), retention.columns)
    ax.set_yticks(range(retention.shape[0]), retention.index)
    for i in range(retention.shape[0]):
        for j in range(retention.shape[1]):
            v = retention.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else NAVY)
    ax.set_xlabel("months since first purchase")
    ax.set_ylabel("first-purchase cohort")
    ax.set_title("Cohort retention — % of each cohort ordering again N months later",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(DOCS / "cohort_retention.png", dpi=130)


# ------------------------------------------------------------ action list

def build_action_list(cust: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """The Monday-morning call list: for every customer worth a call
    (valuable + at risk, or healthy + growable), the segment, the risk,
    and the single best cross-sell with its why and $ size."""
    recs = cross_sell_recommendations(sales)
    best = recs[recs["rank"] == 1].set_index("customer_id")

    focus = cust[cust["rfm_segment"].isin(
        ["Champions", "Loyal", "At Risk (was valuable)", "At Risk",
         "Potential Loyalist"])].copy()
    focus = focus.join(best[["description", "est_revenue_opportunity",
                             "because_similar_to"]], on="customer_id")
    focus = focus.rename(columns={"description": "top_cross_sell"})
    # customers with no white-space left aren't skipped — they're retention calls
    full = focus["top_cross_sell"].isna()
    focus.loc[full, "top_cross_sell"] = "(fully penetrated — retention call)"
    focus.loc[full, "est_revenue_opportunity"] = 0.0
    focus.loc[full, "because_similar_to"] = ""
    cols = ["customer_id", "customer_name", "region", "rep", "rfm_segment",
            "churn_risk", "clv_12m_runrate", "recency_days",
            "top_cross_sell", "est_revenue_opportunity", "because_similar_to"]
    return (focus[cols]
            .sort_values(["churn_risk", "clv_12m_runrate"],
                         ascending=[True, False])
            .reset_index(drop=True))


def main():
    sales = load_sales()
    cust = customer_metrics(sales)
    retention = cohort_retention(sales)
    actions = build_action_list(cust, sales)

    cust.to_csv(OUT / "customer_analytics.csv", index=False)
    (cust.groupby("rfm_segment")
     .agg(customers=("customer_id", "count"),
          revenue=("total_revenue", "sum"),
          avg_clv=("clv_12m_runrate", "mean"))
     .round(0).sort_values("revenue", ascending=False)
     .to_csv(OUT / "rfm_segment_summary.csv"))
    retention.to_csv(OUT / "cohort_retention.csv")
    actions.to_csv(OUT / "action_list.csv", index=False)

    plot_rfm(cust)
    plot_cohorts(retention)

    at_risk_value = cust.loc[cust["rfm_segment"].str.startswith("At Risk"),
                             "total_revenue"].sum()
    print(f"customers analyzed: {len(cust)}")
    print(cust["rfm_segment"].value_counts().to_string())
    print(f"\nrevenue sitting in At-Risk segments: {at_risk_value:,.0f}")
    print(f"action list: {len(actions)} customers with segment + risk + best rec")


if __name__ == "__main__":
    main()
