"""
Product analytics — the portfolio-management KPI suite.

- ABC classification: A = SKUs covering the first 80% of revenue,
  B = next 15%, C = the tail (the inventory-policy workhorse).
- Growth-share portfolio quadrant: revenue (share proxy) x H2-vs-H1 growth,
  bubble = margin % — the BCG lens applied to a SKU catalog.
- Margin analysis: realized margin % per SKU (price dispersion aware).
- Repeat purchase rate: share of a SKU's buyers who bought it 2+ times —
  the stickiness signal that separates staples from one-off curiosities.
- Buyer reach: distinct customers per SKU.

Outputs: output/product_analytics.csv
Visuals: docs/product_portfolio.png, docs/abc_pareto.png

Usage:
    python analytics/product_analytics.py
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

from recommend import load_sales

OUT = ROOT / "output"
DOCS = ROOT / "docs"
OUT.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

NAVY, TEAL, ORANGE, PLUM = "#12436D", "#28A197", "#F46A25", "#801650"
ABC_COLORS = {"A": NAVY, "B": TEAL, "C": "#9AA5B1"}


def product_metrics(sales: pd.DataFrame) -> pd.DataFrame:
    sales = sales.assign(margin=sales["revenue"] - sales["cost"])
    midpoint = sales["order_date"].min() + (sales["order_date"].max()
                                            - sales["order_date"].min()) / 2

    prod = (sales.groupby(["sku", "protein", "description"])
            .agg(revenue=("revenue", "sum"),
                 margin=("margin", "sum"),
                 quantity_lb=("quantity_lb", "sum"),
                 orders=("order_id", "nunique"),
                 buyers=("customer_id", "nunique"))
            .reset_index())
    prod["margin_pct"] = (prod["margin"] / prod["revenue"]).round(4)

    # weekly velocity
    weeks = (sales["order_date"].max() - sales["order_date"].min()).days / 7
    prod["velocity_lb_per_week"] = (prod["quantity_lb"] / weeks).round(1)

    # growth: H2 vs H1 revenue
    halves = (sales.assign(half=np.where(sales["order_date"] <= midpoint, "h1", "h2"))
              .pivot_table(index="sku", columns="half", values="revenue",
                           aggfunc="sum", fill_value=0.0))
    prod = prod.merge(halves, on="sku", how="left").fillna({"h1": 0, "h2": 0})
    prod["growth"] = ((prod["h2"] - prod["h1"])
                      / prod[["h1", "h2"]].max(axis=1).replace(0, np.nan)).fillna(0).round(3)

    # repeat purchase rate: buyers purchasing the SKU on 2+ distinct orders
    per_buyer = (sales.groupby(["sku", "customer_id"])["order_id"].nunique())
    repeat = (per_buyer >= 2).groupby("sku").mean().rename("repeat_purchase_rate")
    prod = prod.join(repeat.round(3), on="sku")

    # ABC classification on cumulative revenue
    prod = prod.sort_values("revenue", ascending=False).reset_index(drop=True)
    cum = prod["revenue"].cumsum() / prod["revenue"].sum()
    prod["abc_class"] = np.select([cum <= 0.80, cum <= 0.95], ["A", "B"], default="C")
    prod["revenue_share"] = (prod["revenue"] / prod["revenue"].sum()).round(4)
    return prod.round({"revenue": 2, "margin": 2})


def plot_portfolio(prod: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x, y = prod["revenue"] / 1000, prod["growth"] * 100
    size = 2200 * prod["margin_pct"]
    for abc, color in ABC_COLORS.items():
        m = prod["abc_class"] == abc
        ax.scatter(x[m], y[m], s=size[m], c=color, alpha=0.75,
                   edgecolors="white", linewidths=0.8, label=f"{abc}-class")
    ax.axhline(0, color="#9AA5B1", lw=1)
    ax.axvline(x.median(), color="#9AA5B1", lw=1, ls="--")
    # annotate the notable corners
    for _, r in prod.nlargest(3, "revenue").iterrows():
        ax.annotate(r["description"], (r["revenue"] / 1000, r["growth"] * 100),
                    fontsize=7.5, color=NAVY, xytext=(6, 4), textcoords="offset points")
    worst = prod[(prod["growth"] < 0)].nlargest(2, "revenue")
    for _, r in worst.iterrows():
        ax.annotate(r["description"], (r["revenue"] / 1000, r["growth"] * 100),
                    fontsize=7.5, color="#C0392B", xytext=(6, -10), textcoords="offset points")
    ax.set_xlabel("revenue ($k) — share proxy")
    ax.set_ylabel("H2 vs H1 growth (%)")
    ax.set_title("Product portfolio: revenue x growth, bubble = margin %\n"
                 "(top-right grows the business; big + declining is the watch list)",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "product_portfolio.png", dpi=130)


def plot_pareto(prod: pd.DataFrame):
    p = prod.sort_values("revenue", ascending=False).reset_index(drop=True)
    cum = p["revenue"].cumsum() / p["revenue"].sum()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    colors = p["abc_class"].map(ABC_COLORS)
    ax.bar(range(len(p)), p["revenue"] / 1000, color=colors, width=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(len(p)), cum * 100, color=ORANGE, lw=2)
    ax2.axhline(80, color=ORANGE, lw=0.8, ls="--")
    n_a = (p["abc_class"] == "A").sum()
    ax2.annotate(f"{n_a} SKUs = 80% of revenue", (n_a, 81), fontsize=9,
                 color=ORANGE, xytext=(10, 6), textcoords="offset points")
    ax.set_xlabel("SKUs, ranked by revenue")
    ax.set_ylabel("revenue ($k)")
    ax2.set_ylabel("cumulative %", color=ORANGE)
    ax2.set_ylim(0, 105)
    ax.set_title("ABC / Pareto: navy = A-class, teal = B, grey = C",
                 fontsize=11, fontweight="bold", color=NAVY, loc="left")
    ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "abc_pareto.png", dpi=130)


def main():
    sales = load_sales()
    prod = product_metrics(sales)
    prod.to_csv(OUT / "product_analytics.csv", index=False)
    plot_portfolio(prod)
    plot_pareto(prod)

    a = prod[prod["abc_class"] == "A"]
    print(f"SKUs analyzed: {len(prod)}")
    print(f"A-class: {len(a)} SKUs carry {a['revenue_share'].sum():.0%} of revenue")
    print(f"stickiest SKU: {prod.nlargest(1, 'repeat_purchase_rate').iloc[0]['description']} "
          f"({prod['repeat_purchase_rate'].max():.0%} repeat rate)")
    decliners = prod[(prod['abc_class'] == 'A') & (prod['growth'] < 0)]
    if not decliners.empty:
        print(f"watch list (A-class, declining): "
              f"{', '.join(decliners['description'].tolist())}")


if __name__ == "__main__":
    main()
