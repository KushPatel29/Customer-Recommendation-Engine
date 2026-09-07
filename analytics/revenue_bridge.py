"""
Why revenue moved, and how much of it is standing on one customer.

Two questions every commercial review opens with, and two that a growth
percentage cannot answer.

1. WHY DID REVENUE MOVE?
   "Revenue is up 4%" is not a finding, it is a prompt. Up because prices rose,
   because volume rose, or because the mix drifted towards dearer product are
   three different businesses with three different responses, and they can
   easily point in opposite directions inside the same 4%. The decomposition
   here is exact:

       volume = (Q1 - Q0) * r0_bar          same average price, more pounds
       mix    = SUM((q1_i - Q1 * s0_i) * r0_i)   pounds moving between SKUs
       rate   = SUM(q1_i * (r1_i - r0_i))        price movement on what shipped

   where q is pounds by SKU, r is realised price per pound, s0 is the prior
   period's share of pounds, and r0_bar is the prior period's average price.
   The three add to the revenue change exactly - a test asserts it to the cent,
   because a bridge that does not reconcile is worse than no bridge: it looks
   authoritative and is wrong in a direction nobody can see.

   The same walk is run on POCKET margin as well as invoice revenue, because
   the two can disagree - and when they do, the business grew its way into
   less money, which is precisely the case a revenue bridge alone will never
   show you.

2. HOW CONCENTRATED IS IT?
   The Herfindahl-Hirschman index on customer revenue share, the top-N shares,
   and how much of the book sits behind an account that has already gone quiet.
   A book that is 40% top-ten and stable is a different risk from one that is
   40% top-ten with two of those ten overdue, and only the second framing tells
   you which. Revenue at risk here is revenue behind accounts customer_
   analytics.py has flagged High against their OWN reorder rhythm - not revenue
   multiplied by a churn probability, because no such probability exists in
   this data and inventing one would put a decimal point on a guess.

What this does NOT do
---------------------
   * Forecast. revenue_forecast.py does that, and this deliberately looks
     backwards at what already happened.
   * Split price into "list change" and "discount change". The invoice price
     variance on this book is 0.08% of list, so that split would be reporting
     noise with three decimal places on it. pocket_margin.py has the lever that
     actually moves.

Outputs (output/):
    revenue_bridge.csv          the walk, on invoice revenue and pocket margin
    revenue_bridge_by_protein.csv  the same walk cut by protein category
    customer_concentration.csv  per customer: share, cumulative share, risk
    revenue_bridge_summary.json the headline figures the report and tests read

Usage:
    python analytics/revenue_bridge.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

# The book splits into two halves of the trading year. Halves rather than
# quarters because 38 SKUs over 13 weeks is thin enough that a mix effect would
# be mostly sampling noise, and a bridge built on noise reads as insight.
SPLIT_MONTHS = 6

# HHI thresholds the competition authorities use, borrowed here for the same
# reason a commercial team borrows them: they are a published scale rather than
# one invented to make this book look diversified.
HHI_BANDS = [(1500, "unconcentrated"), (2500, "moderately concentrated"),
             (float("inf"), "highly concentrated")]

TOP_N = [5, 10, 20]


def load() -> dict:
    return {
        "sales": pd.read_csv(DATA / "sales_lines.csv", parse_dates=["order_date"]),
        "customers": pd.read_csv(DATA / "customers.csv"),
        "pocket": pd.read_csv(OUT / "pocket_by_customer.csv"),
        "analytics": pd.read_csv(OUT / "customer_analytics.csv"),
    }


def halves(sales: pd.DataFrame) -> tuple:
    """(prior, current) - the first and second six months of the book."""
    cutoff = (sales.order_date.max()
              - pd.DateOffset(months=SPLIT_MONTHS)).normalize()
    return sales[sales.order_date <= cutoff], sales[sales.order_date > cutoff]


def decompose(prior: pd.DataFrame, current: pd.DataFrame, key: str,
              qty: str, value: str) -> dict:
    """Price / volume / mix, reconciling exactly to the change in `value`.

    Products present in only one period are handled by the mix and rate terms
    naturally: a new SKU has r0 = the period's average price, so its arrival
    shows up as mix rather than being silently dropped, which is the usual bug
    in a hand-rolled bridge."""
    q0 = prior.groupby(key)[qty].sum()
    q1 = current.groupby(key)[qty].sum()
    v0 = prior.groupby(key)[value].sum()
    v1 = current.groupby(key)[value].sum()

    idx = q0.index.union(q1.index)
    q0, q1 = q0.reindex(idx, fill_value=0.0), q1.reindex(idx, fill_value=0.0)
    v0, v1 = v0.reindex(idx, fill_value=0.0), v1.reindex(idx, fill_value=0.0)

    Q0, Q1 = float(q0.sum()), float(q1.sum())
    V0, V1 = float(v0.sum()), float(v1.sum())
    r0_bar = V0 / Q0 if Q0 else 0.0

    # A product with no prior volume has no prior price; the period average is
    # the only defensible stand-in, and it puts the arrival in the mix term
    # where it belongs.
    r0 = np.where(q0 > 0, v0 / q0.replace(0, np.nan), r0_bar)
    r1 = np.where(q1 > 0, v1 / q1.replace(0, np.nan), 0.0)
    r0 = np.nan_to_num(r0, nan=r0_bar)
    r1 = np.nan_to_num(r1, nan=0.0)
    s0 = (q0 / Q0).values if Q0 else np.zeros(len(idx))

    volume = (Q1 - Q0) * r0_bar
    mix = float(((q1.values - Q1 * s0) * r0).sum())
    rate = float((q1.values * (r1 - r0)).sum())
    return {
        "prior": round(V0, 2), "current": round(V1, 2),
        "change": round(V1 - V0, 2),
        "volume": round(float(volume), 2),
        "mix": round(mix, 2),
        "rate": round(rate, 2),
        "prior_qty": round(Q0, 1), "current_qty": round(Q1, 1),
        "prior_avg_price": round(r0_bar, 4),
        "current_avg_price": round(V1 / Q1, 4) if Q1 else 0.0,
    }


def pocket_lines(sales: pd.DataFrame, pocket: pd.DataFrame) -> pd.DataFrame:
    """Push each customer's realised leakage rate back down onto their lines.

    Leakage is settled per ORDER and cannot be observed per line, so the line
    carries its customer's rate. That is an allocation, not a measurement, and
    it is stated as one - but it is the only way to ask 'did the mix shift
    towards customers whose terms cost us more', which is the question the
    pocket bridge exists for."""
    rate = pocket.set_index("customer_id").leakage_pct
    s = sales.copy()
    s["leakage_rate"] = s.customer_id.map(rate).fillna(0.0)
    s["pocket_revenue"] = s.revenue * (1 - s.leakage_rate)
    s["pocket_margin"] = s.pocket_revenue - s.cost
    return s


def concentration(sales: pd.DataFrame, d: dict) -> tuple:
    rev = sales.groupby("customer_id").revenue.sum().sort_values(ascending=False)
    share = rev / rev.sum()
    c = pd.DataFrame({
        "revenue": rev.round(2),
        "share": share.round(5),
        "cumulative_share": share.cumsum().round(5),
    })
    c["rank"] = range(1, len(c) + 1)

    meta = d["customers"].set_index("customer_id")
    ana = d["analytics"].set_index("customer_id")
    pkt = d["pocket"].set_index("customer_id")
    c["customer_name"] = c.index.map(meta.customer_name)
    c["region"] = c.index.map(meta.region)
    c["rep"] = c.index.map(meta.rep)
    c["persona"] = c.index.map(meta.persona)
    c["churn_risk"] = c.index.map(ana.churn_risk)
    c["rfm_segment"] = c.index.map(ana.rfm_segment)
    c["days_overdue"] = c.index.map(ana.days_overdue)
    c["median_reorder_days"] = c.index.map(ana.median_reorder_days)
    c["pocket_margin"] = c.index.map(pkt.pocket_margin)
    c["leakage_pct"] = c.index.map(pkt.leakage_pct)
    # Revenue sitting behind a customer already flagged High by
    # customer_analytics.py - which measures each account against its OWN
    # reorder rhythm rather than a common cutoff. Deliberately NOT revenue
    # multiplied by an invented churn probability: there is no probability in
    # this data, and manufacturing one would put a decimal point on a guess.
    c["revenue_at_risk"] = np.where(c.churn_risk == "High", c.revenue, 0.0).round(2)
    c["revenue_watchlist"] = np.where(c.churn_risk == "Medium", c.revenue,
                                      0.0).round(2)
    # How far past its own rhythm the account has slipped: 2.0 means it has now
    # been silent for twice its normal gap between orders.
    c["overdue_ratio"] = (c.days_overdue / c.median_reorder_days).round(2)

    # HHI on percentage shares, the convention the 1500/2500 bands are set for.
    hhi = float(((share * 100) ** 2).sum())
    band = next(label for cap, label in HHI_BANDS if hhi < cap)
    return c.reset_index(), hhi, band


def build() -> tuple:
    d = load()
    sales = pocket_lines(d["sales"], d["pocket"])
    prior, current = halves(sales)

    bridges = []
    for label, value in [("Invoice revenue", "revenue"),
                         ("Pocket margin", "pocket_margin")]:
        row = decompose(prior, current, "sku", "quantity_lb", value)
        row["measure"] = label
        bridges.append(row)
    bridge = pd.DataFrame(bridges)[
        ["measure", "prior", "current", "change", "volume", "mix", "rate",
         "prior_qty", "current_qty", "prior_avg_price", "current_avg_price"]]

    by_protein = []
    for protein, grp in sales.groupby("protein"):
        p, c = halves(grp)
        if p.empty or c.empty:
            continue
        row = decompose(p, c, "sku", "quantity_lb", "revenue")
        row["protein"] = protein
        by_protein.append(row)
    by_protein = pd.DataFrame(by_protein)[
        ["protein", "prior", "current", "change", "volume", "mix", "rate"]] \
        .sort_values("change")

    conc, hhi, band = concentration(sales, d)

    # Did the growth land on customers whose terms cost more? The weighted
    # leakage rate across the two halves answers it directly: same catalogue,
    # same prices, a book drifting towards richer terms.
    leak_prior = float((prior.revenue * prior.leakage_rate).sum()
                       / prior.revenue.sum())
    leak_current = float((current.revenue * current.leakage_rate).sum()
                         / current.revenue.sum())

    rev_row = bridge[bridge.measure == "Invoice revenue"].iloc[0]
    pkt_row = bridge[bridge.measure == "Pocket margin"].iloc[0]
    top = {n: float(conc.head(n).share.sum()) for n in TOP_N}
    risky_top10 = conc.head(10)[conc.head(10).churn_risk == "High"]

    summary = {
        "window_start": str(sales.order_date.min().date()),
        "window_end": str(sales.order_date.max().date()),
        "split_at": str(halves(sales)[0].order_date.max().date()),
        "revenue_prior": float(rev_row.prior),
        "revenue_current": float(rev_row.current),
        "revenue_change": float(rev_row.change),
        "revenue_change_pct": round(float(rev_row.change / rev_row.prior), 4),
        "revenue_volume": float(rev_row.volume),
        "revenue_mix": float(rev_row.mix),
        "revenue_rate": float(rev_row.rate),
        "pocket_prior": float(pkt_row.prior),
        "pocket_current": float(pkt_row.current),
        "pocket_change": float(pkt_row.change),
        "pocket_change_pct": round(float(pkt_row.change / pkt_row.prior), 4),
        "pocket_volume": float(pkt_row.volume),
        "pocket_mix": float(pkt_row.mix),
        "pocket_rate": float(pkt_row.rate),
        "leakage_rate_prior": round(leak_prior, 5),
        "leakage_rate_current": round(leak_current, 5),
        "leakage_drift_pts": round((leak_current - leak_prior) * 100, 3),
        "largest_revenue_effect": max(
            [("volume", rev_row.volume), ("mix", rev_row.mix),
             ("rate", rev_row.rate)], key=lambda kv: abs(kv[1]))[0],
        "worst_protein": by_protein.iloc[0].protein,
        "worst_protein_change": float(by_protein.iloc[0].change),
        "best_protein": by_protein.iloc[-1].protein,
        "best_protein_change": float(by_protein.iloc[-1].change),
        "customers": int(len(conc)),
        "hhi": round(hhi, 1),
        "hhi_band": band,
        "top5_share": round(top[5], 4),
        "top10_share": round(top[10], 4),
        "top20_share": round(top[20], 4),
        "customers_to_half_the_book": int(
            (conc.cumulative_share < 0.5).sum() + 1),
        "revenue_at_risk": round(float(conc.revenue_at_risk.sum()), 2),
        "revenue_at_risk_pct": round(
            float(conc.revenue_at_risk.sum() / conc.revenue.sum()), 4),
        "revenue_watchlist": round(float(conc.revenue_watchlist.sum()), 2),
        "customers_at_risk": int((conc.churn_risk == "High").sum()),
        "worst_overdue_ratio": round(float(conc.overdue_ratio.max()), 2),
        "top10_at_risk_customers": int(len(risky_top10)),
        "top10_at_risk_revenue": round(float(risky_top10.revenue.sum()), 2),
        "largest_customer": conc.iloc[0].customer_name,
        "largest_customer_share": float(conc.iloc[0].share),
    }
    return summary, bridge, by_protein, conc


def headline(summary: dict) -> pd.DataFrame:
    scalars = {k: v for k, v in summary.items()
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return pd.DataFrame([scalars])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, bridge, by_protein, conc = build()

    bridge.to_csv(OUT / "revenue_bridge.csv", index=False)
    by_protein.to_csv(OUT / "revenue_bridge_by_protein.csv", index=False)
    conc.to_csv(OUT / "customer_concentration.csv", index=False)
    (OUT / "revenue_bridge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    headline(summary).to_csv(OUT / "revenue_bridge_headline.csv", index=False)

    print()
    print("=" * 74)
    print(f"WHY REVENUE MOVED   (first half to {summary['split_at']}, "
          f"then to {summary['window_end']})")
    print("=" * 74)
    for _, r in bridge.iterrows():
        print(f"  {r.measure}")
        print(f"    {'prior':>10s} ${r.prior:>12,.0f}")
        print(f"    {'volume':>10s} ${r.volume:>+12,.0f}   "
              f"{r.prior_qty:,.0f} lb -> {r.current_qty:,.0f} lb at the prior "
              f"average price")
        print(f"    {'mix':>10s} ${r.mix:>+12,.0f}   pounds moving between SKUs")
        print(f"    {'rate':>10s} ${r.rate:>+12,.0f}   price on what actually "
              f"shipped")
        print(f"    {'current':>10s} ${r.current:>12,.0f}   "
              f"({r.change:+,.0f}, {r.change / r.prior:+.1%})")
        print()
    same = np.sign(summary["revenue_change"]) == np.sign(summary["pocket_change"])
    if not same:
        print("  Revenue and pocket margin moved in OPPOSITE directions: the")
        print("  business grew its way into less money. A revenue bridge on its")
        print("  own would have reported the good half and stopped.")
    else:
        print(f"  Revenue moved {summary['revenue_change_pct']:+.1%} and pocket "
              f"margin {summary['pocket_change_pct']:+.1%} - the same direction, "
              "but not")
        print("  the same distance. The weighted cost of off-invoice terms went "
              "from")
        print(f"  {summary['leakage_rate_prior']:.2%} to "
              f"{summary['leakage_rate_current']:.2%} of revenue "
              f"({summary['leakage_drift_pts']:+.3f} pts): the book drifted "
              "towards")
        print("  accounts whose terms cost more, and no revenue report would "
              "have said so.")
    print()
    print("BY PROTEIN")
    print("=" * 74)
    print(f"  {'protein':12s} {'change':>12s} {'volume':>12s} {'mix':>12s} "
          f"{'rate':>12s}")
    for _, r in by_protein.iterrows():
        print(f"  {r.protein:12s} ${r.change:>+11,.0f} ${r.volume:>+11,.0f} "
              f"${r.mix:>+11,.0f} ${r.rate:>+11,.0f}")
    print()
    print("HOW MUCH OF THE BOOK STANDS ON HOW FEW CUSTOMERS")
    print("=" * 74)
    print(f"  {summary['customers']} customers, HHI {summary['hhi']:,.0f} "
          f"({summary['hhi_band']})")
    print(f"  top 5  {summary['top5_share']:.1%}"
          f"    top 10 {summary['top10_share']:.1%}"
          f"    top 20 {summary['top20_share']:.1%}")
    print(f"  {summary['customers_to_half_the_book']} customers make up half "
          "the book.")
    print(f"  Largest single account: {summary['largest_customer']} at "
          f"{summary['largest_customer_share']:.1%}.")
    print()
    print(f"  {summary['customers_at_risk']} customers are already flagged High "
          f"risk - measured against their")
    print(f"  own reorder rhythm, not a common cutoff - carrying "
          f"${summary['revenue_at_risk']:,.0f} "
          f"({summary['revenue_at_risk_pct']:.1%} of the book),")
    print(f"  with ${summary['revenue_watchlist']:,.0f} more on the Medium "
          "watchlist behind them.")
    print(f"  {summary['top10_at_risk_customers']} of the top ten are in that "
          f"High group, carrying ${summary['top10_at_risk_revenue']:,.0f}.")
    print(f"  The worst account is {summary['worst_overdue_ratio']:.1f}x past "
          "its own normal gap between orders.")
    print()
    print(f"wrote 5 files to {OUT}")


if __name__ == "__main__":
    main()
