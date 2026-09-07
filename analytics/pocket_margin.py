"""
Pocket margin: what the business is actually worth after the invoice is paid.

Every commercial dashboard in this repo, and most in the world, stops at
invoice revenue. That is the number the price list produces, the number the rep
is measured on, and the number that gets argued about in a pricing meeting. It
is not the number that reaches the bank.

Between the two sit the terms: the volume rebate accrued against the year, the
freight absorbed on a small drop, the co-op advertising allowance, the cash
discount taken for paying in ten days, the credit note for a short-dated
delivery. Individually each is a rounding error somebody signed off once.
Together they are the difference between a customer being worth serving and
not, and because none of them appear on an invoice line, they are almost never
measured per customer.

The pocket-price waterfall, and the band underneath it
-----------------------------------------------------
The waterfall walks list price down to pocket margin, one named leak at a
time. The band is the interesting half: the SPREAD of pocket margin across
customers who are all being sold the same catalogue at nearly the same invoice
price. A wide band is the finding. It means the differences between customers
are being set by contract terms nobody re-reads, not by the price list everyone
argues over.

What this module refuses to do
------------------------------
   * Blame a rep or a customer. Terms were agreed by the business, usually
     years ago, usually for a reason that made sense then. The output is where
     the money goes, not who to shout at.
   * Recommend cancelling terms. A rebate is part of a commercial relationship
     and withdrawing it has consequences this data cannot see. What it CAN say
     is which terms are not tracking the value they were granted for - which is
     a renegotiation agenda, not an instruction.
   * Model demand response. Nothing here knows what a customer would do if
     their freight policy changed. Every number is what the last twelve months
     actually settled at.

Outputs (output/):
    pocket_waterfall.csv        list -> invoice -> pocket -> margin, by step
    pocket_by_customer.csv      per customer: the full walk and the leakage rate
    leakage_by_component.csv    which term costs most, ranked on money
    freight_by_order_band.csv   cost to serve against drop size
    pocket_margin_summary.json  the headline figures the report and tests read

Usage:
    python analytics/pocket_margin.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

# The five off-invoice terms, in the order the waterfall walks them. Ordered
# largest-lever-first is tempting and wrong: a waterfall is a sequence of
# events, and reordering it to flatter the story makes the steps mean nothing.
LEAKS = [
    ("rebate_accrued", "Volume rebate",
     "Accrued against the year's purchases. Agreed once, paid quarterly, "
     "rarely revisited against the volume that was promised for it."),
    ("coop_allowance", "Co-op advertising",
     "Retail and deli accounts run promotions; the allowance funds them."),
    ("freight_absorbed", "Freight absorbed",
     "Charged on weight, not on value - which is why it lands hardest on small "
     "drops of cheap product."),
    ("cash_discount_taken", "Cash discount",
     "Taken by the customer for paying early. Free money to a customer with "
     "cash, and a real cost of capital to the seller."),
    ("credit_notes", "Credit notes",
     "Short-dated stock, damage, mis-picks. The only leak on this list that is "
     "a service failure rather than a commercial term."),
]

# Drop-size bands for the cost-to-serve cut. Dollars, not pounds: the question
# is whether the order was worth making, and freight is the cost side of it.
ORDER_BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2500),
               (2500, float("inf"))]


def band_label(lo, hi):
    if hi == float("inf"):
        return f"${lo:,.0f}+"
    return f"${lo:,.0f}-${hi:,.0f}"


def load() -> dict:
    return {
        "sales": pd.read_csv(DATA / "sales_lines.csv", parse_dates=["order_date"]),
        "catalog": pd.read_csv(DATA / "catalog.csv"),
        "customers": pd.read_csv(DATA / "customers.csv"),
        "terms": pd.read_csv(DATA / "customer_terms.csv"),
        "settlements": pd.read_csv(DATA / "settlements.csv",
                                   parse_dates=["order_date"]),
    }


def orders(d: dict) -> pd.DataFrame:
    """One row per order: list, invoice, every leak, pocket, and margin."""
    s = d["sales"]
    cat = d["catalog"].set_index("sku")
    s = s.assign(list_price=s.sku.map(cat.unit_price),
                 list_revenue=lambda x: x.quantity_lb * x.sku.map(cat.unit_price))

    per_order = s.groupby("order_id").agg(
        customer_id=("customer_id", "first"),
        order_date=("order_date", "first"),
        list_revenue=("list_revenue", "sum"),
        invoice_revenue=("revenue", "sum"),
        cogs=("cost", "sum"),
        pounds=("quantity_lb", "sum"),
        lines=("sku", "size"))

    o = per_order.join(d["settlements"].set_index("order_id")[
        [k for k, _, _ in LEAKS]])
    o["invoice_discount"] = (o.list_revenue - o.invoice_revenue).round(2)
    o["off_invoice_total"] = o[[k for k, _, _ in LEAKS]].sum(axis=1).round(2)
    o["pocket_revenue"] = (o.invoice_revenue - o.off_invoice_total).round(2)
    o["pocket_margin"] = (o.pocket_revenue - o.cogs).round(2)
    o["invoice_margin"] = (o.invoice_revenue - o.cogs).round(2)
    o["pocket_margin_pct"] = (o.pocket_margin / o.invoice_revenue).round(4)
    o["leakage_pct"] = (o.off_invoice_total / o.invoice_revenue).round(4)
    return o.reset_index()


def waterfall(o: pd.DataFrame) -> pd.DataFrame:
    """The walk from list to pocket margin, as steps a chart can render."""
    rows = [{"step": "List revenue", "kind": "start",
             "amount": round(float(o.list_revenue.sum()), 2),
             "note": "The catalogue price applied to everything that shipped."}]
    rows.append({"step": "Invoice price variance", "kind": "leak",
                 "amount": -round(float(o.invoice_discount.sum()), 2),
                 "note": "Realised price against the catalogue, running both "
                         "ways. It is the only lever on this chart anyone "
                         "manages weekly, and on this book it nets to a small "
                         "premium - which is the finding, not a rounding "
                         "error: the money is not leaving through the door "
                         "everybody is watching."})
    rows.append({"step": "Invoice revenue", "kind": "subtotal",
                 "amount": round(float(o.invoice_revenue.sum()), 2),
                 "note": "Where every other report in this repo stops."})
    # The five off-invoice terms are ONE step here, not five. Individually they
    # run $15k to $51k against subtotals of $3.2M, so on a single linear axis
    # they render as five bars of nothing and the middle of the walk goes flat
    # - a chart that hides the very thing it was drawn to show. The breakdown
    # lives in leakage_by_component.csv, at a scale where it is legible.
    rows.append({"step": "Off-invoice terms", "kind": "leak",
                 "amount": -round(float(o.off_invoice_total.sum()), 2),
                 "note": "Volume rebate, co-op advertising, freight absorbed, "
                         "cash discount and credit notes. None of them appear "
                         "on an invoice line, which is why almost nobody "
                         "measures them per customer."})
    rows.append({"step": "Pocket revenue", "kind": "subtotal",
                 "amount": round(float(o.pocket_revenue.sum()), 2),
                 "note": "What actually reached the bank."})
    rows.append({"step": "Cost of goods", "kind": "leak",
                 "amount": -round(float(o.cogs.sum()), 2), "note": ""})
    rows.append({"step": "Pocket margin", "kind": "total",
                 "amount": round(float(o.pocket_margin.sum()), 2),
                 "note": "The number the business is actually run on."})
    w = pd.DataFrame(rows)
    w["order"] = range(1, len(w) + 1)
    return w


def by_customer(o: pd.DataFrame, d: dict) -> pd.DataFrame:
    g = o.groupby("customer_id")
    c = pd.DataFrame({
        "orders": g.size(),
        "pounds": g.pounds.sum().round(1),
        "list_revenue": g.list_revenue.sum().round(2),
        "invoice_revenue": g.invoice_revenue.sum().round(2),
        "off_invoice_total": g.off_invoice_total.sum().round(2),
        "pocket_revenue": g.pocket_revenue.sum().round(2),
        "cogs": g.cogs.sum().round(2),
        "invoice_margin": g.invoice_margin.sum().round(2),
        "pocket_margin": g.pocket_margin.sum().round(2),
    })
    for key, _, _ in LEAKS:
        c[key] = g[key].sum().round(2)
    c["leakage_pct"] = (c.off_invoice_total / c.invoice_revenue).round(4)
    c["invoice_margin_pct"] = (c.invoice_margin / c.invoice_revenue).round(4)
    c["pocket_margin_pct"] = (c.pocket_margin / c.invoice_revenue).round(4)
    c["avg_order_value"] = (c.invoice_revenue / c.orders).round(2)

    meta = d["customers"].set_index("customer_id")
    terms = d["terms"].set_index("customer_id")
    for col, src in [("customer_name", meta.customer_name),
                     ("persona", meta.persona), ("region", meta.region),
                     ("rep", meta.rep), ("rebate_pct", terms.rebate_pct),
                     ("freight_policy", terms.freight_policy),
                     ("payment_terms", terms.payment_terms),
                     ("coop_allowance_pct", terms.coop_allowance_pct)]:
        c[col] = c.index.map(src)

    # Quartiles on what the customer is worth, and on what its terms cost.
    # Both computed on the whole book so a filtered page cannot re-cut them.
    c["value_quartile"] = pd.qcut(c.pocket_margin, 4,
                                  labels=["Q4 (lowest)", "Q3", "Q2",
                                          "Q1 (highest)"]).astype(str)
    c["terms_quartile"] = pd.qcut(c.leakage_pct, 4,
                                  labels=["cheapest", "Q3", "Q2",
                                          "richest"]).astype(str)
    # The mismatch: rich terms on a low-value account. Not an accusation -
    # a renegotiation agenda.
    c["terms_outrank_value"] = ((c.terms_quartile == "richest")
                                & c.value_quartile.isin(["Q4 (lowest)", "Q3"])
                                ).astype(int)
    return c.reset_index().sort_values("pocket_margin", ascending=False)


def leakage_components(o: pd.DataFrame) -> pd.DataFrame:
    inv = float(o.invoice_revenue.sum())
    rows = []
    for key, label, note in LEAKS:
        v = float(o[key].sum())
        rows.append({"component": label, "value": round(v, 2),
                     "pct_of_invoice": round(v / inv, 5),
                     "orders_affected": int((o[key] > 0).sum()),
                     "note": note})
    out = pd.DataFrame(rows).sort_values("value", ascending=False)
    out["share_of_leakage"] = (out.value / out.value.sum()).round(4)
    out["cumulative_share"] = out.share_of_leakage.cumsum().round(4)
    return out.reset_index(drop=True)


def freight_by_band(o: pd.DataFrame) -> pd.DataFrame:
    """Cost to serve against drop size.

    Freight is charged on weight and earned on value, so the two diverge
    hardest on small orders of heavy, cheap product. This is the cut that turns
    'freight costs us $X' into a minimum-order-value conversation."""
    rows = []
    for lo, hi in ORDER_BANDS:
        band = o[(o.invoice_revenue >= lo) & (o.invoice_revenue < hi)]
        if band.empty:
            continue
        inv = float(band.invoice_revenue.sum())
        rows.append({
            "order_band": band_label(lo, hi),
            "band_floor": lo,
            "orders": int(len(band)),
            "invoice_revenue": round(inv, 2),
            "avg_order_value": round(inv / len(band), 2),
            "pounds": round(float(band.pounds.sum()), 1),
            "freight_absorbed": round(float(band.freight_absorbed.sum()), 2),
            "freight_pct_of_revenue": round(
                float(band.freight_absorbed.sum()) / inv, 5),
            "pocket_margin": round(float(band.pocket_margin.sum()), 2),
            "pocket_margin_pct": round(
                float(band.pocket_margin.sum()) / inv, 4),
        })
    return pd.DataFrame(rows)


def build() -> tuple:
    d = load()
    o = orders(d)
    w = waterfall(o)
    c = by_customer(o, d)
    comp = leakage_components(o)
    bands = freight_by_band(o)

    inv = float(o.invoice_revenue.sum())
    leak = float(o.off_invoice_total.sum())
    disc = float(o.invoice_discount.sum())
    mismatched = c[c.terms_outrank_value == 1]

    # What moving the worst-terms accounts to the median leakage rate would
    # return. Stated as an opportunity, not a plan: the terms exist for
    # reasons this data cannot see.
    median_leak = float(c.leakage_pct.median())
    above = c[c.leakage_pct > median_leak]
    to_median = float((above.invoice_revenue
                       * (above.leakage_pct - median_leak)).sum())

    worst_band = bands.iloc[0]
    best_band = bands.iloc[-1]

    summary = {
        "orders": int(len(o)),
        "customers": int(len(c)),
        "window_start": str(o.order_date.min().date()),
        "window_end": str(o.order_date.max().date()),
        "list_revenue": round(float(o.list_revenue.sum()), 2),
        "invoice_revenue": round(inv, 2),
        "invoice_discount": round(disc, 2),
        "invoice_discount_pct": round(disc / float(o.list_revenue.sum()), 5),
        "off_invoice_leakage": round(leak, 2),
        "off_invoice_leakage_pct": round(leak / inv, 5),
        # The headline contrast: the managed lever against the unmanaged one.
        "leakage_vs_price_variance_multiple": round(leak / abs(disc), 1)
        if abs(disc) > 1 else None,
        "pocket_revenue": round(float(o.pocket_revenue.sum()), 2),
        "invoice_margin": round(float(o.invoice_margin.sum()), 2),
        "invoice_margin_pct": round(float(o.invoice_margin.sum()) / inv, 4),
        "pocket_margin": round(float(o.pocket_margin.sum()), 2),
        "pocket_margin_pct": round(float(o.pocket_margin.sum()) / inv, 4),
        "margin_points_lost": round(
            (float(o.invoice_margin.sum()) - float(o.pocket_margin.sum()))
            / inv * 100, 2),
        "band_low_pct": round(float(c.pocket_margin_pct.quantile(0.05)), 4),
        "band_high_pct": round(float(c.pocket_margin_pct.quantile(0.95)), 4),
        "band_width_pts": round(
            (float(c.pocket_margin_pct.quantile(0.95))
             - float(c.pocket_margin_pct.quantile(0.05))) * 100, 1),
        "leakage_low_pct": round(float(c.leakage_pct.quantile(0.05)), 4),
        "leakage_high_pct": round(float(c.leakage_pct.quantile(0.95)), 4),
        # Near zero is the point: terms are not tracking customer value.
        "corr_revenue_leakage": round(
            float(c.invoice_revenue.corr(c.leakage_pct)), 3),
        "top_component": comp.iloc[0].component,
        "top_component_value": float(comp.iloc[0].value),
        "top_component_share": float(comp.iloc[0].share_of_leakage),
        "mismatched_customers": int(len(mismatched)),
        "mismatched_revenue": round(float(mismatched.invoice_revenue.sum()), 2),
        "mismatched_leakage": round(float(mismatched.off_invoice_total.sum()), 2),
        "opportunity_to_median": round(to_median, 2),
        "opportunity_pct_of_margin": round(
            to_median / float(o.pocket_margin.sum()), 4),
        "smallest_band": worst_band.order_band,
        "smallest_band_orders": int(worst_band.orders),
        "smallest_band_freight_pct": float(worst_band.freight_pct_of_revenue),
        "smallest_band_margin_pct": float(worst_band.pocket_margin_pct),
        "largest_band": best_band.order_band,
        "largest_band_freight_pct": float(best_band.freight_pct_of_revenue),
        "largest_band_margin_pct": float(best_band.pocket_margin_pct),
    }
    return summary, w, c, comp, bands


def headline(summary: dict) -> pd.DataFrame:
    """The summary as a one-row table for the report to read scalars from.
    One row cannot be double-counted by a roll-up that meets a total it did not
    expect."""
    scalars = {k: v for k, v in summary.items()
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    return pd.DataFrame([scalars])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, w, c, comp, bands = build()

    w.to_csv(OUT / "pocket_waterfall.csv", index=False)
    c.to_csv(OUT / "pocket_by_customer.csv", index=False)
    comp.to_csv(OUT / "leakage_by_component.csv", index=False)
    bands.to_csv(OUT / "freight_by_order_band.csv", index=False)
    (OUT / "pocket_margin_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    headline(summary).to_csv(OUT / "pocket_margin_headline.csv", index=False)

    print()
    print("=" * 74)
    print(f"POCKET MARGIN   ({summary['window_start']} to "
          f"{summary['window_end']})")
    print("=" * 74)
    print(f"  {summary['orders']:,} orders, {summary['customers']} customers")
    print()
    print(f"  {'step':22s} {'amount':>14s}  {'running':>14s}")
    running = 0.0
    for _, r in w.iterrows():
        if r.kind in ("start", "subtotal", "total"):
            running = r.amount
            print(f"  {r.step:22s} {'':>14s}  ${running:>13,.0f}"
                  f"{'  <- everything else in this repo stops here' if r.step == 'Invoice revenue' else ''}")
        else:
            running += r.amount
            print(f"  {r.step:22s} ${r.amount:>13,.0f}  ${running:>13,.0f}")
    print()
    print("THE LEVER THAT IS MANAGED, AND THE ONE THAT IS NOT")
    print("=" * 74)
    direction = "premium" if summary["invoice_discount"] < 0 else "discount"
    print(f"  Invoice price variance  ${abs(summary['invoice_discount']):>10,.0f}"
          f"   {abs(summary['invoice_discount_pct']):.2%} of list, a net "
          f"{direction}")
    print(f"  Off-invoice terms       ${summary['off_invoice_leakage']:>10,.0f}"
          f"   {summary['off_invoice_leakage_pct']:.2%} of invoice")
    print("  The lever that is governed weekly moved revenue by "
          f"{abs(summary['invoice_discount_pct']):.2%}. The one nobody re-reads")
    print(f"  moved it by {summary['off_invoice_leakage_pct']:.2%}.")
    print(f"  Margin falls from {summary['invoice_margin_pct']:.1%} on the "
          f"invoice to {summary['pocket_margin_pct']:.1%} in the bank "
          f"({summary['margin_points_lost']:.1f} points).")
    print()
    print("WHERE IT GOES")
    print("=" * 74)
    for _, r in comp.iterrows():
        print(f"    {r.component:22s} ${r.value:>10,.0f}   "
              f"{r.share_of_leakage:5.1%} of leakage   "
              f"{r.pct_of_invoice:5.2%} of invoice   "
              f"{r.orders_affected:,} orders")
    print()
    print("THE POCKET-PRICE BAND")
    print("=" * 74)
    print("  Same catalogue, near-identical invoice prices, and pocket margin "
          "runs from")
    print(f"  {summary['band_low_pct']:.1%} to {summary['band_high_pct']:.1%} "
          f"across customers - a {summary['band_width_pts']:.1f}-point band.")
    print(f"  Leakage against customer revenue correlates "
          f"{summary['corr_revenue_leakage']:+.3f}: the terms are not tracking")
    print("  the value they were granted for.")
    print(f"  {summary['mismatched_customers']} customers hold the richest "
          f"terms on the lowest value, covering")
    print(f"  ${summary['mismatched_revenue']:,.0f} of revenue and "
          f"${summary['mismatched_leakage']:,.0f} of leakage.")
    print(f"  Bringing every above-median account to the MEDIAN rate returns "
          f"${summary['opportunity_to_median']:,.0f}")
    print(f"  ({summary['opportunity_pct_of_margin']:.1%} of pocket margin) "
          "without touching a single price.")
    print()
    print("COST TO SERVE BY DROP SIZE")
    print("=" * 74)
    print(f"  {'order value':>16s} {'orders':>8s} {'freight %':>11s} "
          f"{'pocket margin %':>17s}")
    for _, r in bands.iterrows():
        print(f"  {r.order_band:>16s} {r.orders:>8,} "
              f"{r.freight_pct_of_revenue:>11.2%} {r.pocket_margin_pct:>17.1%}")
    print()
    print(f"  Freight is charged on weight and earned on value, so the "
          f"{summary['smallest_band']} band")
    print(f"  gives up {summary['smallest_band_freight_pct']:.2%} of revenue to "
          f"freight against {summary['largest_band_freight_pct']:.2%} on the "
          f"{summary['largest_band']} band.")
    print("  That is a minimum-drop conversation, not a pricing one.")
    print()
    print(f"wrote 6 files to {OUT}")


if __name__ == "__main__":
    main()
