"""
The commercial terms behind the invoice, and what they actually settled at.

sales_lines.csv is an invoice ledger: what was sold, at what price, at what
cost. It is complete and it is also the half of commercial reality that gets
managed. The other half is everything that happens AFTER the invoice - the
volume rebate paid at year end, the freight absorbed on small drops, the
co-op advertising allowance, the cash discount taken for paying in ten days,
the credit note for a short-dated delivery. None of it appears on the invoice
line, and all of it is margin.

That is the whole reason a pocket-price analysis exists. Invoice price is
governed, argued over and reported weekly. Off-invoice leakage is agreed once,
buried in a contract, and never looked at again - so the spread between what
two customers pay on paper and what they are worth in the bank is far wider
than anyone in the room believes.

Additive by construction
------------------------
Nothing here touches sales_lines.csv, customers.csv or catalog.csv. Those files
stay byte-identical, every existing figure in this repo keeps its value, and
these two new files decorate them. Invoice revenue still reconciles exactly to
what it always did; pocket revenue is a new, lower number underneath it.

Deliberately misaligned
-----------------------
Terms are NOT drawn in proportion to customer value. A customer's rebate rate,
freight policy and payment terms are set by when they signed, who signed them,
and how hard they pushed - which is exactly how it works in practice, and why
"our best terms are not going to our best customers" is a finding a commercial
review can produce rather than a foregone conclusion. If terms tracked value,
this analysis would have nothing to say and would still look convincing.

Synthetic only. Fixed seed.

Usage:
    python data_generator/generate_commercial_terms.py
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEED = 20260910

# Freight policies and what they cost the seller per pound shipped. "Prepaid"
# means the seller eats it; "collect" means the customer does.
FREIGHT_POLICIES = [
    ("prepaid", 0.42),          # seller absorbs the lane
    ("prepaid over 500lb", 0.24),
    ("collect", 0.0),
]

# Cash-discount terms: (label, discount if paid early, share of invoices on
# which the customer actually takes it).
PAYMENT_TERMS = [
    ("net 30", 0.000, 0.00),
    ("2/10 net 30", 0.020, 0.72),
    ("1/15 net 45", 0.010, 0.55),
    ("net 60", 0.000, 0.00),
]

# Credit notes: short-dated stock, a damaged case, a mis-pick. Rare per order,
# expensive when they happen.
CREDIT_RATE = 0.035          # share of orders drawing a credit
CREDIT_SHARE = (0.05, 0.35)  # share of that order's value credited


def customer_terms(customers, rng):
    """One row per customer: the deal behind the price list."""
    rows = []
    for c in customers:
        # Rebate rate is drawn independently of anything about the customer's
        # size or margin - see the module docstring. Some of the largest
        # rebates land on small accounts, which is the point.
        rebate = round(rng.choice([0.0, 0.0, 0.01, 0.015, 0.02, 0.03, 0.045]), 4)
        commitment = rng.choice([0, 0, 5_000, 12_000, 25_000, 40_000])
        freight, freight_rate = FREIGHT_POLICIES[
            rng.choices(range(3), weights=[0.34, 0.38, 0.28])[0]]
        terms, cash_pct, take_rate = PAYMENT_TERMS[
            rng.choices(range(4), weights=[0.30, 0.34, 0.20, 0.16])[0]]
        # Co-op advertising: retailers and delis run promotions, restaurants
        # do not, so this one IS driven by the customer - it is the exception
        # that makes the others visibly independent.
        coop = round(rng.uniform(0.005, 0.025), 4) \
            if c["persona"] in ("grocery", "charcuterie") and rng.random() < 0.55 \
            else 0.0
        rows.append({
            "customer_id": c["customer_id"],
            "rebate_pct": rebate,
            "annual_commitment_lb": commitment,
            "freight_policy": freight,
            "freight_cost_per_lb": freight_rate,
            "payment_terms": terms,
            "cash_discount_pct": cash_pct,
            "cash_discount_take_rate": take_rate,
            "coop_allowance_pct": coop,
        })
    return rows


def settlements(sales, terms_by_customer, rng):
    """One row per ORDER: what came off the invoice after it was raised.

    Order level rather than line level on purpose. A rebate accrues on the
    order's value, freight is charged on the drop, and a credit note is raised
    against the order - none of them are line-level events, and splitting them
    down to lines would invent a precision that the settlement process does not
    have."""
    orders = {}
    for r in sales:
        o = orders.setdefault(r["order_id"], {
            "order_id": r["order_id"],
            "order_date": r["order_date"],
            "customer_id": r["customer_id"],
            "invoice_revenue": 0.0,
            "pounds": 0.0,
        })
        o["invoice_revenue"] += float(r["revenue"])
        o["pounds"] += float(r["quantity_lb"])

    rows = []
    for o in orders.values():
        t = terms_by_customer[o["customer_id"]]
        rev = o["invoice_revenue"]

        rebate = rev * t["rebate_pct"]
        coop = rev * t["coop_allowance_pct"]
        # Freight is charged on weight, and the "over 500lb" policy only bites
        # on drops that clear the threshold - which is where the small-order
        # leakage on this data actually comes from.
        if t["freight_policy"] == "prepaid":
            freight = o["pounds"] * t["freight_cost_per_lb"]
        elif t["freight_policy"] == "prepaid over 500lb" and o["pounds"] >= 500:
            freight = o["pounds"] * t["freight_cost_per_lb"]
        else:
            freight = 0.0
        cash = (rev * t["cash_discount_pct"]
                if rng.random() < t["cash_discount_take_rate"] else 0.0)
        credit = (rev * rng.uniform(*CREDIT_SHARE)
                  if rng.random() < CREDIT_RATE else 0.0)

        rows.append({
            "order_id": o["order_id"],
            "order_date": o["order_date"],
            "customer_id": o["customer_id"],
            "invoice_revenue": round(rev, 2),
            "pounds_shipped": round(o["pounds"], 2),
            "rebate_accrued": round(rebate, 2),
            "coop_allowance": round(coop, 2),
            "freight_absorbed": round(freight, 2),
            "cash_discount_taken": round(cash, 2),
            "credit_notes": round(credit, 2),
        })
    rows.sort(key=lambda r: r["order_id"])
    return rows


def main():
    rng = random.Random(SEED)
    customers = list(csv.DictReader(
        (DATA / "customers.csv").open(encoding="utf-8")))
    sales = list(csv.DictReader(
        (DATA / "sales_lines.csv").open(encoding="utf-8")))

    terms = customer_terms(customers, rng)
    by_customer = {t["customer_id"]: t for t in terms}
    setts = settlements(sales, by_customer, rng)

    for name, rows in [("customer_terms.csv", terms),
                       ("settlements.csv", setts)]:
        path = DATA / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {path}  ({len(rows):,} rows)")

    leak = sum(r["rebate_accrued"] + r["coop_allowance"] + r["freight_absorbed"]
               + r["cash_discount_taken"] + r["credit_notes"] for r in setts)
    inv = sum(r["invoice_revenue"] for r in setts)
    print()
    print(f"invoice revenue ${inv:,.0f}; ${leak:,.0f} of it "
          f"({leak / inv:.1%}) never reaches the bank.")


if __name__ == "__main__":
    main()
