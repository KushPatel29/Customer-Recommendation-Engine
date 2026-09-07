"""
A waterfall that does not reconcile is worse than no waterfall.

It looks authoritative, it gets screenshotted into a board pack, and it is
wrong in a direction nobody can see. So the first section here walks the chart
arithmetically and demands it land on the number the ledger says, to the cent.

The rest defends the two claims the page is built on: that invoice revenue
still ties out to sales_lines.csv unchanged (this whole layer is additive, and
if it ever stopped being additive every other figure in the repo would move
silently), and that the terms really are independent of customer value - which
is what makes "your best terms are not going to your best customers" a
measurement rather than an assumption baked into the generator.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analytics"))

import pocket_margin as pm  # noqa: E402

DATA = ROOT / "data"
PUBLISHED = ROOT / "output" / "pocket_margin_summary.json"


@pytest.fixture(scope="module")
def built():
    return pm.build()


@pytest.fixture(scope="module")
def orders():
    return pm.orders(pm.load())


# --- the waterfall reconciles -----------------------------------------------

def test_the_waterfall_lands_where_the_ledger_says(built):
    """Walk every step and finish on pocket margin. This is the test the file
    exists for."""
    _, w, _, _, _ = built
    running = 0.0
    for _, r in w.iterrows():
        if r.kind in ("start", "subtotal", "total"):
            assert abs(running - r.amount) < 1.0 or r.kind == "start", (
                f"the walk reached {running:,.2f} but {r.step} says "
                f"{r.amount:,.2f}")
            running = r.amount
        else:
            running += r.amount
    assert w.iloc[-1].step == "Pocket margin"


def test_every_step_is_signed_the_way_a_waterfall_needs(built):
    _, w, _, _, _ = built
    assert (w[w.kind.isin(["start", "subtotal", "total"])].amount > 0).all()
    # Leaks are drawn downwards. The invoice price variance is the exception
    # and is allowed to run either way - on this book it is a small premium.
    leaks = w[(w.kind == "leak") & (w.step != "Invoice price variance")]
    assert (leaks.amount < 0).all()


def test_invoice_revenue_still_ties_to_the_untouched_ledger(orders):
    """The whole terms layer is additive. If invoice revenue ever stops
    matching sales_lines.csv, every other number in this repo moved without
    anybody deciding it should."""
    sales = pd.read_csv(DATA / "sales_lines.csv")
    assert orders.invoice_revenue.sum() == pytest.approx(
        sales.revenue.sum(), abs=1.0)
    assert orders.cogs.sum() == pytest.approx(sales.cost.sum(), abs=1.0)
    assert len(orders) == sales.order_id.nunique()


def test_pocket_revenue_is_invoice_minus_every_named_leak(orders):
    total = sum(orders[k] for k, _, _ in pm.LEAKS)
    assert np.allclose(orders.pocket_revenue,
                       orders.invoice_revenue - total, atol=0.02)
    assert (orders.pocket_revenue <= orders.invoice_revenue + 0.01).all()


def test_pocket_margin_is_never_quietly_better_than_invoice_margin(orders):
    assert (orders.pocket_margin <= orders.invoice_margin + 0.01).all()


def test_the_components_account_for_the_whole_leak(built, orders):
    _, _, _, comp, _ = built
    assert comp.value.sum() == pytest.approx(
        float(orders.off_invoice_total.sum()), abs=1.0)
    assert comp.share_of_leakage.sum() == pytest.approx(1.0, abs=1e-3)
    assert comp.cumulative_share.iloc[-1] == pytest.approx(1.0, abs=1e-3)
    assert comp.value.is_monotonic_decreasing


# --- the terms are independent of value -------------------------------------

def test_terms_are_not_a_restatement_of_customer_size(built):
    """The finding the page rests on. If leakage tracked revenue, "your best
    terms are not going to your best customers" would be an artefact of how the
    data was made rather than something the analysis found."""
    summary, _, c, _, _ = built
    assert abs(summary["corr_revenue_leakage"]) < 0.25, (
        "leakage correlates with revenue - the mismatch finding is circular")
    # ...and there really is a spread to find.
    assert summary["band_width_pts"] > 3.0
    assert c.leakage_pct.max() > c.leakage_pct.min() * 3


def test_a_customer_can_be_large_and_expensive_at_once(built):
    """Both quadrants have to be populated, or the mismatch column is
    describing a distinction this data cannot make."""
    _, _, c, _, _ = built
    big = c.invoice_revenue > c.invoice_revenue.median()
    rich = c.leakage_pct > c.leakage_pct.median()
    for label, mask in [("big and rich", big & rich),
                        ("big and cheap", big & ~rich),
                        ("small and rich", ~big & rich),
                        ("small and cheap", ~big & ~rich)]:
        assert mask.sum() > 0, f"no customer is {label}"


def test_the_mismatch_flag_means_what_the_page_says_it_means(built):
    _, _, c, _, _ = built
    flagged = c[c.terms_outrank_value == 1]
    assert len(flagged) > 0
    assert (flagged.terms_quartile == "richest").all()
    assert flagged.value_quartile.isin(["Q4 (lowest)", "Q3"]).all()


def test_the_opportunity_is_bounded_by_the_leak_it_comes_from(built):
    """A "return this much" number that exceeds the total leak is a modelling
    error dressed as an opportunity."""
    summary, _, _, _, _ = built
    assert 0 < summary["opportunity_to_median"] < summary["off_invoice_leakage"]


# --- cost to serve ----------------------------------------------------------

def test_the_bands_cover_every_order_exactly_once(built, orders):
    _, _, _, _, bands = built
    assert bands.orders.sum() == len(orders)
    assert bands.invoice_revenue.sum() == pytest.approx(
        float(orders.invoice_revenue.sum()), abs=1.0)
    assert bands.band_floor.is_monotonic_increasing


def test_freight_is_charged_on_weight_and_earned_on_value(built):
    """The mechanism behind the cost-to-serve finding: freight as a share of
    revenue has to fall as drops get larger, or the page is describing
    something else."""
    summary, _, _, _, bands = built
    assert bands.iloc[0].freight_pct_of_revenue \
        > bands.iloc[-1].freight_pct_of_revenue
    assert summary["smallest_band_freight_pct"] \
        > summary["largest_band_freight_pct"]


# --- what the README and the report quote -----------------------------------

def test_the_published_summary_is_what_this_code_produces(built):
    summary, _, _, _, _ = built
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    for key, value in summary.items():
        if isinstance(value, float):
            assert published[key] == pytest.approx(value, rel=1e-6), key
        else:
            assert published[key] == value, key
