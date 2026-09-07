"""
Every number the README states in prose must still be the number the code
produces, formatted the way a reader sees it.

The suite already pins the published FIGURES: change the analysis and
`test_pocket_margin.py` and `test_revenue_bridge.py` fail. What nothing pinned
was the other direction — the prose. A figure can move, the JSON assertion can
be updated, and the README can go on quoting last month's run with a green
build the whole way. That is not hypothetical; it happened in a sibling
repository, and the test that caught it there is the model for this one.

So each assertion below reads the value out of the engine output and then looks
for it in the README **as formatted**: thousands separators, signs, percent
places and all. When a figure legitimately changes, this fails and names the
document that needs editing.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
OUT = ROOT / "output"


@pytest.fixture(scope="module")
def prose():
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pocket():
    return json.loads((OUT / "pocket_margin_summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridge():
    return json.loads((OUT / "revenue_bridge_summary.json").read_text(encoding="utf-8"))


def quoted(prose, needle):
    """Match across line breaks: prose wraps, and a figure that lands either
    side of a wrap is still quoted."""
    return " ".join(needle.split()) in " ".join(prose.split())


def test_the_readme_is_the_real_one(prose):
    """Guards against every assertion below passing on a stub."""
    assert len(prose) > 10_000
    assert "## What the invoice does not show" in prose
    assert "## Why revenue moved, and what it is standing on" in prose


def test_the_leak_total_is_quoted_as_published(pocket, prose):
    p = pocket
    assert quoted(prose, f"${p['off_invoice_leakage']:,.0f}")
    assert quoted(prose, f"{p['off_invoice_leakage_pct']:.2%} of invoice revenue")


def test_the_waterfall_table_is_quoted_as_published(pocket, prose):
    p = pocket
    for value in (p["list_revenue"], p["invoice_revenue"], p["pocket_revenue"],
                  p["pocket_margin"]):
        assert quoted(prose, f"${value:,.0f}"), value
    # Written the way a reader reads money: the sign in front of the currency,
    # and a typographic minus rather than a hyphen.
    assert quoted(prose, f"+${-p['invoice_discount']:,.0f}")
    assert quoted(prose, f"−${p['off_invoice_leakage']:,.0f}")


def test_the_managed_versus_unmanaged_contrast_is_quoted_as_published(pocket, prose):
    """The headline of the whole section. If the price lever ever becomes the
    bigger one, this sentence has to be rewritten rather than left standing."""
    p = pocket
    assert abs(p["invoice_discount_pct"]) < p["off_invoice_leakage_pct"]
    assert quoted(prose, f"moved revenue by {abs(p['invoice_discount_pct']):.2%}")
    assert quoted(prose, f"moved it by {p['off_invoice_leakage_pct']:.2%}")
    assert quoted(prose, f"{p['invoice_margin_pct']:.1%} on the invoice to "
                         f"{p['pocket_margin_pct']:.1%}")
    assert quoted(prose, f"{p['margin_points_lost']:.1f} points")
    assert quoted(prose, f'"{p["top_component"]}" alone is '
                         f"{p['top_component_share']:.0%}")


def test_the_pocket_price_band_is_quoted_as_published(pocket, prose):
    p = pocket
    assert quoted(prose, f"{p['customers']} customers")
    assert quoted(prose, f"{p['band_low_pct']:.1%} to {p['band_high_pct']:.1%}")
    assert quoted(prose, f"{p['band_width_pts']:.1f}-point band")
    assert quoted(prose, f"correlates {p['corr_revenue_leakage']:+.3f}")
    assert quoted(prose, f"{p['mismatched_customers']} accounts hold the richest terms")
    assert quoted(prose, f"${p['mismatched_revenue']:,.0f} of revenue")


def test_the_recoverable_opportunity_is_quoted_as_published(pocket, prose):
    p = pocket
    assert quoted(prose, f"${p['opportunity_to_median']:,.0f}")
    assert quoted(prose, f"{p['opportunity_pct_of_margin']:.1%} of pocket margin")


def test_the_cost_to_serve_figures_are_quoted_as_published(pocket, prose):
    p = pocket
    assert quoted(prose, f"The {p['smallest_band']} band")
    assert quoted(prose, f"{p['smallest_band_freight_pct']:.2%} of revenue to freight")
    assert quoted(prose, f"{p['largest_band_freight_pct']:.2%} on the "
                         f"{p['largest_band']} band")
    assert quoted(prose, f"{p['smallest_band_margin_pct']:.1%} pocket margin")
    assert quoted(prose, f"{p['largest_band_margin_pct']:.1%}")


def test_the_bridge_table_is_quoted_as_published(bridge, prose):
    b = bridge
    for value in (b["revenue_prior"], b["revenue_current"],
                  b["pocket_prior"], b["pocket_current"]):
        assert quoted(prose, f"${value:,.0f}"), value
    for value in (b["revenue_volume"], b["revenue_mix"], b["revenue_rate"],
                  b["pocket_volume"], b["pocket_mix"], b["pocket_rate"]):
        assert quoted(prose, f"${value:+,.0f}"), value


def test_the_bridge_reconciles_in_the_document_too(bridge, prose):
    """The table a reader can add up by hand has to add up."""
    b = bridge
    assert b["revenue_volume"] + b["revenue_mix"] + b["revenue_rate"] == \
        pytest.approx(b["revenue_change"], abs=0.05)
    assert b["pocket_volume"] + b["pocket_mix"] + b["pocket_rate"] == \
        pytest.approx(b["pocket_change"], abs=0.05)


def test_the_leakage_drift_finding_is_quoted_as_published(bridge, prose):
    b = bridge
    assert quoted(prose, f"{b['revenue_change_pct']:+.1%}")
    assert quoted(prose, f"{b['pocket_change_pct']:+.1%}")
    assert quoted(prose, f"{b['leakage_rate_prior']:.2%} to "
                         f"{b['leakage_rate_current']:.2%}")
    assert quoted(prose, f"({b['leakage_drift_pts']:+.3f} pts)")
    assert quoted(prose, f"{b['worst_protein']} gave up the most")
    assert quoted(prose, f"${b['worst_protein_change']:+,.0f}")


def test_the_concentration_figures_are_quoted_as_published(bridge, prose):
    b = bridge
    assert quoted(prose, f"{b['customers']} customers at an HHI of {b['hhi']:,.0f}")
    assert quoted(prose, b["hhi_band"])
    assert quoted(prose, f"{b['top10_share']:.1%}")
    assert quoted(prose, f"{b['customers_to_half_the_book']} customers make up")
    assert quoted(prose, f"({b['largest_customer']}) is "
                         f"{b['largest_customer_share']:.1%}")


def test_the_risk_figures_are_quoted_as_published(bridge, prose):
    b = bridge
    assert quoted(prose, f"{b['customers_at_risk']} accounts are already flagged High")
    assert quoted(prose, f"${b['revenue_at_risk']:,.0f}")
    assert quoted(prose, f"({b['revenue_at_risk_pct']:.1%} of the book)")
    assert quoted(prose, f"${b['revenue_watchlist']:,.0f} more on the Medium")
    assert quoted(prose, f"{b['top10_at_risk_customers']} of the top ten")
    assert quoted(prose, f"${b['top10_at_risk_revenue']:,.0f}")
    assert quoted(prose, f"{b['worst_overdue_ratio']:.1f}× past")
