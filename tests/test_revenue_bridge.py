"""
A price/volume/mix bridge is only worth reading if it reconciles exactly.

The three effects have to add to the change in the number they decompose - not
approximately, not "within rounding on a good day". If they do not, the chart
is attributing money to causes that did not produce it, and there is no way to
tell by looking. So the first test here adds them up and demands the cent.

The second thing worth defending is that the decomposition is genuinely
sensitive to the thing it claims to measure. A bridge that reports the same
mix effect whatever the mix does is a chart, not an analysis - so the tests
below construct periods where only volume changes, only price changes, and only
mix changes, and require each to land in its own term and nowhere else.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analytics"))

import revenue_bridge as rb  # noqa: E402

OUT = ROOT / "output"
PUBLISHED = OUT / "revenue_bridge_summary.json"


@pytest.fixture(scope="module")
def built():
    return rb.build()


def frame(rows):
    """(sku, pounds, price) triples into the shape decompose() reads."""
    return pd.DataFrame([{"sku": s, "quantity_lb": q, "revenue": q * p}
                         for s, q, p in rows])


# --- it reconciles ----------------------------------------------------------

def test_the_three_effects_add_to_the_change(built):
    _, bridge, _, _ = built
    for _, r in bridge.iterrows():
        assert r.volume + r.mix + r.rate == pytest.approx(r.change, abs=0.02), (
            f"{r.measure} bridge does not reconcile: "
            f"{r.volume:+,.2f} {r.mix:+,.2f} {r.rate:+,.2f} != {r.change:+,.2f}")
        assert r.current - r.prior == pytest.approx(r.change, abs=0.02)


def test_every_protein_bridge_reconciles_too(built):
    _, _, by_protein, _ = built
    assert len(by_protein) > 1
    for _, r in by_protein.iterrows():
        assert r.volume + r.mix + r.rate == pytest.approx(r.change, abs=0.02), \
            r.protein


def test_the_protein_bridges_sum_to_the_whole_book(built):
    """A cut that does not add back to the total is a cut of something else."""
    _, bridge, by_protein, _ = built
    total = bridge[bridge.measure == "Invoice revenue"].iloc[0]
    assert by_protein.change.sum() == pytest.approx(total.change, abs=1.0)
    assert by_protein.prior.sum() == pytest.approx(total.prior, abs=1.0)


# --- each effect is sensitive to its own cause ------------------------------

def test_pure_volume_growth_lands_entirely_in_volume():
    prior = frame([("A", 100, 10.0), ("B", 100, 20.0)])
    current = frame([("A", 150, 10.0), ("B", 150, 20.0)])
    d = rb.decompose(prior, current, "sku", "quantity_lb", "revenue")
    assert d["volume"] == pytest.approx(d["change"], abs=0.01)
    assert d["mix"] == pytest.approx(0.0, abs=0.01)
    assert d["rate"] == pytest.approx(0.0, abs=0.01)


def test_a_pure_price_rise_lands_entirely_in_rate():
    prior = frame([("A", 100, 10.0), ("B", 100, 20.0)])
    current = frame([("A", 100, 11.0), ("B", 100, 22.0)])
    d = rb.decompose(prior, current, "sku", "quantity_lb", "revenue")
    assert d["rate"] == pytest.approx(d["change"], abs=0.01)
    assert d["volume"] == pytest.approx(0.0, abs=0.01)
    assert d["mix"] == pytest.approx(0.0, abs=0.01)


def test_pounds_moving_to_dearer_product_lands_entirely_in_mix():
    """Same total pounds, same prices, different split - the textbook case the
    volume term must NOT absorb."""
    prior = frame([("A", 150, 10.0), ("B", 50, 20.0)])
    current = frame([("A", 50, 10.0), ("B", 150, 20.0)])
    d = rb.decompose(prior, current, "sku", "quantity_lb", "revenue")
    assert d["volume"] == pytest.approx(0.0, abs=0.01)
    assert d["rate"] == pytest.approx(0.0, abs=0.01)
    assert d["mix"] == pytest.approx(d["change"], abs=0.01)
    assert d["mix"] > 0, "moving pounds to the dearer SKU must raise revenue"


def test_a_sku_that_only_exists_in_one_period_is_not_dropped():
    """The usual bug in a hand-rolled bridge: an arrival or a discontinuation
    silently vanishes and the three effects quietly stop adding up."""
    prior = frame([("A", 100, 10.0)])
    current = frame([("A", 100, 10.0), ("NEW", 40, 15.0)])
    d = rb.decompose(prior, current, "sku", "quantity_lb", "revenue")
    assert d["change"] == pytest.approx(600.0, abs=0.01)
    assert d["volume"] + d["mix"] + d["rate"] == pytest.approx(
        d["change"], abs=0.01)


def test_an_unchanged_period_produces_no_effects_at_all():
    """The null case. A bridge that finds structure in nothing will find it
    everywhere."""
    same = frame([("A", 100, 10.0), ("B", 60, 20.0)])
    d = rb.decompose(same, same.copy(), "sku", "quantity_lb", "revenue")
    for k in ("change", "volume", "mix", "rate"):
        assert d[k] == pytest.approx(0.0, abs=0.01), k


# --- the pocket bridge ------------------------------------------------------

def test_the_pocket_bridge_is_not_a_copy_of_the_revenue_one(built):
    """If the two walked identically, allocating leakage down to lines achieved
    nothing and the second bridge is decoration."""
    _, bridge, _, _ = built
    rev = bridge[bridge.measure == "Invoice revenue"].iloc[0]
    pkt = bridge[bridge.measure == "Pocket margin"].iloc[0]
    assert pkt.prior < rev.prior
    assert abs(pkt.change - rev.change) > 1.0


def test_the_leakage_drift_is_measured_not_assumed(built):
    """The finding that the book drifted towards richer terms has to come from
    the two halves actually differing."""
    summary, _, _, _ = built
    assert summary["leakage_rate_prior"] != summary["leakage_rate_current"]
    assert summary["leakage_drift_pts"] == pytest.approx(
        (summary["leakage_rate_current"] - summary["leakage_rate_prior"]) * 100,
        abs=1e-3)


# --- concentration ----------------------------------------------------------

def test_shares_sum_to_one_and_the_ranking_is_a_ranking(built):
    _, _, _, conc = built
    assert conc.share.sum() == pytest.approx(1.0, abs=1e-4)
    assert conc.cumulative_share.iloc[-1] == pytest.approx(1.0, abs=1e-4)
    assert conc["rank"].tolist() == list(range(1, len(conc) + 1))
    assert conc.revenue.is_monotonic_decreasing


def test_hhi_matches_its_own_definition(built):
    summary, _, _, conc = built
    assert summary["hhi"] == pytest.approx(
        float(((conc.share * 100) ** 2).sum()), abs=0.5)
    # Bounds: 10,000 is a monopoly, 10,000/n is perfectly even.
    assert 10_000 / len(conc) <= summary["hhi"] <= 10_000
    assert summary["hhi_band"] in {b for _, b in rb.HHI_BANDS}


def test_the_hhi_band_is_the_published_scale_not_a_local_one(built):
    summary, _, _, _ = built
    expected = next(label for cap, label in rb.HHI_BANDS
                    if summary["hhi"] < cap)
    assert summary["hhi_band"] == expected


def test_top_n_shares_are_nested(built):
    summary, _, _, _ = built
    assert summary["top5_share"] <= summary["top10_share"] \
        <= summary["top20_share"] < 1.0


def test_revenue_at_risk_counts_flagged_accounts_not_a_guessed_probability(built):
    """Deliberately not revenue x an invented churn probability. If this ever
    becomes a weighted number, it has stopped being measurable."""
    summary, _, _, conc = built
    high = conc[conc.churn_risk == "High"]
    assert summary["revenue_at_risk"] == pytest.approx(
        float(high.revenue.sum()), abs=1.0)
    assert summary["customers_at_risk"] == len(high)
    assert 0 < summary["revenue_at_risk_pct"] < 1
    # Every customer is in exactly one band, and the bands are the ones
    # customer_analytics.py publishes.
    assert set(conc.churn_risk.unique()) <= {"Low", "Medium", "High"}


# --- what the README and the report quote -----------------------------------

def test_the_published_summary_is_what_this_code_produces(built):
    summary, _, _, _ = built
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    for key, value in summary.items():
        if isinstance(value, float):
            assert published[key] == pytest.approx(value, rel=1e-6), key
        else:
            assert published[key] == value, key
