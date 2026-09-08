"""The ranker's features must not have seen the purchases it is predicting.

`two_stage_hits` hid 25% of each customer's SKUs, trained on customer-disjoint
labels, and then read `output/product_analytics.csv` and
`output/customer_analytics.csv` for its features. Those are built from the FULL
sales file — the one containing the very purchases being predicted.

A customer's recency, frequency, monetary value, CLV and protein breadth all
move when you add a purchase, and so do a product's repeat rate, velocity and
revenue share. So the labels were clean and the features leaked, which is the
harder half to see: the split looks rigorous, and the score is still inflated.

These tests pin the property directly — features built from the held-out data
differ from features built without it, and the evaluation uses the latter.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation" / "evaluate_holdout.py"


def test_the_evaluation_does_not_read_full_data_analytics():
    """The CSVs are built from every purchase, including the hidden ones."""
    source = EVAL.read_text(encoding="utf-8")
    for leaked in ("output\" / \"product_analytics.csv", "output\" / \"customer_analytics.csv"):
        assert leaked not in source, (
            "the holdout evaluation reads analytics built on the full sales "
            "file, so the ranker's features encode the purchases it is scored on"
        )
    assert "_features_without_the_holdout" in source, (
        "features are no longer rebuilt without the held-out purchases"
    )


def test_features_are_rebuilt_from_data_with_the_holdout_removed():
    import sys
    sys.path.insert(0, str(ROOT))
    from evaluation.evaluate_holdout import _features_without_the_holdout

    sig = inspect.signature(_features_without_the_holdout)
    assert list(sig.parameters) == ["sales", "hidden_by_customer"]


def test_removing_a_purchase_actually_moves_the_features():
    """Otherwise the fix above would be theatre.

    If a customer's metrics were unchanged by hiding a quarter of their SKUs,
    there would have been nothing to leak. This proves the leak was real.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from evaluation.evaluate_holdout import _features_without_the_holdout
    from recommend import load_sales

    sales = load_sales()
    customer = sales["customer_id"].value_counts().index[0]
    owned = sorted(sales.loc[sales["customer_id"] == customer, "sku"].unique())
    if len(owned) < 4:
        pytest.skip("that customer has too few SKUs to hide a quarter of them")
    hidden = set(owned[: max(1, len(owned) // 4)])

    _, full = _features_without_the_holdout(sales, {})
    _, held = _features_without_the_holdout(sales, {customer: hidden})

    def row(frame):
        return frame.loc[frame["customer_id"] == customer].iloc[0]

    before, after = row(full), row(held)
    moved = [c for c in ("F", "M", "protein_breadth")
             if c in full.columns and before[c] != after[c]]
    assert moved, (
        "hiding a quarter of a customer's SKUs changed none of their features — "
        "either the holdout is not being removed, or these features carry no "
        "information about what was bought"
    )
