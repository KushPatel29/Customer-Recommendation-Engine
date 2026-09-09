"""The engine must not lean on pandas behaviour that pandas has already retired.

`build_customer_sku_matrix` passed `fill_value=0.0` into a pivot over an int64
column. Pandas 3 warns; pandas 4 raises. requirements.txt pins `pandas>=2.0`
with no ceiling, and this function sits under six call sites including
`evaluate_holdout`, which produces the 84.9% collaborative-filtering figure the
README leads with -- so a `pip install -r requirements.txt` after pandas 4 ships
would have turned the headline claim into a stack trace.

Rather than pin the one line, this walks the whole recommendation path with
pandas' own change-warnings promoted to errors. Anything the library intends to
break next fails here first, while it is still only a warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from engine.recommend import (
    build_customer_sku_matrix,
    cross_sell_recommendations,
    customer_similarity,
    growth_targets,
    load_sales,
    sku_affinity,
    top_customers_by_region,
)

#: Pandas 3 groups "this will change in a future version" under one base class.
#: On pandas 2 that class does not exist and the same messages arrive as plain
#: FutureWarning/DeprecationWarning, so fall back to those.
CHANGE_WARNINGS = tuple(
    filter(None, [getattr(pd.errors, "PandasChangeWarning", None)])
) or (FutureWarning, DeprecationWarning)


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    return load_sales()


def _run_the_engine(sales: pd.DataFrame) -> None:
    top = top_customers_by_region(sales)
    growth_targets(sales, top)
    matrix = build_customer_sku_matrix(sales)
    sim = customer_similarity(matrix)
    cross_sell_recommendations(sales, matrix=matrix, sim=sim)
    sku_affinity(sales)


def test_the_engine_runs_without_a_pandas_change_warning(sales):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for category in CHANGE_WARNINGS:
            warnings.simplefilter("error", category)
        _run_the_engine(sales)


def test_that_check_can_actually_fail(sales):
    """A promoted-warning test that promotes nothing passes on anything.

    This reproduces the exact call that was wrong -- a float fill_value over an
    int64 column -- and demands the filter above turn it into an exception.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for category in CHANGE_WARNINGS:
            warnings.simplefilter("error", category)
        assert sales["quantity_lb"].dtype == np.int64, (
            "quantity_lb is no longer an integer column, so the fill_value this "
            "test guards is no longer a mismatch -- rewrite the check"
        )
        with pytest.raises(CHANGE_WARNINGS):
            sales.pivot_table(
                index="customer_id", columns="sku",
                values="quantity_lb", aggfunc="sum", fill_value=0.0,
            )


def test_the_matrix_is_unchanged_by_the_integer_fill(sales):
    """The fix must be a no-op on the numbers, not a quiet re-scaling.

    Both fills produce float64 after np.log1p, so the matrices have to be
    exactly equal -- not merely close.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = np.log1p(sales.pivot_table(
            index="customer_id", columns="sku",
            values="quantity_lb", aggfunc="sum", fill_value=0.0,
        ))
    assert build_customer_sku_matrix(sales).equals(legacy)


def test_zero_means_never_bought(sales):
    """What fill_value is actually for, pinned so a future edit cannot drop it."""
    matrix = build_customer_sku_matrix(sales)
    assert matrix.notna().all().all(), "a missing pair should be filled, not NaN"
    bought = set(map(tuple, sales[["customer_id", "sku"]].drop_duplicates().to_numpy()))
    customer = matrix.index[0]
    never = [s for s in matrix.columns if (customer, s) not in bought]
    assert never, "pick a customer who has not bought everything"
    assert (matrix.loc[customer, never] == 0.0).all()
