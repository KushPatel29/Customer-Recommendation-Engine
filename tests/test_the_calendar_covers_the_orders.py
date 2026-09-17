"""
The month table has to cover every month an order lands in.

It was typed out - thirteen months from July 2025 - and the orders happen to fit
inside it today. Two other reports in this portfolio had the same calendar and
data that moved: one lost a year of cash to a (Blank) month, the other three
validation runs, with every total still right. Nothing errors when a calendar is
short, so the range is derived from the orders now and this holds it there.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = next(ROOT.glob("powerbi/pbip/*.SemanticModel")) / "definition"


def calendar_tmdl() -> str:
    return (MODEL / "tables" / "dim_month.tmdl").read_text(encoding="utf-8")


def test_the_calendar_reads_the_orders_it_has_to_cover():
    tmdl = calendar_tmdl()
    assert "sales_lines.csv" in tmdl
    assert '"order_date"' in tmdl, "the span must come from the column the join is built on"
    assert "List.Min" in tmdl and "List.Max" in tmdl


def test_the_calendar_is_not_pinned_to_a_literal_date():
    assert not re.search(r"#date\(\d{4}", calendar_tmdl())


def test_the_join_is_built_from_the_same_column():
    """order_month is the month of order_date; a calendar derived from any other
    date would cover a different range."""
    sales = (MODEL / "tables" / "fact_sales.tmdl").read_text(encoding="utf-8")
    assert re.search(r'"order_month",\s*each Date\.ToText\(\[order_date\]', sales)
    relationships = (MODEL / "relationships.tmdl").read_text(encoding="utf-8")
    assert re.search(r"fromColumn:\s*fact_sales\.order_month\s*\n\s*toColumn:\s*dim_month\.month",
                     relationships)


def test_every_order_month_lies_inside_the_derived_span():
    orders = pd.to_datetime(pd.read_csv(ROOT / "data" / "sales_lines.csv",
                                        usecols=["order_date"])["order_date"])
    months = set(orders.dt.strftime("%Y-%m"))
    span = set(pd.period_range(min(months), max(months), freq="M").strftime("%Y-%m"))
    assert months <= span
