"""
Data contracts — pandera schemas enforced at the pipeline boundaries.

Two boundaries are protected:
  * source contract  : what the generator (or, in production, the upstream
    order system) must deliver before the engine is allowed to run
  * output contract  : what the engine must deliver before Power BI, the API,
    and the Streamlit console are allowed to serve it

Run standalone (exits non-zero on any violation — CI runs this after the
pipeline, so a contract break fails the build):

    python contracts/schemas.py
"""

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check as C
from pandera.pandas import Column

ROOT = Path(__file__).resolve().parent.parent

CUSTOMER_ID = C.str_matches(r"^CUST-\d{3}$")
SKU_ID = C.str_matches(r"^SKU-\d{3}$")

sales_lines_schema = pa.DataFrameSchema(
    {
        "order_id": Column(str, C.str_matches(r"^ORD-\d{5}$")),
        "order_date": Column(pa.DateTime, coerce=True),
        "customer_id": Column(str, CUSTOMER_ID),
        "customer_name": Column(str, C.str_length(min_value=1)),
        "region": Column(str, C.str_length(min_value=1)),
        "rep": Column(str, C.str_length(min_value=1)),
        "sku": Column(str, SKU_ID),
        "protein": Column(str),
        "description": Column(str),
        "quantity_lb": Column(float, C.gt(0), coerce=True),
        "unit_price": Column(float, C.gt(0)),
        "revenue": Column(float, C.ge(0)),
        "cost": Column(float, C.ge(0)),
    },
    checks=C(lambda df: (df["revenue"] - df["quantity_lb"] * df["unit_price"]).abs() < 0.05,
             name="revenue_equals_qty_x_price",
             error="revenue must reconcile to quantity_lb * unit_price"),
    strict=True,
    unique_column_names=True,
)

cross_sell_schema = pa.DataFrameSchema(
    {
        "customer_id": Column(str, CUSTOMER_ID),
        "customer_name": Column(str),
        "rank": Column(int, C.in_range(1, 10), coerce=True),
        "sku": Column(str, SKU_ID),
        "protein": Column(str),
        "description": Column(str),
        "score": Column(float, C.gt(0)),
        "est_revenue_opportunity": Column(float, C.ge(0)),
        "because_similar_to": Column(str, nullable=True),
    },
    strict=True,
)

customer_analytics_schema = pa.DataFrameSchema(
    {
        "customer_id": Column(str, CUSTOMER_ID),
        "total_revenue": Column(float, C.ge(0)),
        "total_margin": Column(float),
        "orders": Column(int, C.gt(0), coerce=True),
        "churn_risk": Column(str, C.isin(["Low", "Medium", "High"])),
        "rfm_segment": Column(str, C.str_length(min_value=1)),
        "clv_12m_runrate": Column(float, C.ge(0)),
        "margin_pct": Column(float, C.in_range(-1, 1)),
    },
    strict=False,  # analytic table carries many more columns; contract pins the load-bearing ones
)


CONTRACTS = [
    ("data/sales_lines.csv", sales_lines_schema, dict(parse_dates=["order_date"])),
    ("output/cross_sell_recommendations.csv", cross_sell_schema, {}),
    ("output/customer_analytics.csv", customer_analytics_schema, {}),
]


def validate_all() -> list[str]:
    """Validate every contracted file; returns a list of failure summaries."""
    failures: list[str] = []
    for relpath, schema, read_kwargs in CONTRACTS:
        df = pd.read_csv(ROOT / relpath, **read_kwargs)
        try:
            schema.validate(df, lazy=True)
            print(f"PASS  {relpath}  ({len(df):,} rows)")
        except pa.errors.SchemaErrors as err:
            failures.append(f"{relpath}: {len(err.failure_cases)} violations")
            print(f"FAIL  {relpath}")
            print(err.failure_cases.head(10).to_string(index=False))
    return failures


if __name__ == "__main__":
    broken = validate_all()
    if broken:
        print("\ncontract violations:", "; ".join(broken))
        sys.exit(1)
    print("\nall data contracts hold")
