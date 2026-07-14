"""
Rep console — interactive front-end for the recommendation engine.

A sales rep picks a customer and gets, on one screen: who the customer is
(RFM segment, churn risk, run-rate CLV), what to pitch next (cross-sell recs
with the "because similar to" explanation and the $ opportunity), and the
basket-affinity talk tracks for what's already in the order.

Runs entirely from the committed pipeline outputs:

    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

st.set_page_config(page_title="Cross-Sell Rep Console", page_icon="🥩", layout="wide")

NAVY = "#12436D"
TEAL = "#28A197"


@st.cache_data
def load():
    return {
        "customers": pd.read_csv(OUT / "customer_analytics.csv"),
        "recs": pd.read_csv(OUT / "cross_sell_recommendations.csv"),
        "affinity": pd.read_csv(OUT / "sku_affinity.csv"),
        "evaluation": pd.read_csv(OUT / "holdout_evaluation.csv"),
    }


data = load()
customers = data["customers"]

st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0'>Cross-Sell Rep Console</h1>",
            unsafe_allow_html=True)
st.caption("Item-based collaborative filtering, holdout-evaluated: "
           "CF hit-rate@10 **84.9%** vs popularity baseline 75.4%. "
           "Every recommendation carries its reason and its dollar value.")

# ------------------------------------------------------------- selectors
left, right = st.columns([1, 3])
with left:
    region = st.selectbox("Region", ["All"] + sorted(customers["region"].unique()))
    pool = customers if region == "All" else customers[customers["region"] == region]
    ordered = pool.sort_values("total_revenue", ascending=False)["customer_id"].tolist()
    # default to the biggest open cross-sell opportunity, not just the biggest
    # customer (who is often fully penetrated)
    opp = (data["recs"][data["recs"]["customer_id"].isin(ordered)]
           .groupby("customer_id")["est_revenue_opportunity"].sum())
    default = ordered.index(opp.idxmax()) if len(opp) else 0
    pick = st.selectbox(
        "Customer",
        ordered,
        index=default,
        format_func=lambda cid: f"{cid} — "
        f"{pool.set_index('customer_id').at[cid, 'customer_name']}",
    )

cust = customers[customers["customer_id"] == pick].iloc[0]

with right:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue (TTM window)", f"${cust['total_revenue']:,.0f}")
    c2.metric("12-mo run-rate CLV", f"${cust['clv_12m_runrate']:,.0f}")
    c3.metric("RFM segment", cust["rfm_segment"])
    c4.metric("Churn risk", cust["churn_risk"],
              delta="overdue" if cust["days_overdue"] and cust["days_overdue"] > 0 else None,
              delta_color="inverse")
    c5.metric("Rep", cust["rep"])

st.divider()

# ------------------------------------------------------------- recs
recs = (data["recs"][data["recs"]["customer_id"] == pick]
        .sort_values("rank"))

col_recs, col_aff = st.columns([3, 2])

with col_recs:
    st.subheader("What to pitch next")
    if recs.empty:
        st.info("Fully penetrated — the engine has no white-space SKUs left. "
                "This customer is a retention call, not a cross-sell call.")
    else:
        show = recs[["rank", "description", "protein", "score",
                     "est_revenue_opportunity", "because_similar_to"]].rename(columns={
            "rank": "#", "description": "SKU", "protein": "Protein",
            "score": "CF score", "est_revenue_opportunity": "$ opportunity",
            "because_similar_to": "Because they buy like…"})
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={
                         "$ opportunity": st.column_config.NumberColumn(format="$%.0f"),
                         "CF score": st.column_config.ProgressColumn(
                             format="%.2f", min_value=0,
                             max_value=float(show["CF score"].max())),
                     })
        total = recs["est_revenue_opportunity"].sum()
        st.markdown(f"**Total indicative opportunity: "
                    f"<span style='color:{TEAL}'>${total:,.0f}</span>**",
                    unsafe_allow_html=True)

with col_aff:
    st.subheader("Basket talk tracks")
    st.caption("“Orders with A are N.N× more likely to include B” — pitch these "
               "when A is already on the order.")
    aff = data["affinity"].head(8)[["description_a", "description_b", "lift"]]
    aff.columns = ["If they order…", "…suggest", "Lift"]
    st.dataframe(aff, hide_index=True, use_container_width=True,
                 column_config={"Lift": st.column_config.NumberColumn(format="%.2f×")})

st.divider()
ev = data["evaluation"]
st.caption(f"Model card: evaluated on {len(ev)} holdout customers · "
           f"CF re-discovered {ev['cf_hits'].sum()} of {ev['hidden'].sum()} hidden SKUs · "
           "scores are batch-computed by engine/recommend.py; this console and the "
           "FastAPI service serve the same table.")
