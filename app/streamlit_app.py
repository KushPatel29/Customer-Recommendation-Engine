"""Recommendation Product Decision Room and rep-assist console."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

st.set_page_config(
    page_title="Recommendation Product Decision Room",
    page_icon="◎",
    layout="wide",
)

NAVY = "#12324B"
TEAL = "#0F766E"
ORANGE = "#C25B24"

st.markdown(
    f"""
    <style>
    .stApp {{ background: #F7F9F8; color: {NAVY}; }}
    [data-testid="stMetric"] {{ background: white; border: 1px solid #D8E2E4;
        border-radius: 4px; padding: 0.85rem 1rem; box-shadow: 0 4px 18px rgba(18,50,75,.05); }}
    [data-testid="stMetricLabel"] {{ color: #4B6575; letter-spacing: .04em; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; }}
    [data-testid="stRadio"] label p {{ color: {NAVY} !important; font-weight: 650; }}
    .decision-band {{ background: {NAVY}; color: white; border-left: 7px solid {TEAL};
        padding: 1rem 1.2rem; margin: .8rem 0 1rem; }}
    .boundary {{ background: #FFF7ED; border: 1px solid #F3C9A9; border-left: 5px solid {ORANGE};
        padding: .85rem 1rem; margin: .75rem 0 1rem; }}
    .evidence-step {{ background: white; border-top: 3px solid {TEAL}; padding: .8rem;
        min-height: 118px; box-shadow: 0 3px 14px rgba(18,50,75,.05); }}
    .eyebrow {{ color: {TEAL}; letter-spacing: .12em; font-weight: 700; font-size: .78rem; }}
    h1, h2, h3 {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load() -> dict[str, object]:
    packet_file = OUT / "recommendation_release_packet.json"
    experiment_file = OUT / "recommendation_experiment_plan.json"
    slo_file = OUT / "serving_slo.json"
    return {
        "customers": pd.read_csv(OUT / "customer_analytics.csv"),
        "recs": pd.read_csv(OUT / "cross_sell_recommendations.csv"),
        "affinity": pd.read_csv(OUT / "sku_affinity.csv"),
        "evaluation": pd.read_csv(OUT / "holdout_evaluation.csv"),
        "quality": pd.read_csv(OUT / "recommendation_quality_scorecard.csv"),
        "slices": pd.read_csv(OUT / "recommendation_slice_evaluation.csv"),
        "eligibility": pd.read_csv(OUT / "recommendation_eligibility_audit.csv"),
        "packet": json.loads(packet_file.read_text(encoding="utf-8")),
        "experiment": json.loads(experiment_file.read_text(encoding="utf-8")),
        "slo": json.loads(slo_file.read_text(encoding="utf-8")),
    }


data = load()
customers = data["customers"]

st.markdown("<div class='eyebrow'>PRODUCT ANALYTICS · RECOMMENDATION GOVERNANCE</div>", unsafe_allow_html=True)
st.title("Recommendation Product Decision Room")
st.caption(
    "A governed synthetic decision laboratory: prove relevance, inspect who and what the model misses, "
    "verify every served item, and pre-register the online decision before claiming commercial lift."
)

page = st.radio(
    "Workspace",
    ["Release decision", "Rep assist", "Experiment & reliability"],
    horizontal=True,
    label_visibility="collapsed",
)


def release_decision() -> None:
    packet = data["packet"]
    quality = data["quality"]
    slices = data["slices"]
    eligibility = data["eligibility"]
    benchmark = data["slo"]["latest_benchmark"]

    st.markdown(
        f"<div class='decision-band'><b>{packet['release_status']}</b><br>"
        "Ship the collaborative-filtering champion only as a human-reviewed rep-assist pilot. "
        "The learned ranker remains the challenger because it lost the offline bake-off.</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Warm-customer recall@10", f"{packet['offline_recall_at_10']:.1%}")
    c2.metric("Lift vs popularity", f"{packet['offline_recall_at_10'] / packet['popularity_recall_at_10']:.2f}×")
    c3.metric("Catalog coverage", f"{packet['catalog_coverage']:.0%}")
    c4.metric("Eligibility pass", f"{packet['eligibility_pass_rate']:.0%}")
    c5.metric("API p95", f"{benchmark['p95_ms']:.1f} ms")

    st.markdown(
        "<div class='boundary'><b>Decision boundary:</b> offline recall measures warm-customer "
        "basket recovery. It does not measure incremental sales, cold-start quality or rep adoption. "
        "Only the pre-registered customer-level experiment can support a commercial-lift claim.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Evidence ladder")
    e1, e2, e3, e4 = st.columns(4)
    for col, number, title, body in [
        (e1, "01", "Relevance", "CF beats popularity on a leakage-aware warm-customer holdout."),
        (e2, "02", "Product quality", "Coverage, novelty, diversity and calibration expose trade-offs."),
        (e3, "03", "Serving safety", "Every item is eligible, explained, suppressible and fallback-safe."),
        (e4, "04", "Online impact", "Sticky assignment, SRM and guardrails precede promotion."),
    ]:
        col.markdown(
            f"<div class='evidence-step'><b>{number} · {title}</b><br><small>{body}</small></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Multi-objective release scorecard")
    shown = quality.copy()
    shown["value"] = shown["value"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        shown[["objective", "metric", "value", "release_threshold", "status", "measurement_basis"]],
        hide_index=True,
        use_container_width=True,
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Slice diagnostics")
        dimension = st.selectbox("Slice dimension", sorted(slices["dimension"].unique()))
        view = slices[slices["dimension"] == dimension].sort_values("recall_at_10")
        st.bar_chart(view.set_index("slice")["recall_at_10"], color=TEAL)
        st.dataframe(view, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Eligibility control")
        served = int((eligibility["eligibility_decision"] == "SERVE").sum())
        blocked = int((eligibility["eligibility_decision"] == "SUPPRESS").sum())
        st.metric("Audited recommendations", f"{len(eligibility):,}")
        st.metric("Serve / suppress", f"{served:,} / {blocked:,}")
        st.caption(
            "Serve requires a catalog match, unowned product, non-empty reason and positive indicative opportunity."
        )

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download release packet",
        data=json.dumps(packet, indent=2),
        file_name="recommendation_release_packet.json",
        mime="application/json",
    )
    d2.download_button(
        "Download slice evidence",
        data=slices.to_csv(index=False),
        file_name="recommendation_slice_evaluation.csv",
        mime="text/csv",
    )
    d3.download_button(
        "Download eligibility audit",
        data=eligibility.to_csv(index=False),
        file_name="recommendation_eligibility_audit.csv",
        mime="text/csv",
    )


def rep_assist() -> None:
    st.subheader("Human-reviewed call plan")
    left, right = st.columns([1, 3])
    with left:
        region = st.selectbox("Region", ["All"] + sorted(customers["region"].unique()))
        pool = customers if region == "All" else customers[customers["region"] == region]
        ordered = pool.sort_values("total_revenue", ascending=False)["customer_id"].tolist()
        opportunity = (
            data["recs"][data["recs"]["customer_id"].isin(ordered)]
            .groupby("customer_id")["est_revenue_opportunity"]
            .sum()
        )
        default = ordered.index(opportunity.idxmax()) if len(opportunity) else 0
        pick = st.selectbox(
            "Customer",
            ordered,
            index=default,
            format_func=lambda customer_id: f"{customer_id} — "
            f"{pool.set_index('customer_id').at[customer_id, 'customer_name']}",
        )
    customer = customers[customers["customer_id"] == pick].iloc[0]
    with right:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Revenue window", f"${customer['total_revenue']:,.0f}")
        c2.metric("12-mo run-rate CLV", f"${customer['clv_12m_runrate']:,.0f}")
        c3.metric("RFM segment", customer["rfm_segment"])
        c4.metric("Churn risk", customer["churn_risk"])
        c5.metric("Rep", customer["rep"])

    recs = data["recs"][data["recs"]["customer_id"] == pick].sort_values("rank")
    col_recs, col_aff = st.columns([3, 2])
    with col_recs:
        st.subheader("Eligible pitches")
        if recs.empty:
            st.info("No eligible white-space item remains. Treat this as a retention call.")
        else:
            show = recs[
                ["rank", "description", "protein", "score", "est_revenue_opportunity", "because_similar_to"]
            ].rename(
                columns={
                    "rank": "#",
                    "description": "SKU",
                    "protein": "Protein",
                    "score": "CF score",
                    "est_revenue_opportunity": "$ opportunity",
                    "because_similar_to": "Reason: similar customer",
                }
            )
            st.dataframe(
                show,
                hide_index=True,
                use_container_width=True,
                column_config={"$ opportunity": st.column_config.NumberColumn(format="$%.0f")},
            )
            st.markdown(
                f"**Indicative—not forecast—opportunity: <span style='color:{TEAL}'>"
                f"${recs['est_revenue_opportunity'].sum():,.0f}</span>**",
                unsafe_allow_html=True,
            )
    with col_aff:
        st.subheader("Basket talk tracks")
        affinity = data["affinity"].head(8)[["description_a", "description_b", "lift"]]
        affinity.columns = ["If they order…", "…suggest", "Observed lift"]
        st.dataframe(affinity, hide_index=True, use_container_width=True)
        st.caption("The representative decides whether context makes the suggestion appropriate.")


def experiment_and_reliability() -> None:
    experiment = data["experiment"]
    slo = data["slo"]
    benchmark = slo["latest_benchmark"]
    st.subheader("Pre-registered online decision")
    st.markdown(f"**{experiment['decision']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Randomization", "Customer")
    c2.metric("Assignment", "50 / 50 sticky")
    c3.metric("Minimum exposure", "1,000 / variant")
    c4.metric("Promotion alpha", "0.05")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Outcome hierarchy")
        st.write(f"**Primary:** {experiment['primary_metric']}")
        for metric in experiment["secondary_metrics"]:
            st.write(f"• {metric}")
        st.markdown("#### Guardrails")
        for guardrail in experiment["guardrails"]:
            st.write(f"• {guardrail}")
    with right:
        st.markdown("#### Readout controls")
        for check, rule in experiment["quality_checks"].items():
            st.write(f"**{check.replace('_', ' ').title()}:** {rule}")
        st.info(experiment["promotion_rule"])

    st.subheader("Serving reliability")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("p50", f"{benchmark['p50_ms']:.1f} ms", f"target {slo['targets_ms']['p50']} ms")
    s2.metric("p95", f"{benchmark['p95_ms']:.1f} ms", f"target {slo['targets_ms']['p95']} ms")
    s3.metric("p99", f"{benchmark['p99_ms']:.1f} ms", f"target {slo['targets_ms']['p99']} ms")
    s4.metric("Fallback", "READY" if benchmark["all_targets_met"] else "REVIEW")
    st.success(
        "Degraded mode uses regional popularity, excludes owned items, carries a visible reason code "
        "and invents no dollar opportunity. Unknown identity or unverifiable eligibility fails closed."
    )
    st.caption(f"Benchmark basis: {benchmark['environment']} · {benchmark['requests']} warm in-process requests.")


if page == "Release decision":
    release_decision()
elif page == "Rep assist":
    rep_assist()
else:
    experiment_and_reliability()

st.caption(
    "All customers, purchases, catalog items, opportunities and experiment traffic are synthetic. "
    "Source: github.com/KushPatel29/customer-recommendation-engine"
)
