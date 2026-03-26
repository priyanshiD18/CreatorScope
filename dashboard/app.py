"""
CreatorScope — Streamlit Dashboard
====================================
Tab 1: Ecosystem Overview
Tab 2: Creator Deep Dive
Tab 3: Experiment Toolkit (Power Analysis)
Tab 4: Growth Forecast
Tab 5: Churn Risk Radar
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.power import TTestIndPower

from src.data_loader import generate_synthetic_channels, generate_synthetic_videos
from src.kpi_metrics import (
    compute_engagement_rate, compute_consistency_score,
    gini_coefficient, full_kpi_report,
)
from src.cohort_analysis import (
    build_cohorts, survival_curve, build_cluster_features, segment_creators,
)
from src.models.forecasting import (
    make_channel_timeseries, forecast_arima, forecast_prophet,
    detect_anomalies, build_churn_features, train_churn_model,
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreatorScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading creator data...")
def load_data():
    raw_dir = Path("data/raw")
    if (raw_dir / "global_yt_stats.csv").exists():
        from src.data_loader import load_global_yt_stats, load_trending_videos
        channels = load_global_yt_stats()
        videos   = load_trending_videos()
    else:
        channels = generate_synthetic_channels(n=500, seed=42)
        videos   = generate_synthetic_videos(channels, n_per_channel=30, seed=42)
    videos["engagement_rate"] = compute_engagement_rate(videos)
    return channels, videos


channels, videos = load_data()

@st.cache_data(show_spinner="Running segmentation...")
def get_segments():
    feat = build_cluster_features(channels, videos)
    seg, _ = segment_creators(feat, n_clusters=5)
    return seg

@st.cache_data(show_spinner="Computing churn model...")
def get_churn():
    feat  = build_churn_features(channels, videos, inactivity_days=90)
    result = train_churn_model(feat)
    return result["churn_scores"].merge(
        channels[["channel_id", "channel_name", "category", "country", "subscribers"]],
        on="channel_id", how="left"
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("CreatorScope")
st.sidebar.caption("Product Analytics Dashboard")
tab_choice = st.sidebar.radio(
    "Navigate",
    ["Ecosystem Overview", "Creator Deep Dive",
     "Experiment Toolkit", "Growth Forecast", "Churn Risk Radar"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ecosystem Overview
# ═══════════════════════════════════════════════════════════════════════════════
if tab_choice == "Ecosystem Overview":
    st.title("Ecosystem Overview")

    kpi = full_kpi_report(channels, videos)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Channels",    f"{kpi['total_channels']:,}")
    c2.metric("Total Videos",      f"{kpi['total_videos']:,}")
    c3.metric("Gini Coefficient",  f"{kpi['gini_coefficient']:.3f}", help="0=equal, 1=max inequality")
    c4.metric("Category HHI",      f"{kpi['hhi_categories']:.0f}", help="<1500=diverse, >2500=concentrated")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("View Distribution (Power Law)")
        views_sorted = np.sort(channels["total_views"].dropna().values)[::-1]
        cumulative   = np.cumsum(views_sorted) / (views_sorted.sum() + 1e-9)
        pop_share    = np.linspace(0, 1, len(cumulative))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pop_share * 100, y=cumulative * 100,
                                  name="Lorenz Curve", fill="tozeroy", fillcolor="rgba(0,100,200,0.1)"))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Perfect Equality",
                                  line=dict(dash="dash", color="grey")))
        fig.update_layout(xaxis_title="% of Channels", yaxis_title="% of Total Views",
                           title=f"Lorenz Curve  |  Gini = {kpi['gini_coefficient']:.3f}")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Category Leaderboard (by upload frequency)")
        cat_stats = (
            channels.groupby("category")["avg_weekly_uploads"]
            .median()
            .reset_index()
            .sort_values("avg_weekly_uploads", ascending=True)
        )
        fig2 = px.bar(cat_stats, x="avg_weekly_uploads", y="category", orientation="h",
                       color="avg_weekly_uploads", color_continuous_scale="Blues",
                       labels={"avg_weekly_uploads": "Median uploads/week", "category": ""},
                       title="Median Upload Frequency by Category")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Geographic Distribution")
    geo = channels.groupby("country")["subscribers"].sum().reset_index()
    fig3 = px.choropleth(geo, locations="country", locationmode="ISO-3",
                          color="subscribers", color_continuous_scale="Viridis",
                          title="Total Subscribers by Country (Top Creators)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Engagement by Channel Tier")
    ch_v = videos.merge(channels[["channel_id", "subscribers"]], on="channel_id", how="left")
    ch_v["tier"] = pd.cut(ch_v["subscribers"].fillna(0),
                           bins=[0, 1e4, 1e5, 1e6, 1e7, np.inf],
                           labels=["Nano", "Micro", "Mid", "Macro", "Mega"])
    tier_med = ch_v.groupby("tier")["engagement_rate"].median().reset_index()
    fig4 = px.bar(tier_med, x="tier", y="engagement_rate", color="tier",
                   title="Median Engagement Rate by Channel Size",
                   labels={"engagement_rate": "Engagement rate", "tier": "Tier"})
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Creator Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "Creator Deep Dive":
    st.title("Creator Deep Dive")

    channel_names = channels["channel_name"].tolist()
    selected_name = st.selectbox("Select a creator", channel_names)
    sel_ch = channels[channels["channel_name"] == selected_name].iloc[0]
    ch_id  = sel_ch["channel_id"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Subscribers",  f"{int(sel_ch.get('subscribers', 0)):,}")
    col2.metric("Category",     sel_ch.get("category", "N/A"))
    col3.metric("Country",      sel_ch.get("country", "N/A"))

    # Segment
    seg_df = get_segments()
    seg_row = seg_df[seg_df["channel_id"] == ch_id]
    if len(seg_row):
        st.info(f"Segment: **{seg_row.iloc[0]['segment']}**")

    st.subheader("View Trajectory")
    ch_videos = videos[videos["channel_id"] == ch_id].sort_values("publish_time")
    if len(ch_videos) > 0:
        ts = make_channel_timeseries(videos, ch_id, freq="W")
        fig = px.line(x=ts.index, y=ts.values,
                       labels={"x": "Date", "y": "Weekly views"},
                       title=f"Weekly Views — {selected_name}")
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly-flagged videos
        ch_anom = detect_anomalies(ch_videos)
        flagged = ch_anom[ch_anom["is_anomaly"]]
        if len(flagged):
            st.subheader("Anomaly-Flagged Videos")
            st.dataframe(flagged[["title", "publish_time", "views",
                                   "engagement_rate", "anomaly_type"]].head(10))
        else:
            st.success("No anomalies detected for this creator.")
    else:
        st.warning("No video data for this creator.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Experiment Toolkit
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "Experiment Toolkit":
    st.title("Experiment Power Calculator")
    st.markdown("Design A/B experiments — enter parameters to compute required sample size.")

    col1, col2 = st.columns(2)
    with col1:
        baseline    = st.number_input("Baseline metric value", value=1.2, step=0.1)
        mde_pct     = st.slider("Minimum detectable effect (%)", 1, 50, 10) / 100
        alpha       = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05)
        power       = st.select_slider("Power (1−β)", options=[0.70, 0.80, 0.90, 0.95], value=0.80)
        std_est     = st.number_input("Std dev estimate", value=0.8, step=0.1)
        daily_users = st.number_input("Daily active creators", value=5000, step=500)

    with col2:
        effect_size = (baseline * mde_pct) / (std_est + 1e-9)
        analysis    = TTestIndPower()
        n_per_group = int(np.ceil(analysis.solve_power(effect_size, alpha=alpha, power=power)))
        total_n     = n_per_group * 2
        runtime     = int(np.ceil(total_n / daily_users))

        st.metric("Cohen's d",         f"{effect_size:.3f}")
        st.metric("N per group",       f"{n_per_group:,}")
        st.metric("Total sample size", f"{total_n:,}")
        st.metric("Estimated runtime", f"{runtime} days")

    # Power curve
    st.subheader("Power Curve")
    es_range = np.linspace(0.05, 1.2, 200)
    pw_range = [analysis.solve_power(e, alpha=alpha, nobs1=n_per_group) for e in es_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=es_range, y=pw_range, name="Power curve"))
    fig.add_hline(y=power, line_dash="dash", line_color="tomato", annotation_text=f"Target power={power}")
    fig.add_vline(x=effect_size, line_dash="dash", line_color="green",
                   annotation_text=f"d={effect_size:.2f}")
    fig.update_layout(xaxis_title="Effect size (Cohen's d)", yaxis_title="Power",
                       title="Statistical Power Curve", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Growth Forecast
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "Growth Forecast":
    st.title("Growth Forecast")

    ch_name = st.selectbox("Select creator", channels["channel_name"].tolist(), key="fc_ch")
    ch_id   = channels[channels["channel_name"] == ch_name].iloc[0]["channel_id"]
    model_choice = st.radio("Forecast model", ["ARIMA", "Prophet"], horizontal=True)
    steps   = st.slider("Forecast horizon (weeks)", 4, 52, 12)

    ts = make_channel_timeseries(videos, ch_id, freq="W")
    if len(ts) < 4:
        st.warning("Not enough data to forecast.")
    else:
        with st.spinner("Fitting model..."):
            fn = forecast_arima if model_choice == "ARIMA" else forecast_prophet
            result = fn(ts, steps=steps)

        fc = result.get("forecast", pd.Series(dtype=float))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical", line=dict(color="grey")))
        if len(fc):
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name=f"{model_choice} Forecast",
                                      line=dict(color="steelblue", dash="dash")))
            if "lower" in result and "upper" in result:
                fig.add_trace(go.Scatter(
                    x=list(result["lower"].index) + list(result["upper"].index[::-1]),
                    y=list(result["lower"].values) + list(result["upper"].values[::-1]),
                    fill="toself", fillcolor="rgba(0,100,200,0.15)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% CI"
                ))
        fig.update_layout(title=f"{model_choice} Forecast — {ch_name}",
                           xaxis_title="Date", yaxis_title="Weekly views")
        st.plotly_chart(fig, use_container_width=True)

        if model_choice == "Prophet" and "components" in result:
            st.subheader("Seasonality Decomposition")
            comp = result["components"]
            fig2, axes = plt.subplots(1, 2, figsize=(12, 3))
            if "weekly" in comp.columns:
                axes[0].plot(comp["ds"].iloc[:52], comp["weekly"].iloc[:52])
                axes[0].set_title("Weekly seasonality")
            if "yearly" in comp.columns:
                axes[1].plot(comp["ds"], comp["yearly"])
                axes[1].set_title("Yearly seasonality")
            st.pyplot(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Churn Risk Radar
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "Churn Risk Radar":
    st.title("Churn Risk Radar")
    st.markdown("Creators ranked by predicted probability of going inactive (≥90 days no upload).")

    with st.spinner("Computing churn scores..."):
        churn_df = get_churn()

    top_n = st.slider("Show top N at-risk creators", 5, 50, 20)
    at_risk = churn_df.sort_values("churn_probability", ascending=False).head(top_n)

    fig = px.bar(
        at_risk,
        x="churn_probability",
        y="channel_name",
        orientation="h",
        color="churn_probability",
        color_continuous_scale="Reds",
        labels={"churn_probability": "Churn probability", "channel_name": "Creator"},
        title=f"Top {top_n} At-Risk Creators",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Factor Breakdown")
    display_cols = ["channel_name", "category", "churn_probability",
                    "gap_trend", "engagement_trend", "days_since_last", "churned"]
    display_cols = [c for c in display_cols if c in at_risk.columns]

    styled = at_risk[display_cols].style.background_gradient(
        subset=["churn_probability"], cmap="Reds"
    ).format({"churn_probability": "{:.1%}", "gap_trend": "{:.1f}",
               "engagement_trend": "{:.3f}"})
    st.dataframe(styled, use_container_width=True)

    st.subheader("Recommended Interventions")
    st.markdown("""
| Risk signal | Intervention |
|---|---|
| Upload gap tripled | Send personalized "We miss you" email with growth stats |
| Engagement down 40%+ | Surface similar-niche creator collaboration suggestions |
| Stalled at < 1K subs | Early milestone badge + featured in new-creator newsletter |
| No uploads in 60 days | In-studio prompt: "Your last video got X views — keep going!" |
""")

# Tab 1 complete: Lorenz curve, Gini, category leaderboard, choropleth map

# Tab 2 complete: creator deep dive, segment label, anomaly-flagged video table

# Tab 3: live power calculator — input MDE and get sample size + runtime estimate

# Tab 4: forecast tab with ARIMA/Prophet toggle and confidence band visualisation

# Tab 5: churn risk radar sorted by probability with intervention playbook table

# Tab 1 complete: Lorenz curve, Gini, category leaderboard, choropleth map

# Tab 2 complete: creator deep dive, segment label, anomaly-flagged video table
