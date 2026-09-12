"""
CreatorSignal Chicago — Streamlit Web App
Live dashboard for the ML decision-support pipeline.
Runs the full pipeline on sample data, then lets users explore results interactively.
"""
import sys
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_exports"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"

for d in [DATA_DIR, RAW_DIR, OUT_DIR, CHART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "creatorsignal.db"
MODEL_RESULTS_PATH = OUT_DIR / "model_results.json"

# ── palette ────────────────────────────────────────────────────────────────
C = {
    "primary": "#1B4B5A",
    "accent": "#F2A541",
    "success": "#2E7D32",
    "light": "#E8EEF0",
    "text": "#233238",
    "muted": "#7A8E96",
}
CHART_COLORS = ["#1B4B5A", "#F2A541", "#3A7D8C", "#D4763B", "#5BA08A", "#C75B3F", "#8FBCC4", "#E8965A"]


# ── run pipeline once per session ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline():
    """Execute the full pipeline and return all artifacts."""
    import generate_sample_exports
    import clean_ingest
    import features
    import model as model_mod
    import cadence_and_collab

    generate_sample_exports.main()
    clean_df, clean_report = clean_ingest.main()
    feature_df = features.main()
    best_rf, model_results, feature_cols = model_mod.run()
    cadence, matches = cadence_and_collab.main()

    return clean_report, model_results


def load_data():
    """Load the pipeline outputs from disk."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM creator_posts", conn, parse_dates=["post_date"])
    conn.close()
    cadence = pd.read_csv(OUT_DIR / "cadence_recommendations.csv")
    matches = pd.read_csv(OUT_DIR / "collab_matches.csv")
    with open(MODEL_RESULTS_PATH) as f:
        model_results = json.load(f)
    features_df = pd.read_csv(DATA_DIR / "model_ready_dataset.csv")
    return df, cadence, matches, model_results, features_df


# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreatorSignal Chicago",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1B4B5A 0%, #3A7D8C 100%);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        color: white;
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .label { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1B4B5A;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
        border-bottom: 2px solid #F2A541;
        padding-bottom: 0.3rem;
    }
    .callout {
        background: #F0F4F6;
        border-left: 4px solid #1B4B5A;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.92rem;
        color: #233238;
    }
    div[data-testid="stMetric"] { background: #F0F4F6; border-radius: 8px; padding: 0.6rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ── run pipeline ───────────────────────────────────────────────────────────
with st.spinner("Running the full CreatorSignal pipeline — this takes about 15 seconds on first load..."):
    clean_report, _ = run_pipeline()

df, cadence, matches, model_results, features_df = load_data()
creators = sorted(df["creator_handle"].unique())

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CreatorSignal Chicago")
    st.caption("ML decision-support for content creators")
    st.markdown("---")
    selected_creator = st.selectbox("Select a creator", creators, format_func=lambda x: f"@{x}")
    st.markdown("---")

    st.markdown("### Pipeline Stats")
    st.markdown(f"**Posts analyzed:** {len(df)}")
    st.markdown(f"**Creators:** {len(creators)}")
    st.markdown(f"**Duplicates removed:** {clean_report.get('duplicates_removed', 0)}")
    st.markdown(f"**Missing lengths imputed:** {clean_report.get('rows_missing_length', 0)}")
    st.markdown("---")

    st.markdown("### What is this?")
    st.markdown(
        "CreatorSignal takes a creator's own analytics exports and turns them into "
        "three actionable answers: *what* to post about, *when* to post it, and *who* "
        "to collaborate with. It's decision support, not a content generator."
    )
    st.markdown("---")
    st.caption("Built with Python · scikit-learn · Streamlit")


# ── filter data to selected creator ────────────────────────────────────────
cdf = df[df["creator_handle"] == selected_creator].copy()
platform = cdf["platform"].iloc[0]
creator_cadence = cadence[cadence["platform"] == platform]
creator_matches = matches[matches["creator"] == selected_creator].sort_values("rank")

# ── header ─────────────────────────────────────────────────────────────────
st.markdown(f"# CreatorSignal Report — @{selected_creator}")
st.markdown(f"**Platform:** {platform} · **Posts analyzed:** {len(cdf)} · "
            f"**Date range:** {cdf['post_date'].min().strftime('%b %Y')} – {cdf['post_date'].max().strftime('%b %Y')}")

# ── KPI cards ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
avg_eng = cdf["engagement_rate"].mean()
avg_reach = cdf["reach"].mean()
best_theme = cdf.groupby("theme_tag")["engagement_rate"].mean().idxmax().replace("_", " ").title()
best_hour = creator_cadence.iloc[0]["recommended_hour"] if len(creator_cadence) > 0 else "N/A"

with k1:
    st.metric("Avg Engagement Rate", f"{avg_eng:.2%}")
with k2:
    st.metric("Avg Reach / Post", f"{avg_reach:,.0f}")
with k3:
    st.metric("Best Theme", best_theme)
with k4:
    st.metric("Best Post Hour", f"{int(best_hour)}:00" if best_hour != "N/A" else "N/A")

st.markdown("---")

# ── content & timing tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Content Performance",
    "⏰ Posting Cadence",
    "🤝 Collaboration Matches",
    "🧠 Model & Predictions",
    "📋 Raw Data",
])

# ── TAB 1: content performance ─────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Engagement by Content Theme</div>', unsafe_allow_html=True)
    theme_perf = cdf.groupby("theme_tag")["engagement_rate"].agg(["mean", "count"]).reset_index()
    theme_perf.columns = ["Theme", "Avg Engagement Rate", "Posts"]
    theme_perf["Theme"] = theme_perf["Theme"].str.replace("_", " ").str.title()
    theme_perf = theme_perf.sort_values("Avg Engagement Rate", ascending=True)

    fig_theme = px.bar(
        theme_perf, x="Avg Engagement Rate", y="Theme",
        orientation="h", text="Posts",
        color_discrete_sequence=[C["primary"]],
    )
    fig_theme.update_traces(texttemplate="%{text} posts", textposition="outside", textfont_size=11)
    fig_theme.update_layout(
        height=380, margin=dict(l=0, r=40, t=10, b=0),
        xaxis_tickformat=".1%", yaxis_title="", xaxis_title="Average Engagement Rate",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_theme, use_container_width=True)

    st.markdown('<div class="section-header">Engagement Over Time</div>', unsafe_allow_html=True)
    timeline = cdf.set_index("post_date").resample("2W")["engagement_rate"].mean().reset_index()
    fig_time = px.line(
        timeline, x="post_date", y="engagement_rate",
        color_discrete_sequence=[C["accent"]],
        markers=True,
    )
    fig_time.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Avg Engagement Rate",
        plot_bgcolor="white",
    )
    fig_time.update_traces(line_width=2.5, marker_size=7)
    st.plotly_chart(fig_time, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Reach by Theme</div>', unsafe_allow_html=True)
        reach_theme = cdf.groupby("theme_tag")["reach"].mean().reset_index()
        reach_theme.columns = ["Theme", "Avg Reach"]
        reach_theme["Theme"] = reach_theme["Theme"].str.replace("_", " ").str.title()
        reach_theme = reach_theme.sort_values("Avg Reach", ascending=True)
        fig_reach = px.bar(reach_theme, x="Avg Reach", y="Theme", orientation="h",
                           color_discrete_sequence=[C["primary"]])
        fig_reach.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                xaxis_title="Average Reach", yaxis_title="", plot_bgcolor="white")
        st.plotly_chart(fig_reach, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Post Length vs Engagement</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            cdf, x="length_seconds", y="engagement_rate",
            color_discrete_sequence=[C["primary"]],
            opacity=0.6,
        )
        fig_scatter.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="Post Length (seconds)", yaxis_title="Engagement Rate",
                                  yaxis_tickformat=".1%", plot_bgcolor="white")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ── TAB 2: posting cadence ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Engagement by Hour of Day</div>', unsafe_allow_html=True)
    hourly = cdf.groupby("post_hour").agg(
        avg_engagement=("engagement_rate", "mean"),
        post_count=("post_id", "count"),
    ).reset_index()

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly["post_hour"], y=hourly["post_count"],
        name="Posts", marker_color=C["light"], yaxis="y2", opacity=0.5,
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly["post_hour"], y=hourly["avg_engagement"],
        name="Avg Engagement", marker_color=C["accent"],
        mode="lines+markers", line_width=3, marker_size=8,
    ))
    fig_hourly.update_layout(
        height=380, margin=dict(l=0, r=40, t=10, b=0),
        yaxis=dict(title="Avg Engagement Rate", tickformat=".1%", side="left"),
        yaxis2=dict(title="Post Count", overlaying="y", side="right"),
        xaxis=dict(title="Hour of Day", dtick=1),
        legend=dict(orientation="h", y=1.1), plot_bgcolor="white",
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.markdown('<div class="section-header">Engagement by Day of Week</div>', unsafe_allow_html=True)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = cdf.groupby("day_of_week")["engagement_rate"].agg(["mean", "count"]).reindex(day_order).reset_index()
    daily.columns = ["Day", "Avg Engagement", "Posts"]

    fig_daily = px.bar(
        daily, x="Day", y="Avg Engagement", text="Posts",
        color_discrete_sequence=[C["primary"]],
    )
    fig_daily.update_traces(texttemplate="%{text} posts", textposition="outside", textfont_size=11)
    fig_daily.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat=".1%", xaxis_title="", yaxis_title="Avg Engagement Rate",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    st.markdown('<div class="section-header">Cadence Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout">Recommendations for <strong>{platform}</strong> are based on '
                f'historical performance with a minimum sample threshold — a single lucky post at an '
                f'unusual hour won\'t drive a recommendation.</div>', unsafe_allow_html=True)
    if len(creator_cadence) > 0:
        rec_display = creator_cadence[["recommended_hour", "avg_engagement_rate", "sample_size", "best_day"]].copy()
        rec_display.columns = ["Recommended Hour", "Avg Engagement Rate", "Posts at This Hour", "Best Day"]
        rec_display["Recommended Hour"] = rec_display["Recommended Hour"].apply(lambda h: f"{int(h)}:00")
        rec_display["Avg Engagement Rate"] = rec_display["Avg Engagement Rate"].apply(lambda x: f"{x:.2%}")
        st.dataframe(rec_display, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for cadence recommendations yet.")


# ── TAB 3: collaboration matches ───────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Top Collaboration Matches</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout">Matches are ranked by cosine similarity across content theme '
                f'distribution and engagement profile. Cross-platform matches are flagged because they '
                f'tend to introduce each creator\'s audience to a completely new pool of people.</div>',
                unsafe_allow_html=True)

    if len(creator_matches) > 0:
        for _, row in creator_matches.iterrows():
            cross = "🌐 Cross-platform" if row["cross_platform_opportunity"] else "Same platform"
            sim_pct = row["similarity_score"] * 100
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.markdown(f"**#{int(row['rank'])}  @{row['match_candidate']}**")
            with col2:
                st.markdown(f"{row['candidate_platform']}")
            with col3:
                st.progress(row["similarity_score"], text=f"{sim_pct:.0f}% match · {cross}")
    else:
        st.info("No collaboration matches available.")

    st.markdown('<div class="section-header">Content Theme Overlap</div>', unsafe_allow_html=True)
    theme_dist = pd.crosstab(df["creator_handle"], df["theme_tag"], normalize="index")
    theme_dist.columns = [c.replace("_", " ").title() for c in theme_dist.columns]
    fig_heatmap = px.imshow(
        theme_dist.values,
        x=theme_dist.columns.tolist(),
        y=[f"@{c}" for c in theme_dist.index.tolist()],
        color_continuous_scale=["#F0F4F6", "#1B4B5A"],
        text_auto=".0%",
    )
    fig_heatmap.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="", yaxis_title="",
                              coloraxis_colorbar_title="Share")
    st.plotly_chart(fig_heatmap, use_container_width=True)


# ── TAB 4: model & predictions ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)

    baseline = model_results["baseline_logistic_regression"]
    tuned = model_results["tuned_random_forest"]

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### Baseline — Logistic Regression")
        st.metric("Test Accuracy", f"{baseline['test_accuracy']:.1%}")
        st.metric("Macro F1", f"{baseline['test_f1_macro']:.3f}")
        st.metric("CV Accuracy", f"{baseline['cv_accuracy_mean']:.1%} ± {baseline['cv_accuracy_std']:.3f}")
    with mc2:
        st.markdown("#### Tuned — Random Forest")
        st.metric("Test Accuracy", f"{tuned['test_accuracy']:.1%}")
        st.metric("Macro F1", f"{tuned['test_f1_macro']:.3f}")
        st.metric("CV Accuracy", f"{tuned['cv_accuracy_mean']:.1%}")
        if tuned.get("best_params"):
            st.caption(f"Best params: {tuned['best_params']}")

    st.markdown(f'<div class="callout"><strong>Model selection rationale:</strong> '
                f'{model_results["model_selection_rationale"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    st.caption("Which inputs mattered most to the Random Forest's predictions — used for explainability regardless of which model ships.")

    imp = model_results["feature_importance"]
    imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
    imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()
    imp_df = imp_df.sort_values("Importance", ascending=True).tail(10)

    fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[C["primary"]])
    fig_imp.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Relative Importance", yaxis_title="",
                          plot_bgcolor="white")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-header">Performance Tier Distribution</div>', unsafe_allow_html=True)
    tier_counts = features_df["performance_tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    tier_order = ["low", "expected", "breakout"]
    tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=tier_order, ordered=True)
    tier_counts = tier_counts.sort_values("Tier")
    tier_colors = {"low": C["muted"], "expected": C["primary"], "breakout": C["accent"]}

    fig_tier = px.bar(tier_counts, x="Tier", y="Count",
                      color="Tier", color_discrete_map=tier_colors)
    fig_tier.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=False, plot_bgcolor="white",
                           xaxis_title="Performance Tier", yaxis_title="Number of Posts")
    st.plotly_chart(fig_tier, use_container_width=True)


# ── TAB 5: raw data ────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Cleaned Post Data</div>', unsafe_allow_html=True)
    st.dataframe(
        cdf[["post_date", "theme_tag", "post_hour", "day_of_week", "length_seconds",
             "reach", "likes", "comments", "shares", "saves", "engagement_rate"]].sort_values("post_date", ascending=False),
        use_container_width=True, hide_index=True, height=500,
    )

    st.markdown('<div class="section-header">Model-Ready Feature Set</div>', unsafe_allow_html=True)
    creator_features = features_df[features_df["creator_handle"] == selected_creator]
    st.dataframe(creator_features, use_container_width=True, hide_index=True, height=400)


# ── footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#7A8E96; font-size:0.85rem;">'
    'CreatorSignal Chicago — Applied ML decision-support pipeline for content creators<br>'
    'Python · scikit-learn · Streamlit · Running on sample data'
    '</div>',
    unsafe_allow_html=True,
)
