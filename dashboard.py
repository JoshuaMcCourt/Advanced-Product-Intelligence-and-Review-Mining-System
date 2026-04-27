# app/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# Page Config

st.set_page_config(
    page_title="Product Intelligence System",
    layout="wide"
)

st.title("Product Intelligence & Trust System")
st.markdown(
    "Fraud detection • Product health • Explainability • Temporal monitoring"
)


# Data Loading

@st.cache_data(show_spinner=True)
def load_data():
    """
    Loads processed pipeline output safely.
    """

    path = "data/processed/processed_reviews.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Missing dataset at {path}. Run pipeline first.")
        return pd.DataFrame()

    return df


df = load_data()

if df.empty:
    st.stop()


# Safety Normalization

REQUIRED_COLS = [
    "risk_score",
    "sentiment",
    "cluster",
    "fake_review_flag"
]

for col in REQUIRED_COLS:
    if col not in df.columns:
        df[col] = 0


# Sidebar Filters

st.sidebar.header("Filters")

min_risk = st.sidebar.slider(
    "Minimum Risk Score",
    0.0, 1.0, 0.5
)

cluster_options = ["All"] + sorted(df["cluster"].dropna().unique().tolist())

selected_cluster = st.sidebar.selectbox(
    "Cluster",
    options=cluster_options
)


# Apply filters
filtered_df = df[df["risk_score"] >= min_risk]

if selected_cluster != "All":
    filtered_df = filtered_df[filtered_df["cluster"] == selected_cluster]


# KPI Section

st.subheader("System Overview")

def safe_mean(col):
    return round(filtered_df[col].mean(), 3) if len(filtered_df) > 0 else 0


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Avg Risk Score", safe_mean("risk_score"))
col3.metric("Fake Review Rate", safe_mean("fake_review_flag"))
col4.metric("Avg Sentiment", safe_mean("sentiment"))


# Risk Distribution

st.subheader("Risk Distribution")

fig = px.histogram(
    filtered_df,
    x="risk_score",
    nbins=50,
    title="Risk Score Distribution"
)

st.plotly_chart(fig, use_container_width=True)


# Cluster Distribution

st.subheader("Cluster Analysis")

cluster_counts = (
    filtered_df["cluster"]
    .value_counts()
    .reset_index()
)

cluster_counts.columns = ["cluster", "count"]

fig2 = px.bar(
    cluster_counts,
    x="cluster",
    y="count",
    title="Cluster Distribution"
)

st.plotly_chart(fig2, use_container_width=True)


# Sentiment vs Risk

st.subheader("Sentiment vs Risk")

fig3 = px.scatter(
    filtered_df,
    x="sentiment",
    y="risk_score",
    color="cluster",
    title="Sentiment vs Risk Score"
)

st.plotly_chart(fig3, use_container_width=True)


# Temporal View

st.subheader("Temporal Intelligence")

if "timestamp" in filtered_df.columns and len(filtered_df) > 0:

    temp = filtered_df.copy()

    temp["timestamp"] = pd.to_datetime(temp["timestamp"], errors="coerce")
    temp = temp.dropna(subset=["timestamp"])

    trend = (
        temp.set_index("timestamp")
        .resample("7D")["risk_score"]
        .mean()
        .reset_index()
    )

    fig4 = px.line(
        trend,
        x="timestamp",
        y="risk_score",
        title="Risk Trend Over Time"
    )

    st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("No timestamp data available for temporal analysis.")


# High Risk Reviews

st.subheader("High Risk Reviews")

top_risk = filtered_df.sort_values(
    "risk_score",
    ascending=False
).head(10)

display_cols = [
    "review_text",
    "risk_score",
    "sentiment",
    "fake_review_flag",
    "cluster"
]

existing_cols = [c for c in display_cols if c in top_risk.columns]

st.dataframe(top_risk[existing_cols])


# Explainability Viewer

st.subheader("Explainability")

if "explanation" in filtered_df.columns:

    valid_idx = filtered_df.index[: min(100, len(filtered_df))]

    idx = st.selectbox(
        "Select Review Index",
        valid_idx
    )

    st.json(filtered_df.loc[idx, "explanation"])

else:
    st.info("Run pipeline with explainability enabled to view explanations.")


# Product Intelligence

if "product_id" in filtered_df.columns:

    st.subheader("Product Intelligence")

    product_scores = filtered_df.groupby("product_id").agg({
        "risk_score": "mean",
        "sentiment": "mean",
        "fake_review_flag": "mean"
    }).reset_index()

    fig5 = px.scatter(
        product_scores,
        x="sentiment",
        y="risk_score",
        size="fake_review_flag",
        title="Product Health Map"
    )

    st.plotly_chart(fig5, use_container_width=True)


# Raw Data Explorer

st.subheader("Raw Data Explorer")

st.dataframe(filtered_df.head(100))


st.success("Dashboard loaded successfully")