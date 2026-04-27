# src/explainability.py

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict


# Risk Banding

def get_risk_band(prob: float) -> str:

    if prob >= 0.8:
        return "high"
    elif prob >= 0.5:
        return "medium"
    else:
        return "low"


# Global Statistics Builder

def compute_global_stats(df: pd.DataFrame) -> Dict:

    """
    Computes stable thresholds for explanation logic.
    Must be computed ON training data in production.
    """

    required_cols = [
        "anomaly_score",
        "sentiment_rating_gap",
        "caps_ratio",
        "cluster_distance"
    ]

    stats = {}

    for col in required_cols:
        if col in df.columns:
            stats[col] = {
                "p10": df[col].quantile(0.1),
                "p90": df[col].quantile(0.9),
                "p95": df[col].quantile(0.95)
            }

    return stats


# Review-level Explanation

def explain_review(row: pd.Series, stats: Dict) -> dict:

    reasons = []


    # Anomaly Signal

    if "anomaly_score" in row and "anomaly_score" in stats:
        if row["anomaly_score"] < stats["anomaly_score"]["p10"]:
            reasons.append("Highly unusual behavioural pattern")


    # Sentiment Mismatch

    if "sentiment_rating_gap" in row and "sentiment_rating_gap" in stats:
        if row["sentiment_rating_gap"] > stats["sentiment_rating_gap"]["p90"]:
            reasons.append("Strong mismatch between sentiment and rating")


    # Length Signal

    if "length_zscore" in row:
        if abs(row["length_zscore"]) > 2:
            reasons.append("Unusual review length distribution")


    # Caps / Spam Signal

    if "caps_ratio" in row and "caps_ratio" in stats:
        if row["caps_ratio"] > stats["caps_ratio"]["p95"]:
            reasons.append("Excessive capitalization (spam-like pattern)")


    # Cluster Outlier

    if "cluster_distance" in row and "cluster_distance" in stats:
        if row["cluster_distance"] > stats["cluster_distance"]["p90"]:
            reasons.append("Semantic outlier within cluster")


    # Final Risk Score

    prob = float(row.get("fraud_probability", 0.0))
    risk_band = get_risk_band(prob)

    return {
        "risk_score": float(row.get("risk_score", 0.0)),
        "fraud_probability": prob,
        "risk_level": risk_band,
        "reasons": reasons if reasons else ["No strong anomaly signals detected"]
    }


# Batch Review Explanation

def generate_review_explanations(
    df: pd.DataFrame,
    top_n: int = 10,
    use_training_stats: bool = False,
    stats: Dict = None
) -> pd.DataFrame:

    df = df.copy()

    # If no external stats provided, compute from current dataset
    # However, in production, stats MUST come from training data
    if stats is None:
        stats = compute_global_stats(df)

    # Rank high-risk reviews
    top_df = df.sort_values("risk_score", ascending=False).head(top_n).copy()

    top_df["explanation"] = top_df.apply(
        lambda row: explain_review(row, stats),
        axis=1
    )

    return top_df[[
        "review_text",
        "risk_score",
        "fraud_probability",
        "explanation"
    ]]


# Cluster Explanation

def generate_cluster_explanations(
    df: pd.DataFrame,
    top_n_words: int = 10
) -> dict:

    cluster_summary = {}

    if "cluster" not in df.columns:
        raise ValueError("Missing required column: cluster")

    for cluster_id in sorted(df["cluster"].unique()):

        cluster_df = df[df["cluster"] == cluster_id]


        # Text Signal Extraction

        if "clean_text" in cluster_df.columns:
            text = " ".join(cluster_df["clean_text"].astype(str))
            words = text.split()

            # Filter noisy tokens
            words = [w for w in words if len(w) > 2]
            top_words = Counter(words).most_common(top_n_words)
        else:
            top_words = []

 
        # Metrics (safe access)

        def safe_mean(col):
            return float(cluster_df[col].mean()) if col in cluster_df else 0.0

        def safe_rate(condition):
            return float(condition.mean()) if len(condition) > 0 else 0.0

        avg_risk = safe_mean("risk_score")
        avg_sentiment = safe_mean("sentiment")

        fraud_rate = safe_mean("fake_review_flag") if "fake_review_flag" in cluster_df else 0.0

        high_risk_ratio = (
            safe_rate(cluster_df["fraud_probability"] > 0.8)
            if "fraud_probability" in cluster_df
            else 0.0
        )

        cluster_summary[int(cluster_id)] = {
            "size": int(len(cluster_df)),
            "top_words": top_words,
            "avg_risk": avg_risk,
            "avg_sentiment": avg_sentiment,
            "fraud_rate": fraud_rate,
            "high_risk_ratio": high_risk_ratio
        }

    return cluster_summary