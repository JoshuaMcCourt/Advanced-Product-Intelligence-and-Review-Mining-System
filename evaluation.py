# src/evaluation.py

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Optional

from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    average_precision_score
)


# Dataset Diagnostics

def compute_basic_metrics(df: pd.DataFrame) -> Dict:

    return {
        "total_reviews": int(len(df)),
        "fake_rate": float(df["fake_review_flag"].mean()) if "fake_review_flag" in df else 0.0,
        "avg_sentiment": float(df["sentiment"].mean()) if "sentiment" in df else 0.0,
        "avg_rating": float(df["rating"].mean()) if "rating" in df else 0.0,
        "avg_risk_score": float(df["risk_score"].mean()) if "risk_score" in df else 0.0
    }


def feature_distribution_summary(df: pd.DataFrame) -> Dict:

    def safe(col: str, fn=np.std):
        return float(fn(df[col])) if col in df and len(df[col]) > 0 else 0.0

    return {
        "sentiment_std": safe("sentiment", np.std),
        "rating_std": safe("rating", np.std),
        "risk_score_std": safe("risk_score", np.std),
        "sentiment_skew": float(df["sentiment"].skew()) if "sentiment" in df else 0.0
    }


# Correlation Analysis

def correlation_analysis(df: pd.DataFrame) -> Dict:

    cols = [c for c in [
        "sentiment",
        "rating",
        "risk_score",
        "anomaly_score"
    ] if c in df.columns]

    if len(cols) < 2:
        return {}

    return df[cols].corr(numeric_only=True).to_dict()


# Anomaly Evaluation

def evaluate_anomaly_detection(df: pd.DataFrame) -> Dict:

    if "fake_review_flag" not in df.columns or "anomaly_score" not in df.columns:
        return {}

    y_true = df["fake_review_flag"]
    y_score = df["anomaly_score"]

    # Ensure valid binary labels
    if y_true.nunique() < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "note": "Insufficient label variance"
        }

    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = None

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = None

    return {
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None
    }


# Clustering Evaluation

def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict:

    results = {}

    # Silhouette requires > 1 cluster
    if len(np.unique(labels)) > 1:
        try:
            results["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            results["silhouette"] = None
    else:
        results["silhouette"] = None

    # Entropy stability
    counts = np.array(list(Counter(labels).values()), dtype=np.float64)

    if counts.sum() == 0:
        results["cluster_entropy"] = 0.0
        return results

    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log(probs + 1e-9))

    results["cluster_entropy"] = float(entropy)

    return results


# Cluster Interpretation

def get_cluster_keywords(df: pd.DataFrame, top_n: int = 10) -> Dict:

    if "cluster" not in df.columns or "clean_text" not in df.columns:
        return {}

    cluster_words = {}

    for c in sorted(df["cluster"].unique()):

        cluster_df = df[df["cluster"] == c]

        text = " ".join(cluster_df["clean_text"].astype(str))

        words = [w for w in text.split() if len(w) > 2]

        cluster_words[int(c)] = Counter(words).most_common(top_n)

    return cluster_words


# Product Intelligence

def compute_product_scores(df: pd.DataFrame) -> Optional[pd.DataFrame]:

    if "product_id" not in df.columns:
        return None

    agg_dict = {}

    for col in ["sentiment", "rating", "fake_review_flag", "risk_score"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    agg_dict["review_id"] = "count" if "review_id" in df.columns else "size"

    product_scores = df.groupby("product_id").agg(agg_dict)

    product_scores = product_scores.rename(columns={
        "review_id": "review_count"
    })

    if "sentiment" in product_scores and "rating" in product_scores:

        product_scores["quality_score"] = (
            product_scores["sentiment"] * 0.4 +
            (product_scores["rating"] / 5.0) * 0.4 -
            product_scores.get("fake_review_flag", 0) * 0.2
        )

    return product_scores


# High Risk Extraction

def get_high_risk_reviews(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    if "risk_score" not in df.columns:
        return pd.DataFrame()

    return df.sort_values("risk_score", ascending=False).head(top_n)


# Risk Validation

def evaluate_risk_score(df: pd.DataFrame) -> Dict:

    def safe_corr(a, b):
        if a in df and b in df:
            return float(df[a].corr(df[b]))
        return None

    return {
        "risk_vs_sentiment_corr": safe_corr("risk_score", "sentiment"),
        "risk_vs_rating_corr": safe_corr("risk_score", "rating"),
        "risk_distribution_std": float(df["risk_score"].std()) if "risk_score" in df else 0.0
    }


# Full Evaluation Pipeline

def run_evaluation(df: pd.DataFrame, X: np.ndarray = None) -> Dict:

    results = {}

    results["basic_metrics"] = compute_basic_metrics(df)
    results["distribution"] = feature_distribution_summary(df)

    results["correlation"] = correlation_analysis(df)

    results["anomaly_eval"] = evaluate_anomaly_detection(df)

    if X is not None and "cluster" in df.columns:
        results["clustering"] = evaluate_clustering(X, df["cluster"].values)

    results["cluster_keywords"] = get_cluster_keywords(df)

    results["product_scores"] = compute_product_scores(df)

    results["high_risk_reviews"] = get_high_risk_reviews(df)

    results["risk_validation"] = evaluate_risk_score(df)

    return results