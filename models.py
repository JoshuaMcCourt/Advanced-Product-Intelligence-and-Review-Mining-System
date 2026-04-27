# src/models.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Feature Splitting

def split_features(X: np.ndarray, n_structured: int) -> Tuple[np.ndarray, np.ndarray]:

    if X.shape[1] <= n_structured:
        raise ValueError("Invalid feature split: structured dimension exceeds X shape.")

    X_structured = X[:, :n_structured]
    X_embeddings = X[:, n_structured:]

    return X_structured, X_embeddings


# Safe Scaling

def scale_features(X: np.ndarray) -> Tuple[StandardScaler, np.ndarray]:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hard safety for NaN/inf leakage
    X_scaled = np.nan_to_num(X_scaled)

    return scaler, X_scaled


# Anomaly Detection

def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05
):

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    anomaly_score = model.decision_function(X)
    preds = model.predict(X)

    fake_flag = (preds == -1).astype(int)

    return model, anomaly_score, fake_flag


# Clustering

def train_kmeans(X: np.ndarray, n_clusters: int = 6):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X)

    # Distance to assigned centroid
    distances = np.linalg.norm(
        X - model.cluster_centers_[clusters],
        axis=1
    )

    sil_score = None
    if len(np.unique(clusters)) > 1:
        try:
            sil_score = silhouette_score(X, clusters)
        except Exception:
            sil_score = None

    return model, clusters, distances, sil_score


# Risk Feature Engineering

def build_risk_features(df: pd.DataFrame) -> np.ndarray:

    def safe(name: str, default: float = 0.0) -> np.ndarray:

        if name not in df.columns:
            return np.full(len(df), default)

        col = df[name]

        if col.isna().all():
            return np.full(len(df), default)

        return col.fillna(default).to_numpy()

    features = np.vstack([
        safe("anomaly_score"),
        safe("sentiment_rating_gap"),
        safe("cluster_distance"),
        safe("length_zscore"),
        safe("caps_ratio")
    ]).T

    return np.nan_to_num(features)


# Risk Scoring Engine

def compute_risk_score(df: pd.DataFrame) -> np.ndarray:


    # Anomaly Normalisation

    anomaly = df.get("anomaly_score", pd.Series(np.zeros(len(df))))

    anomaly_min = anomaly.min()
    anomaly_max = anomaly.max()

    anomaly_norm = 1 - (
        (anomaly - anomaly_min) /
        (anomaly_max - anomaly_min + 1e-6)
    )


    # Cluster Distance Normalisation

    dist = df.get("cluster_distance", pd.Series(np.zeros(len(df))))

    dist_min = dist.min()
    dist_max = dist.max()

    dist_norm = (
        (dist - dist_min) /
        (dist_max - dist_min + 1e-6)
    )


    # Sentiment Gap

    sentiment_gap = df.get(
        "sentiment_rating_gap",
        pd.Series(np.zeros(len(df)))
    )


    # Length Signal

    length_z = np.abs(df.get("length_zscore", pd.Series(np.zeros(len(df)))))


    # Weighted Fusion Score

    risk = (
        0.4 * anomaly_norm +
        0.3 * sentiment_gap +
        0.2 * dist_norm +
        0.1 * length_z
    )

    # Final normalization
    risk_min = risk.min()
    risk_max = risk.max()

    risk = (risk - risk_min) / (risk_max - risk_min + 1e-6)

    return np.nan_to_num(risk)


# Main Model Pipeline

def run_models(
    df: pd.DataFrame,
    X: np.ndarray,
    feature_cols: List[str],
    n_clusters: int = 6,
    contamination: float = 0.05
) -> Dict:

    df = df.copy()


    # Split Feature Space

    n_structured = len(feature_cols)

    X_structured, X_embeddings = split_features(X, n_structured)


    # Independent Scaling

    scaler_struct, Xs = scale_features(X_structured)
    scaler_embed, Xe = scale_features(X_embeddings)


    # Feature Fusion Layer

    X_final = np.hstack([
        Xs,
        Xe * 0.7  # Embedding down-weighting (stabilises clustering)
    ])

    X_final = np.nan_to_num(X_final)


    # Anomaly Model

    iso_model, anomaly_score, fake_flag = train_isolation_forest(
        X_final,
        contamination
    )

    df["anomaly_score"] = anomaly_score
    df["fake_review_flag"] = fake_flag


    # Clustering Model

    kmeans_model, clusters, distances, sil_score = train_kmeans(
        Xe,
        n_clusters
    )

    df["cluster"] = clusters
    df["cluster_distance"] = distances


    # Risk Engine

    df["risk_score"] = compute_risk_score(df)

    risk_features = build_risk_features(df)


    # Output Contract

    return {
        "df": df,
        "X_final": X_final,
        "risk_features": risk_features,

        "models": {
            "isolation_forest": iso_model,
            "kmeans": kmeans_model
        },

        "scalers": {
            "structured": scaler_struct,
            "embeddings": scaler_embed
        },

        "metrics": {
            "silhouette_score": sil_score
        }
    }