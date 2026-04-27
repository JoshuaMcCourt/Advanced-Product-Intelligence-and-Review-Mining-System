# src/temporal_analysis.py

import pandas as pd
import numpy as np


# Internal Time Handling

def _prepare_time(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove invalid timestamps
    df = df.dropna(subset=["timestamp"])

    return df


# Review Volume (velocity)

def compute_review_velocity(df: pd.DataFrame, window: str = "7D") -> pd.DataFrame:

    df = _prepare_time(df)

    ts = (
        df.set_index("timestamp")
          .resample(window)
          .size()
          .rename("review_count")
          .reset_index()
    )

    # Smoothing for trend stability
    ts["review_count_smooth"] = ts["review_count"].rolling(
        window=3, min_periods=1
    ).mean()

    return ts


# Sentiment Drift Analysis

def compute_sentiment_drift(df: pd.DataFrame, window: str = "7D") -> pd.DataFrame:

    df = _prepare_time(df)

    if "sentiment" not in df.columns:
        raise ValueError("Missing required column: sentiment")

    ts = (
        df.set_index("timestamp")
          .resample(window)["sentiment"]
          .mean()
          .reset_index()
    )

    ts["sentiment_smooth"] = ts["sentiment"].rolling(
        window=3, min_periods=1
    ).mean()

    # Expanding baseline (cumulative expectation)
    ts["sentiment_baseline"] = ts["sentiment"].expanding().mean()

    ts["sentiment_drift"] = ts["sentiment"] - ts["sentiment_baseline"]

    return ts


# Fraud Spike Detection

def detect_fraud_spikes(df: pd.DataFrame, window: str = "7D") -> pd.DataFrame:

    df = _prepare_time(df)

    if "fake_review_flag" not in df.columns:
        raise ValueError("Missing required column: fake_review_flag")

    ts = (
        df.set_index("timestamp")
          .resample(window)["fake_review_flag"]
          .mean()
          .reset_index()
    )


    # Local Baseline (rolling)

    ts["baseline"] = ts["fake_review_flag"].rolling(
        window=3, min_periods=1
    ).mean()

    ts["deviation"] = ts["fake_review_flag"] - ts["baseline"]


    # Robust Z-score

    mean = ts["fake_review_flag"].mean()
    std = ts["fake_review_flag"].std()

    std = max(std, 1e-6)

    ts["zscore"] = (ts["fake_review_flag"] - mean) / std


    # Spike Flag (robust threshold)

    # Slightly more stable than strict z > 2 in sparse data
    ts["spike"] = ts["zscore"] > 2.0

    return ts


# Full Temporal Pipeline

def run_temporal_analysis(df: pd.DataFrame) -> dict:

    """
    Full temporal intelligence pipeline.

    Produces multi-resolution signals:
    - 1D: high sensitivity (micro spikes)
    - 7D: smoothed trend (macro behaviour)
    """

    return {

        # Review Velocity
        "review_velocity_7d": compute_review_velocity(df, "7D"),
        "review_velocity_1d": compute_review_velocity(df, "1D"),

        # Sentiment Drift
        "sentiment_drift_7d": compute_sentiment_drift(df, "7D"),
        "sentiment_drift_1d": compute_sentiment_drift(df, "1D"),

        # Fraud Spikes
        "fraud_spikes_7d": detect_fraud_spikes(df, "7D"),
        "fraud_spikes_1d": detect_fraud_spikes(df, "1D"),
    }