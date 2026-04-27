# src/features.py

import numpy as np
import pandas as pd
from typing import Tuple, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


# Model Initialization

sentiment_model = SentimentIntensityAnalyzer()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Sentiment Features

def compute_sentiment(texts: List[str]) -> np.ndarray:

    if not texts:
        return np.array([])

    return np.array([
        sentiment_model.polarity_scores(str(t))["compound"]
        for t in texts
    ], dtype=np.float32)


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "clean_text" not in df.columns:
        raise ValueError("Missing required column: clean_text")

    df["sentiment"] = compute_sentiment(df["clean_text"].tolist())

    df["rating_norm"] = df["rating"] / 5.0

    df["sentiment_rating_gap"] = np.abs(
        df["sentiment"] - df["rating_norm"]
    )

    return df


# Embeddings

def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # L2 normalization (stabilises clustering + anomaly detection)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)

    embeddings = embeddings / norms

    return embeddings


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "clean_text" not in df.columns:
        raise ValueError("Missing required column: clean_text")

    texts = df["clean_text"].fillna("").tolist()

    embeddings = generate_embeddings(texts)

    if len(embeddings) != len(df):
        raise ValueError("Embedding-row mismatch detected")

    df["embedding"] = list(embeddings)
    df["embedding_norm"] = np.linalg.norm(embeddings, axis=1)

    return df


# Derived Features

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required_cols = [
        "char_length",
        "word_count"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required feature column: {c}")

    df["word_density"] = df["word_count"] / (df["char_length"] + 1e-6)

    # Important:
    # I don't recompute z-scores here if already computed in preprocessing.
    # Will only create fallback if missing.

    if "length_zscore" not in df.columns:
        df["length_zscore"] = (
            (df["char_length"] - df["char_length"].mean()) /
            (df["char_length"].std() + 1e-6)
        )

    if "word_count_zscore" not in df.columns:
        df["word_count_zscore"] = (
            (df["word_count"] - df["word_count"].mean()) /
            (df["word_count"].std() + 1e-6)
        )

    return df


# Feature Matrix Builder

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:

    feature_cols = [
        # Sentiment features
        "sentiment",
        "rating_norm",
        "sentiment_rating_gap",

        # Structural features
        "char_length",
        "word_count",
        "avg_word_length",

        # Stylistic features
        "caps_ratio",
        "punctuation_density",
        "exclamation_count",

        # Behavioral features
        "length_zscore",
        "word_count_zscore",
        "word_density",

        # Embedding metadata
        "embedding_norm"
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X_structured = df[feature_cols].to_numpy(dtype=np.float32)

    if "embedding" not in df.columns:
        raise ValueError("Missing embeddings for feature matrix construction")

    embeddings = np.stack(df["embedding"].values).astype(np.float32)

    if X_structured.shape[0] != embeddings.shape[0]:
        raise ValueError("Feature mismatch between structured data and embeddings")

    X = np.hstack([X_structured, embeddings])

    return X, feature_cols


# Full Pipeline

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:

    df = df.copy()

    df = add_sentiment_features(df)
    df = add_embeddings(df)
    df = add_derived_features(df)

    X, feature_cols = build_feature_matrix(df)

    return df, X, feature_cols