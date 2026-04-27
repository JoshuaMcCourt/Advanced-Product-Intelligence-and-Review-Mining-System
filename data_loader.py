# src/data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# Schema Contract

REQUIRED_COLS = [
    "review_id",
    "product_id",
    "user_id",
    "review_text",
    "rating",
    "timestamp"
]


def validate_schema(df: pd.DataFrame):
    """
    Final safety check after full ingestion pipeline.
    Ensures downstream modules never receive malformed data.
    """

    missing = [c for c in REQUIRED_COLS if c not in df.columns]

    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")


# Schema Standardisation

def _standardise_schema(df: pd.DataFrame) -> pd.DataFrame:

    column_map = {
        # Amazon
        "reviews.text": "review_text",
        "reviews.rating": "rating",
        "reviews.title": "summary",

        # Kaggle variants
        "Text": "review_text",
        "Score": "rating",
        "Summary": "summary",

        # Alternative naming
        "reviewText": "review_text",
        "overall": "rating",
        "stars": "rating"
    }

    df = df.copy()

    for src, dst in column_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})

    return df


# Timestamp Handling

def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    timestamp_sources = ["timestamp", "reviewTime", "Time"]

    parsed = False

    for col in timestamp_sources:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
            parsed = True
            break

    if not parsed:
        df["timestamp"] = pd.date_range(
            start="2020-01-01",
            periods=len(df),
            freq="h"
        )

    # No invalid timestamps
    df = df.dropna(subset=["timestamp"])

    return df


# Cleaning Pipeline

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Text normalization
    df["review_text"] = df["review_text"].astype(str).str.strip()

    # Numeric safety
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Remove invalid rows early
    df = df.dropna(subset=["review_text", "rating"])

    # Deduplication (text-level assumption)
    df = df.drop_duplicates(subset=["review_text"])

    # Length filter (noise reduction)
    df = df[df["review_text"].str.len() > 5]

    # Enforce rating bounds
    df["rating"] = df["rating"].clip(1, 5)

    return df


# ID Generation (Deterministic)

def _generate_ids(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    n = len(df)

    if "review_id" not in df.columns:
        df["review_id"] = np.arange(n)

    if "product_id" not in df.columns:
        df["product_id"] = [
            f"product_{i % 100}" for i in range(n)
        ]

    if "user_id" not in df.columns:
        df["user_id"] = [
            f"user_{i % 500}" for i in range(n)
        ]

    return df


# Main Review Loader

def load_reviews(path: str, verbose: bool = True) -> pd.DataFrame:

    df = pd.read_csv(path)

    if verbose:
        print("Raw columns:", list(df.columns))

    # Schema alignment
    df = _standardise_schema(df)

    # Hard requirement before proceeding
    if "review_text" not in df.columns:
        raise ValueError("Missing required column: review_text after schema standardisation")

    # Safe default ONLY at ingestion boundary
    if "rating" not in df.columns:
        df["rating"] = 3.0

    # Cleaning
    df = _clean_data(df)

    # Timestamps
    df = _parse_timestamp(df)

    # IDs
    df = _generate_ids(df)

    # Final contract enforcement
    validate_schema(df)

    if verbose:
        print("Final columns:", list(df.columns))
        print(f"Final shape: {df.shape}")

    return df


# Metadata Loader

def load_metadata(path: str, verbose: bool = True) -> Optional[pd.DataFrame]:

    try:
        df_meta = pd.read_csv(path)

        if verbose:
            print("Metadata columns:", list(df_meta.columns))

        return df_meta

    except FileNotFoundError:
        if verbose:
            print("Metadata not found — continuing without it.")
        return None


# Public Entry Point

def load_data(
    reviews_path: str,
    metadata_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:

    df_reviews = load_reviews(reviews_path, verbose)
    df_meta = load_metadata(metadata_path, verbose) if metadata_path else None

    return df_reviews, df_meta