# src/preprocessing.py

import re
import pandas as pd
import numpy as np
from typing import Optional

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils import zscore


# Initialization

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    # Fallback for environments without nltk data downloaded
    stop_words = set()

lemmatizer = WordNetLemmatizer()


# Precompiled regex (performance improvement)
URL_PATTERN = re.compile(r"http\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-z0-9\s]")
PUNCT_PATTERN = re.compile(r"[!?.,]")


# Language Detection

def is_english(text: str) -> bool:
    text = str(text)

    if not text or len(text) == 0:
        return False

    alpha_chars = re.findall(r"[a-zA-Z]", text)
    alpha_ratio = len(alpha_chars) / max(len(text), 1)

    return alpha_ratio > 0.7


# Text Cleaning

def clean_text(text: str) -> str:

    text = str(text).lower().strip()

    # Fast regex cleaning
    text = URL_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = NON_ALPHANUM_PATTERN.sub(" ", text)

    words = text.split()

    cleaned_words = []
    for w in words:
        if w in stop_words:
            continue
        if len(w) <= 1:
            continue

        cleaned_words.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned_words)


# Feature Extraction

def extract_text_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:

    df = df.copy()
    text_series = df[text_column].astype(str)


    # Structural Features

    df["char_length"] = text_series.str.len()
    df["word_count"] = text_series.str.split().apply(len)

    df["avg_word_length"] = np.where(
        df["word_count"] > 0,
        df["char_length"] / df["word_count"],
        0.0
    )


    # Stylistic Features

    df["caps_ratio"] = text_series.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )

    df["punctuation_density"] = text_series.apply(
        lambda x: len(PUNCT_PATTERN.findall(x)) / max(len(x), 1)
    )

    df["exclamation_count"] = text_series.str.count("!")


    # Behavioural Features (normalised signals)

    df["length_zscore"] = zscore(df["char_length"])
    df["word_count_zscore"] = zscore(df["word_count"])


    # Risk-oriented Heuristics

    df["short_text_flag"] = (df["char_length"] < 30).astype(int)
    df["high_caps_flag"] = (df["caps_ratio"] > 0.3).astype(int)
    df["excess_punctuation_flag"] = (df["punctuation_density"] > 0.1).astype(int)

    return df


# Main Pipeline

def preprocess_text_column(
    df: pd.DataFrame,
    text_column: str = "review_text",
    min_length: int = 20,
    remove_duplicates: bool = True,
    filter_non_english: bool = False,
    verbose: bool = True
) -> pd.DataFrame:

    df = df.copy()

    if verbose:
        print("Starting preprocessing pipeline...")


    # Validation

    if text_column not in df.columns:
        raise ValueError(f"Missing required column: {text_column}")

    df[text_column] = df[text_column].astype(str)
    df = df.dropna(subset=[text_column])


    # Deduplication

    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[text_column])

        if verbose:
            print(f"Duplicates removed: {before - len(df)}")


    # Length Filter

    df = df[df[text_column].str.len() > min_length]


    # Language Filter

    if filter_non_english:
        df = df[df[text_column].apply(is_english)]


    # Feature Extraction (raw text)

    df = extract_text_features(df, text_column)


    # Clean Text Generation

    df["clean_text"] = df[text_column].apply(clean_text)

    # Remove only unusable rows
    df = df[df["clean_text"].str.len() > 0]


    # Numeric Safety Handling

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Median imputation (robust to outliers)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


    # Final Sanity Check

    if verbose:
        print(f"Preprocessing complete. Final shape: {df.shape}")

    return df