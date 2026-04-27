# src/utils.py

import os
import json
import logging
import pandas as pd
from datetime import datetime


# Logging Setup

def setup_logger(name: str = "product_intelligence") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# File Saving Utilities

def _safe_mkdir(path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _append_timestamp(path: str) -> str:
    base, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


def save_dataframe(df: pd.DataFrame, path: str, timestamp: bool = False):
    if timestamp:
        path = _append_timestamp(path)

    _safe_mkdir(path)
    df.to_csv(path, index=False, encoding="utf-8")


def save_json(obj: dict, path: str, timestamp: bool = False):
    if timestamp:
        path = _append_timestamp(path)

    _safe_mkdir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, default=str)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Model Persistence

def save_model(model, path: str):
    import joblib
    _safe_mkdir(path)
    joblib.dump(model, path)


def load_model(path: str):
    import joblib
    return joblib.load(path)


# Feature Utilities

def normalize_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    denom = max_val - min_val

    if denom == 0:
        return pd.Series(0.0, index=series.index)

    return (series - min_val) / (denom + 1e-9)


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series(0.0, index=series.index)

    return (series - mean) / (std + 1e-9)


# Timing / Profiling

class Timer:
    def __init__(self, name="block", logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        end = datetime.now()
        duration = (end - self.start).total_seconds()

        message = f"{self.name} took {duration:.4f} seconds"

        if self.logger:
            self.logger.info(message)
        else:
            print(message)