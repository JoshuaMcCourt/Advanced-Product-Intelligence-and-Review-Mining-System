# src/__init__.py

"""
Product Intelligence System

Production-grade NLP + ML pipeline for:

- Review sentiment + behavioural analysis
- Fake review detection (anomaly detection + probabilistic calibration)
- Product / topic clustering
- Explainable AI (review-level + cluster-level interpretability)
- Temporal anomaly detection (spikes, drift, burst patterns)
- Product-level risk scoring system


Architecture Overview

Core Pipeline:

    data_loader → preprocessing → features → models

Intelligence Layers:

    calibration → explainability → temporal_analysis

Evaluation Layer:

    evaluation (diagnostics, metrics, system validation)


Canonical Flow

    df = load_data(...)

    df = preprocess_text_column(df)

    feature_df = build_features(df)

    model_outputs = run_models(feature_df)

    calibrated_outputs = run_calibration(model_outputs)

    review_explanations = generate_review_explanations(calibrated_outputs)

    cluster_explanations = generate_cluster_explanations(calibrated_outputs)

    temporal_results = run_temporal_analysis(calibrated_outputs)

    evaluation_results = run_evaluation(calibrated_outputs)


Data Contracts

run_models(feature_df) must return a DataFrame containing:

    - anomaly_score        : float
    - cluster_label        : int
    - risk_features        : engineered signals derived from model outputs
    - sentiment_score      : float (if used downstream)
    - embedding features   : optional (for clustering / similarity)

run_calibration(model_outputs) expects:

    - anomaly_score
    - cluster_label
    - risk_features

and outputs:

    - fraud_probability    : calibrated probability (0-1)
    - final risk score     : optional composite metric


All downstream modules (explainability, temporal, evaluation)
assume these fields exist.


Design Principles

- Modular but pipeline-driven (clear stage boundaries)
- Deterministic preprocessing, probabilistic modeling
- Separation of:
    preprocessing vs feature engineering vs modeling
- Explainability as a first-class component
- Temporal dynamics treated as independent signal layer
- Reproducible outputs via config-driven execution


Notes

- run_pipeline is intentionally not exposed here
  to avoid circular imports and unintended execution
  in notebook / interactive environments.

- This module exposes the PUBLIC API of the system.
  Only stable, high-level functions are included.
"""


# Core Data Layer

from src.data_loader import load_data
from src.preprocessing import preprocess_text_column


# Feature Engineering

from src.features import build_features


# Modeling Layer

# NOTE:
# run_models performs:
# - Anomaly detection (e.g., Isolation Forest)
# - Clustering (e.g., KMeans)
# - Risk signal construction

from src.models import run_models


# Intelligence Layers

# Calibration: converts raw risk signals → calibrated probability
from src.calibration import run_calibration


# Explainability: human-interpretable outputs
from src.explainability import (
    generate_review_explanations,
    generate_cluster_explanations
)


# Temporal Analysis: time-based anomaly + drift detection
from src.temporal_analysis import run_temporal_analysis


# Evaluation Layer

from src.evaluation import run_evaluation


# Utilities

from src.utils import setup_logger, Timer


# Public API

__all__ = [
    # Core Pipeline
    "load_data",
    "preprocess_text_column",
    "build_features",
    "run_models",

    # Intelligence Layers
    "run_calibration",
    "generate_review_explanations",
    "generate_cluster_explanations",
    "run_temporal_analysis",

    # Evaluation
    "run_evaluation",

    # Utilities
    "setup_logger",
    "Timer",
]