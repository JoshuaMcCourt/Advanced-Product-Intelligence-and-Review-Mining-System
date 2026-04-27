# src/pipeline.py

import pandas as pd
import numpy as np
import random

from src.data_loader import load_data
from src.preprocessing import preprocess_text_column
from src.features import build_features
from src.models import run_models
from src.evaluation import run_evaluation

from src.explainability import (
    generate_review_explanations,
    generate_cluster_explanations
)

from src.temporal_analysis import run_temporal_analysis
from src.calibration import run_calibration

from src.utils import setup_logger, Timer, save_json, save_model


# Reproducibility

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


# Pipeline Orchestrator

def run_pipeline(
    reviews_path: str,
    metadata_path: str = None,
    n_clusters: int = 6,
    contamination: float = 0.05,
    verbose: bool = True,
    save_artifacts: bool = True,
    seed: int = 42
):

    set_seed(seed)

    logger = setup_logger()
    logger.info("Starting Product Intelligence Pipeline")


    # Data Loading

    with Timer("Data Loading"):
        df_reviews, df_meta = load_data(
            reviews_path,
            metadata_path,
            verbose=verbose
        )

    logger.info(f"Data loaded: {df_reviews.shape[0]} rows")
    logger.info(f"Columns detected: {df_reviews.columns.tolist()}")


    # Preprocessing

    with Timer("Preprocessing"):
        df = preprocess_text_column(
            df_reviews,
            text_column="review_text"
        )

    logger.info(f"Preprocessing complete: {df.shape[0]} rows")


    # Feature Engineering

    with Timer("Feature Engineering"):
        df, X, feature_cols = build_features(df)

    logger.info(f"Feature engineering complete: {len(feature_cols)} features")


    # Modeling

    with Timer("Model Training"):
        model_outputs = run_models(
            df=df,
            X=X,
            feature_cols=feature_cols,
            n_clusters=n_clusters,
            contamination=contamination
        )

    df = model_outputs["df"]

    logger.info("Model training complete")


    # Calibration Block

    with Timer("Calibration"):

        risk_features = model_outputs["risk_features"]

        if "fake_review_flag" not in df.columns:
            logger.warning("Missing fake_review_flag — skipping calibration")

            calibration_model = None
            calibration_scaler = None
            df["fraud_probability"] = 0.0

        else:
            y_dummy = df["fake_review_flag"].values
            unique_classes = np.unique(y_dummy)

            if len(unique_classes) < 2:
                logger.warning(
                    "Calibration skipped: only one class present in fake_review_flag"
                )

                calibration_model = None
                calibration_scaler = None
                df["fraud_probability"] = 0.0

            else:
                # Capture all outputs
                calibration_model, calibration_scaler, fraud_probs = run_calibration(
                    risk_features,
                    y_dummy
                )

                df["fraud_probability"] = fraud_probs

    logger.info("Calibration complete")

    # Temporal Analysis

    with Timer("Temporal Analysis"):
        temporal_results = run_temporal_analysis(df)

    logger.info("Temporal analysis complete")


    # Explainability

    with Timer("Explainability"):
        review_explanations = generate_review_explanations(df)
        cluster_explanations = generate_cluster_explanations(df)

    logger.info("Explainability complete")


    # Evaluation

    with Timer("Evaluation"):
        evaluation_results = run_evaluation(
            df=df,
            X=model_outputs["X_final"]
        )

    logger.info("Evaluation complete")

    # Visualisation

    from src.visualization import (
    plot_risk_distribution,
    plot_cluster_distribution
    )

    plot_risk_distribution(df, "outputs/charts")
    plot_cluster_distribution(df, "outputs/charts")


    # Artifact Persistence

    if save_artifacts:

        logger.info("Saving artifacts...")

        save_json(evaluation_results, "outputs/reports/evaluation.json")
        save_json(temporal_results, "outputs/reports/temporal_analysis.json")
        save_json(feature_cols, "outputs/reports/feature_columns.json")

        save_model(
            model_outputs["models"]["isolation_forest"],
            "models/isolation_forest.pkl"
        )

        save_model(
            model_outputs["models"]["kmeans"],
            "models/kmeans.pkl"
        )

        save_model(
            model_outputs["scalers"]["structured"],
            "models/scaler.pkl"
        )


    # Final Output Package

    results = {
        "data": df,

        "models": model_outputs["models"],

        "scalers": {
            **model_outputs["scalers"],
            "calibration": calibration_scaler
        },

        "calibration_model": calibration_model,

        "temporal_analysis": temporal_results,
        "review_explanations": review_explanations,
        "cluster_explanations": cluster_explanations,

        "evaluation": evaluation_results,
        "feature_columns": feature_cols
    }

    logger.info("Pipeline execution complete")

    return results