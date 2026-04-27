# run_pipeline.py

import yaml
import importlib
from pathlib import Path

from src.pipeline import run_pipeline
from src.utils import (
    save_dataframe,
    save_json,
    save_model,
    setup_logger
)


# Config Loading

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Directory Safety Layer

def ensure_dirs(config: dict):
    """
    Ensures all required output directories exist.
    Prevents silent failures during artifact saving.
    """

    Path(config["output"]["processed_data_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["model_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["report_path"]).mkdir(parents=True, exist_ok=True)


# Main Execution Entry

def main():


    # Setup

    config = load_config()
    logger = setup_logger()

    logger.info("Starting Product Intelligence System")

    ensure_dirs(config)


    # Input Paths

    reviews_path = config["data"]["data_path"]
    metadata_path = config["data"].get("metadata_path")

    output_data_path = config["output"]["processed_data_path"]
    model_path = config["output"]["model_path"]
    report_path = config["output"]["report_path"]


    # Pipeline Execution

    logger.info("Running ML pipeline...")

    results = run_pipeline(
        reviews_path=reviews_path,
        metadata_path=metadata_path,
        n_clusters=config["model"]["n_clusters"],
        contamination=config["model"]["contamination"],
        verbose=True
    )

    df = results["data"]

    logger.info(f"Pipeline complete. Processed rows: {len(df)}")


    # Artifact Saving

    logger.info("Saving outputs...")


    # Dataset

    save_dataframe(
        df,
        f"{output_data_path}/processed_reviews.csv"
    )


    # Models

    models = results.get("models", {})

    if "isolation_forest" in models:
        save_model(
            models["isolation_forest"],
            f"{model_path}/isolation_forest.pkl"
        )

    if "kmeans" in models:
        save_model(
            models["kmeans"],
            f"{model_path}/kmeans.pkl"
        )

    if results.get("calibration_model") is not None:
        save_model(
            results["calibration_model"],
            f"{model_path}/calibration_model.pkl"
        )


    # Scalers (safe iteration)

    for name, scaler in results.get("scalers", {}).items():
        if scaler is not None:
            save_model(
                scaler,
                f"{model_path}/scaler_{name}.pkl"
            )


    # Reports (safe writes)

    if "evaluation" in results:
        save_json(
            results["evaluation"],
            f"{report_path}/evaluation.json"
        )

    if "cluster_explanations" in results:
        save_json(
            results["cluster_explanations"],
            f"{report_path}/cluster_explanations.json"
        )

    if "temporal_analysis" in results:
        save_json(
            results["temporal_analysis"],
            f"{report_path}/temporal_analysis.json"
        )


    # Tables

    if "review_explanations" in results:
        save_dataframe(
            results["review_explanations"],
            f"{report_path}/review_explanations.csv"
        )


    # Completion Log

    logger.info("All outputs saved successfully")
    logger.info("Pipeline finished successfully")


# Entry Point

if __name__ == "__main__":
    main()