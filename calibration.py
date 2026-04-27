# src/calibration.py

import numpy as np
from typing import Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


# Train Calibration Model

def train_calibration_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[CalibratedClassifierCV, StandardScaler]:

    """
    Trains a calibrated probability model for fraud/risk scoring.

    Uses:
    - Logistic Regression (base model)
    - Platt scaling via CalibratedClassifierCV (true calibration layer)
    """


    # Safety Checks

    if len(np.unique(y)) < 2:
        raise ValueError("Calibration failed: only one class present in y.")

    if X.shape[0] != len(y):
        raise ValueError("X and y size mismatch.")


    # Train/test Split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


    # Feature Scaling

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    
    # Important:
    # Calibration must be trained ONLY on training data 
    # to avoid probability leakage

    base_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )


    # Probability Calibration

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",   # Platt scaling (stable for medium datasets)
        cv=3
    )

    calibrated_model.fit(X_train_scaled, y_train)

    return calibrated_model, scaler


# Prediction

def calibrate_risk_model(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    X: np.ndarray
) -> np.ndarray:

    """
    Returns calibrated probabilities (0-1).
    """

    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]

    return probs


# Full Pipeline

def run_calibration(
    X: np.ndarray,
    y: np.ndarray,
    return_model: bool = True
):
    """
    Full calibration pipeline:

    Returns:
    - Calibrated model
    - Scaler
    - Probability scores
    """

    model, scaler = train_calibration_model(X, y)

    probs = calibrate_risk_model(model, scaler, X)

    if return_model:
        return model, scaler, probs

    return probs