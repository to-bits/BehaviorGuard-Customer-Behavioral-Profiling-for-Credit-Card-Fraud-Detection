"""Prediction utilities for the serialized BehaviorGuard model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_THRESHOLD = 0.54
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "final_analysis"
    / "models"
    / "xgboost.joblib"
)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load the existing fitted pipeline without retraining it."""
    return joblib.load(model_path)


def predict_risk(
    model: Any,
    features: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Return a model probability and thresholded fraud decision."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if len(features) != 1:
        raise ValueError("predict_risk expects exactly one transaction row")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    fraud_probability = float(model.predict_proba(features)[0, 1])
    decision = "fraud" if fraud_probability >= threshold else "normal"

    if fraud_probability >= threshold:
        risk_level = "high"
    elif fraud_probability >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "fraud_probability": fraud_probability,
        "risk_score": round(fraud_probability * 100.0, 2),
        "risk_level": risk_level,
        "decision": decision,
        "threshold": float(threshold),
    }
