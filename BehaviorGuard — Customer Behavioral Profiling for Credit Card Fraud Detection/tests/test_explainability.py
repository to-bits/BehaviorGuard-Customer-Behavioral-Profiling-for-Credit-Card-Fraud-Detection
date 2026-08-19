from pathlib import Path

import pandas as pd

from src.explainability import BehaviorGuardExplainer
from src.predict import load_model, predict_risk


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "transactions_behavioral.csv"
MODEL_PATH = (
    ROOT / "results" / "final_analysis" / "models" / "xgboost.joblib"
)


def test_prediction_and_explanation_are_separate():
    model = load_model(MODEL_PATH)
    row = pd.read_csv(DATA_PATH, nrows=1).drop(columns=["Class", "Transaction_Time"])

    prediction = predict_risk(model, row)
    explanation = BehaviorGuardExplainer(model).explain_prediction(
        row,
        prediction,
        top_n=3,
    )

    assert 0.0 <= prediction["fraud_probability"] <= 1.0
    assert prediction["risk_score"] == round(
        prediction["fraud_probability"] * 100.0,
        2,
    )
    assert len(explanation["top_factors"]) == 3
    assert explanation["top_positive_contributors"] or explanation[
        "top_negative_contributors"
    ]


def test_global_importance_uses_model_features():
    model = load_model(MODEL_PATH)
    rows = pd.read_csv(DATA_PATH, nrows=2).drop(
        columns=["Class", "Transaction_Time"]
    )

    importance = BehaviorGuardExplainer(model).global_feature_importance(rows)

    assert importance
    assert {item["feature"] for item in importance} == set(model.feature_names_in_)
    assert all(item["mean_abs_shap"] >= 0 for item in importance)
