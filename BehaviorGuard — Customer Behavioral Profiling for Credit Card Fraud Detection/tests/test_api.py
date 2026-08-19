from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import MODEL_FEATURES, app


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "transactions_behavioral.csv"


def _payload() -> dict:
    row = pd.read_csv(DATA_PATH, nrows=1).drop(
        columns=["Class", "Transaction_Time"]
    )
    return row[MODEL_FEATURES].iloc[0].to_dict()


def test_health_and_model_info():
    with TestClient(app) as client:
        health = client.get("/health")
        info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True}
    assert info.status_code == 200
    assert info.json()["feature_count"] == 37
    assert "model_version" in info.json()
    assert "path" not in info.json()


def test_predict_and_explain_contracts():
    with TestClient(app) as client:
        prediction = client.post("/predict", json=_payload())
        explanation = client.post("/explain?top_n=3", json=_payload())

    assert prediction.status_code == 200
    prediction_body = prediction.json()
    assert set(prediction_body) == {
        "fraud_probability",
        "risk_score",
        "risk_level",
        "decision",
        "model_version",
    }
    assert 0 <= prediction_body["fraud_probability"] <= 1

    assert explanation.status_code == 200
    explanation_body = explanation.json()
    assert len(explanation_body["top_factors"]) == 3
    assert explanation_body["top_positive_contributors"] or explanation_body[
        "top_negative_contributors"
    ]


def test_invalid_request_returns_clear_validation_error():
    payload = _payload()
    payload.pop("Amount")

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Amount" in response.text
