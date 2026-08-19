"""Minimal FastAPI service for the trained BehaviorGuard detector."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.explainability import BehaviorGuardExplainer
from src.predict import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, load_model, predict_risk

LOGGER = logging.getLogger("behavior.guard.api")
MODEL_VERSION = "xgboost-final-v1"

MODEL_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount", "Hour", "Time_Period", "Amount_Log",
    "Amount_Category", "Transactions_Last_1H", "Transactions_Last_24H",
    "Amount_ZScore",
]


class TransactionRequest(BaseModel):
    """The stable frontend input contract for one processed transaction."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)
    Hour: int = Field(ge=0, le=23)
    Time_Period: Literal["Night", "Morning", "Afternoon", "Evening"]
    Amount_Log: float
    Amount_Category: Literal["Zero", "Low", "Medium", "High", "Very High"]
    Transactions_Last_1H: float = Field(ge=0)
    Transactions_Last_24H: float = Field(ge=0)
    Amount_ZScore: float

    def to_frame(self) -> pd.DataFrame:
        values = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        return pd.DataFrame([values], columns=MODEL_FEATURES)


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    amount: float
    fraud_probability: float
    risk_score: float
    risk_level: Literal["low", "medium", "high"]
    decision: Literal["normal", "fraud"]
    model_version: str
    threshold: float


class FactorResponse(BaseModel):
    feature: str
    contribution: float
    direction: Literal["toward_fraud", "reduces_risk"]
    display_value: str


class ExplainResponse(PredictionResponse):
    top_factors: list[FactorResponse]
    top_positive_contributors: list[FactorResponse]
    top_negative_contributors: list[FactorResponse]
    explanation_note: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    feature_count: int
    threshold: float


class PredictionRecord(PredictionResponse):
    investigation_status: Literal["new", "reviewing", "resolved"] = "new"


class AlertStatusUpdate(BaseModel):
    status: Literal["new", "reviewing", "resolved"]


class PredictionListResponse(BaseModel):
    items: list[PredictionRecord]


class TransactionDetailResponse(PredictionRecord):
    behavioral_summary: dict[str, float | str]


PREDICTIONS: dict[str, TransactionDetailResponse] = {}


def _cors_origins() -> list[str]:
    configured = os.getenv("BEHAVIORGUARD_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all inference dependencies once for the application lifetime."""
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Loading BehaviorGuard model")
    model = load_model(DEFAULT_MODEL_PATH)
    app.state.model = model
    app.state.scaler = model.named_steps.get("preprocessor")
    app.state.explainer = BehaviorGuardExplainer(model)
    app.state.threshold = DEFAULT_THRESHOLD
    app.state.model_version = MODEL_VERSION
    LOGGER.info("BehaviorGuard model loaded")
    yield
    app.state.model = None
    app.state.scaler = None
    app.state.explainer = None


app = FastAPI(
    title="BehaviorGuard API",
    version=MODEL_VERSION,
    description="Fraud risk scoring for the existing BehaviorGuard model.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type"],
)


def _prediction(request: Request, transaction: TransactionRequest) -> dict[str, Any]:
    try:
        result = predict_risk(
            request.app.state.model,
            transaction.to_frame(),
            threshold=request.app.state.threshold,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "prediction_id": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": float(transaction.Amount),
        "fraud_probability": result["fraud_probability"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "decision": result["decision"],
        "model_version": request.app.state.model_version,
        "threshold": float(request.app.state.threshold),
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="XGBoost fraud detector",
        model_version=request.app.state.model_version,
        feature_count=len(MODEL_FEATURES),
        threshold=request.app.state.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, transaction: TransactionRequest) -> PredictionResponse:
    LOGGER.info("Scoring transaction")
    import uuid

    result = _prediction(request, transaction)
    result["prediction_id"] = f"pred_{uuid.uuid4().hex[:12]}"
    record = TransactionDetailResponse(
        **result,
        behavioral_summary={
            "hour": transaction.Hour,
            "time_period": transaction.Time_Period,
            "amount_category": transaction.Amount_Category,
            "transactions_last_1h": transaction.Transactions_Last_1H,
            "transactions_last_24h": transaction.Transactions_Last_24H,
            "amount_z_score": transaction.Amount_ZScore,
        },
    )
    PREDICTIONS[record.prediction_id] = record
    return PredictionResponse(**record.model_dump())


@app.post("/explain", response_model=ExplainResponse)
def explain(
    request: Request,
    transaction: TransactionRequest,
    top_n: int = 5,
) -> ExplainResponse:
    if not 1 <= top_n <= 20:
        raise HTTPException(status_code=422, detail="top_n must be between 1 and 20")

    LOGGER.info("Explaining transaction")
    prediction = _prediction(request, transaction)
    prediction["prediction_id"] = ""
    explanation = request.app.state.explainer.explain_prediction(
        transaction.to_frame(),
        prediction,
        top_n=top_n,
    )
    return ExplainResponse(**explanation)


@app.get("/predictions", response_model=PredictionListResponse)
def recent_predictions(limit: int = 20) -> PredictionListResponse:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    records = sorted(
        PREDICTIONS.values(),
        key=lambda item: item.timestamp,
        reverse=True,
    )[:limit]
    return PredictionListResponse(items=[PredictionRecord(**record.model_dump()) for record in records])


@app.get("/alerts", response_model=PredictionListResponse)
def high_risk_alerts(limit: int = 20) -> PredictionListResponse:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    records = [
        record for record in PREDICTIONS.values()
        if record.decision == "fraud"
    ]
    records.sort(key=lambda item: item.timestamp, reverse=True)
    return PredictionListResponse(items=[PredictionRecord(**record.model_dump()) for record in records[:limit]])


@app.get("/transactions/{prediction_id}", response_model=TransactionDetailResponse)
def transaction_detail(prediction_id: str) -> TransactionDetailResponse:
    record = PREDICTIONS.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record


@app.patch("/alerts/{prediction_id}", response_model=PredictionRecord)
def update_alert_status(prediction_id: str, update: AlertStatusUpdate) -> PredictionRecord:
    record = PREDICTIONS.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    updated = record.model_copy(update={"investigation_status": update.status})
    PREDICTIONS[prediction_id] = TransactionDetailResponse(**updated.model_dump())
    return PredictionRecord(**updated.model_dump())
