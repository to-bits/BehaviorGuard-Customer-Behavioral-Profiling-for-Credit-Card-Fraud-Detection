# BehaviorGuard API

Run from the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cd "BehaviorGuard — Customer Behavioral Profiling for Credit Card Fraud Detection"
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /explain?top_n=5`
- `GET /predictions?limit=20`
- `GET /alerts?limit=20`
- `GET /transactions/{prediction_id}`
- `PATCH /alerts/{prediction_id}` with `{"status":"new|reviewing|resolved"}`

The request body for `/predict` and `/explain` must contain the 37 processed model features. The API loads the existing XGBoost pipeline once during startup and applies the validated `0.54` threshold. Local frontend CORS defaults to `http://localhost:3000` and `http://localhost:5173`; override with `BEHAVIORGUARD_CORS_ORIGINS`.

Prediction and alert records are intentionally lightweight in-memory session entities. They store the prediction summary and behavioral summary only; raw model features and filesystem paths are not persisted or returned. Restarting the API clears this demo session store.

Run API tests from the project directory:

```bash
pytest tests/test_api.py
```
