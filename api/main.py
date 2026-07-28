"""
Fraud scoring API. Loads whatever model currently holds the 'champion'
alias in the MLflow registry - promoting a new model never requires
touching this file, only reassigning the alias.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import json

app = FastAPI(title="Fraud Detection API")

MODEL_URI = "models:/fraud_detection_classifier@champion"
model = mlflow.pyfunc.load_model(MODEL_URI)

with open("models/fraud_detection/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)


class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    day_of_week: int
    transactions_last_60_sec: int
    amount_last_7_days: float
    merchant_risk_score: float
    country: str
    channel: str
    merchant_category: str
    customer_risk_level: str
    country_changed: bool | None = None


@app.post("/score")
def score_transaction(tx: Transaction):
    raw = pd.DataFrame([tx.dict()])

    country_dummies = pd.get_dummies(raw["country"], prefix="country")
    channel_dummies = pd.get_dummies(raw["channel"], prefix="channel")
    merchant_cat_dummies = pd.get_dummies(raw["merchant_category"], prefix="merchant_cat")
    risk_dummies = pd.get_dummies(raw["customer_risk_level"], prefix="customer_risk")
    changed_dummies = pd.get_dummies(raw["country_changed"], prefix="country_changed", dummy_na=True)

    full = pd.concat(
        [raw, country_dummies, channel_dummies, merchant_cat_dummies, risk_dummies, changed_dummies],
        axis=1,
    )

    # reindex to the EXACT columns the model was trained on - any column
    # not present in this request (e.g. a country not seen here) becomes 0
    aligned = full.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    prediction = model.predict(aligned)
    return {"fraud_prediction": int(prediction[0])}


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}