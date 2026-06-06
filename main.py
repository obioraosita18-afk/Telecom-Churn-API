"""
main.py — Customer Churn Prediction API
Run locally:  uvicorn main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# ── Load artefacts ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model           = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
    scaler          = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
except FileNotFoundError:
    raise RuntimeError("Model artefacts not found. Run `python train.py` first.")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ─────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    Literal["Yes", "No"]
    Partner:          Literal["Yes", "No"]
    Dependents:       Literal["Yes", "No"]
    tenure:           float = Field(..., ge=0)
    PhoneService:     Literal["Yes", "No"]
    MultipleLines:    Literal["Yes", "No", "No phone service"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["Yes", "No", "No internet service"]
    OnlineBackup:     Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport:      Literal["Yes", "No", "No internet service"]
    StreamingTV:      Literal["Yes", "No", "No internet service"]
    StreamingMovies:  Literal["Yes", "No", "No internet service"]
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod:    Literal[
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"
                      ]
    MonthlyCharges:   float = Field(..., ge=0)
    TotalCharges:     float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.2
            }
        }
    }

# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction:      bool
    churn_label:           str
    churn_probability:     float
    retention_probability: float
    risk_level:            str

# ── Helper ─────────────────────────────────────────────────────────────────────
def preprocess(data: CustomerData) -> pd.DataFrame:
    raw = pd.DataFrame([data.model_dump()])
    raw["tenure_year"] = raw["tenure"] / 12
    raw["Churn"] = "No"

    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
    ]
    for col in cat_cols:
        raw[col] = raw[col].astype("category")

    encoded = pd.get_dummies(raw, columns=cat_cols, drop_first=True)

    numerical_cols = ["tenure_year", "tenure", "MonthlyCharges", "TotalCharges"]
    scaled_vals = scaler.transform(encoded[numerical_cols])
    for i, col in enumerate(numerical_cols):
        encoded[f"scaled_{col}"] = scaled_vals[:, i]

    encoded.drop(["tenure", "MonthlyCharges", "TotalCharges", "tenure_year"], axis=1, inplace=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Customer Churn Prediction API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData):
    try:
        features    = preprocess(customer)
        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        risk = "High" if probability >= 0.70 else ("Medium" if probability >= 0.40 else "Low")

        return PredictionResponse(
            churn_prediction      = bool(prediction),
            churn_label           = "Churn" if prediction else "No Churn",
            churn_probability     = round(float(probability), 4),
            retention_probability = round(1 - float(probability), 4),
            risk_level            = risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerData]):
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100.")
    results = []
    for i, customer in enumerate(customers):
        try:
            features    = preprocess(customer)
            prediction  = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            risk = "High" if probability >= 0.70 else ("Medium" if probability >= 0.40 else "Low")
            results.append({
                "index": i,
                "churn_prediction":      bool(prediction),
                "churn_label":           "Churn" if prediction else "No Churn",
                "churn_probability":     round(float(probability), 4),
                "retention_probability": round(1 - float(probability), 4),
                "risk_level":            risk,
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return {"predictions": results, "total": len(results)}
