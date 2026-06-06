"""
train.py — Train the churn model and save artefacts to the same directory.
Run once before starting the API:  python train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df["TotalCharges"] != " "].copy()
df.drop("customerID", axis=1, inplace=True)

df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
df[["TotalCharges", "MonthlyCharges"]] = df[["TotalCharges", "MonthlyCharges"]].astype(float)

# ── Feature engineering ───────────────────────────────────────────────────────
df["tenure_year"] = df["tenure"] / 12

cat_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]
for col in cat_cols:
    df[col] = df[col].astype("category")

model_df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Scale ─────────────────────────────────────────────────────────────────────
numerical_cols = ["tenure_year", "tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
model_df[[f"scaled_{c}" for c in numerical_cols]] = scaler.fit_transform(model_df[numerical_cols])

# ── Split ─────────────────────────────────────────────────────────────────────
drop_cols = ["tenure", "MonthlyCharges", "TotalCharges", "tenure_year", "Churn_Yes"]
X = model_df.drop(drop_cols, axis=1)
y = model_df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── Save artefacts ────────────────────────────────────────────────────────────
joblib.dump(model,            os.path.join(BASE_DIR, "churn_model.pkl"))
joblib.dump(scaler,           os.path.join(BASE_DIR, "scaler.pkl"))
joblib.dump(list(X.columns),  os.path.join(BASE_DIR, "feature_columns.pkl"))

print("\nSaved → churn_model.pkl, scaler.pkl, feature_columns.pkl")
