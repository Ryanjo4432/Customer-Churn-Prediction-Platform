import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("app/model/churn_model.pkl")
SCALER_PATH = Path("app/model/scaler.pkl")
METADATA_PATH = Path("app/model/model_metadata.json")


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model not found — run train.py first")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    return model, scaler, metadata


def predict_churn(input_df: pd.DataFrame) -> dict:
    model, scaler, metadata = load_artifacts()

    feature_cols = metadata["feature_columns"]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_spend_per_month"]

    # make sure all expected columns exist, fill missing with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # keep only what the model was trained on, in the right order
    input_df = input_df[feature_cols]

    # scale the numeric columns
    cols_to_scale = [c for c in num_cols if c in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_pred = int(churn_prob >= 0.5)

    # risk level — makes it easier to read in the UI
    if churn_prob >= 0.75:
        risk = "HIGH"
    elif churn_prob >= 0.45:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "will_churn": bool(churn_pred),
        "churn_probability": round(float(churn_prob), 4),
        "risk_level": risk,
        "top_churn_factors": metadata["top_features"][:5],
        "model_used": metadata["model_name"],
    }
