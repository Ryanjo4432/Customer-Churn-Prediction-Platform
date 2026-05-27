import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocessing.preprocess import preprocess


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "f1": round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    print(f"\n--- {model_name} ---")
    print(f"F1:        {metrics['f1']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nFull Report:")
    print(classification_report(y_test, preds, target_names=["Stayed", "Churned"]))

    return metrics


def get_top_features(model, feature_names: list, top_n: int = 10) -> list:
    # random forest gives feature importances, logistic gives coefficients
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    indices = np.argsort(importances)[::-1][:top_n]
    top = [(feature_names[i], round(float(importances[i]), 4)) for i in indices]
    return top


def train(raw_data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    print("running preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(raw_data_path)

    feature_names = list(X_train.columns)

    # --- model 1: logistic regression (solid baseline) ---
    print("\ntraining logistic regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")

    # --- model 2: random forest (usually better for this) ---
    print("\ntraining random forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")

    # pick the better model based on f1 (accuracy is misleading for churn)
    if rf_metrics["f1"] >= lr_metrics["f1"]:
        best_model = rf
        best_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = lr
        best_name = "Logistic Regression"
        best_metrics = lr_metrics

    print(f"\nbest model: {best_name} with F1={best_metrics['f1']}")

    # top factors driving churn
    top_features = get_top_features(best_model, feature_names)
    print(f"\ntop churn factors:")
    for feat, score in top_features:
        print(f"  {feat}: {score}")

    # save everything
    os.makedirs("app/model", exist_ok=True)
    joblib.dump(best_model, "app/model/churn_model.pkl")

    # save metadata so the API knows what model is loaded
    metadata = {
        "model_name": best_name,
        "metrics": best_metrics,
        "top_features": top_features,
        "feature_columns": feature_names,
    }
    with open("app/model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nmodel saved to app/model/churn_model.pkl")
    print("metadata saved to app/model/model_metadata.json")

    return best_model, best_metrics, top_features


if __name__ == "__main__":
    train()
