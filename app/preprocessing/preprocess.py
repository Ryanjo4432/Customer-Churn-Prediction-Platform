import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# columns we actually need for training
FEATURE_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

TARGET_COL = "Churn"


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # TotalCharges comes in as string for some reason, fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # drop the ~11 rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # don't need customerID for ML
    df = df.drop(columns=["customerID"], errors="ignore")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # turn Yes/No into 1/0
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # one hot encode the multi-category columns
    cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # tenure groups — short term customers churn way more
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )
    df = pd.get_dummies(df, columns=["tenure_group"], drop_first=True)

    # avg monthly spend — good signal for value vs risk
    df["avg_spend_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    # flag high risk: new customer + expensive plan
    df["high_risk"] = ((df["tenure"] < 12) & (df["MonthlyCharges"] > 65)).astype(int)

    return df


def scale_numeric(X_train, X_test):
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_spend_per_month"]

    # only scale columns that actually exist after encoding
    cols_to_scale = [c for c in num_cols if c in X_train.columns]

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # save scaler so the API can use it later
    os.makedirs("app/model", exist_ok=True)
    joblib.dump(scaler, "app/model/scaler.pkl")

    return X_train, X_test, scaler


def preprocess(raw_path: str, test_size: float = 0.2, random_state: int = 42):
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = encode_features(df)
    df = add_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test, scaler = scale_numeric(X_train, X_test)

    # save processed data so we don't redo this every time
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"train size: {X_train.shape}, test size: {X_test.shape}")
    print(f"churn rate in train: {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test, scaler
