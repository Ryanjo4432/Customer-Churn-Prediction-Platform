import pytest
import pandas as pd
import numpy as np
from app.preprocessing.preprocess import (
    clean_data,
    encode_features,
    add_features,
)


def test_clean_data_fixes_total_charges(sample_raw_df):
    # TotalCharges comes as string in real data, should become float
    df = clean_data(sample_raw_df.copy())
    assert df["TotalCharges"].dtype in [float, np.float64]


def test_clean_data_drops_customer_id(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    assert "customerID" not in df.columns


def test_clean_data_drops_nulls(sample_raw_df):
    # inject a bad row with null TotalCharges
    bad_row = sample_raw_df.iloc[[0]].copy()
    bad_row["TotalCharges"] = " "  # this is how kaggle dataset has bad rows
    df_with_bad = pd.concat([sample_raw_df, bad_row], ignore_index=True)

    df = clean_data(df_with_bad)
    assert df["TotalCharges"].isna().sum() == 0


def test_encode_features_binary_columns(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)

    # gender, Partner, Dependents, Churn should all be 0 or 1
    for col in ["gender", "Partner", "Dependents", "Churn"]:
        assert df[col].isin([0, 1]).all(), f"{col} should be binary after encoding"


def test_encode_features_creates_dummies(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)

    # one hot encoding should create new columns for Contract
    contract_cols = [c for c in df.columns if "Contract" in c]
    assert len(contract_cols) > 0, "contract columns should be one-hot encoded"


def test_add_features_creates_high_risk(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)
    df = add_features(df)

    assert "high_risk" in df.columns
    # tenure=2 + monthly=95.5 should be high risk
    assert df["high_risk"].iloc[0] == 1
    # tenure=60 + monthly=25 should not be high risk
    assert df["high_risk"].iloc[1] == 0


def test_add_features_creates_avg_spend(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)
    df = add_features(df)

    assert "avg_spend_per_month" in df.columns
    assert (df["avg_spend_per_month"] >= 0).all()


def test_add_features_creates_tenure_groups(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)
    df = add_features(df)

    tenure_group_cols = [c for c in df.columns if "tenure_group" in c]
    assert len(tenure_group_cols) > 0, "tenure group columns should exist"


def test_no_nulls_after_full_pipeline(sample_raw_df):
    df = clean_data(sample_raw_df.copy())
    df = encode_features(df)
    df = add_features(df)

    assert df.isnull().sum().sum() == 0, "no nulls should remain after full pipeline"
