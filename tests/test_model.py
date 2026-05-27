import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_model_file_exists_after_training():
    # if this fails you havent run train.py yet
    model_path = Path("app/model/churn_model.pkl")
    if not model_path.exists():
        pytest.skip("run train.py first then rerun tests")
    assert model_path.exists()


def test_scaler_file_exists_after_training():
    scaler_path = Path("app/model/scaler.pkl")
    if not scaler_path.exists():
        pytest.skip("run train.py first then rerun tests")
    assert scaler_path.exists()


def test_metadata_file_has_expected_keys():
    import json
    meta_path = Path("app/model/model_metadata.json")
    if not meta_path.exists():
        pytest.skip("model metadata not found, run train.py first")

    with open(meta_path) as f:
        meta = json.load(f)

    assert "model_name" in meta
    assert "metrics" in meta
    assert "top_features" in meta
    assert "feature_columns" in meta


def test_metadata_metrics_are_reasonable():
    import json
    meta_path = Path("app/model/model_metadata.json")
    if not meta_path.exists():
        pytest.skip("run train.py first")

    with open(meta_path) as f:
        meta = json.load(f)

    # a half decent churn model should get at least 0.5 f1
    assert meta["metrics"]["f1"] > 0.5, "f1 score too low, something is wrong"
    assert meta["metrics"]["roc_auc"] > 0.7, "roc auc too low, check the pipeline"


def test_predict_churn_output_shape():
    from pathlib import Path
    if not Path("app/model/churn_model.pkl").exists():
        pytest.skip("model not found")

    from app.model.predict import predict_churn

    # minimal input df
    input_data = {
        "tenure": 5,
        "MonthlyCharges": 80.0,
        "TotalCharges": 400.0,
        "avg_spend_per_month": 80.0,
        "high_risk": 1,
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Credit_card": 0,
        "PaymentMethod_Mailed_check": 0,
        "tenure_group_1-2yr": 0,
        "tenure_group_2-4yr": 0,
        "tenure_group_4+yr": 0,
    }

    df = pd.DataFrame([input_data])
    result = predict_churn(df)

    # check all expected keys come back
    assert "will_churn" in result
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "top_churn_factors" in result
    assert "model_used" in result

    # sanity checks
    assert isinstance(result["will_churn"], bool)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_churn_risk_levels():
    from pathlib import Path
    if not Path("app/model/churn_model.pkl").exists():
        pytest.skip("model not found")

    from app.model.predict import predict_churn

    # mock predict_proba to control the output
    with patch("app.model.predict.joblib.load") as mock_load:
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8]])
        mock_scaler.transform.return_value = np.zeros((1, 4))
        mock_load.side_effect = [mock_model, mock_scaler]

        with patch("app.model.predict.open"), patch("app.model.predict.json.load") as mock_json:
            mock_json.return_value = {
                "feature_columns": ["tenure", "MonthlyCharges", "TotalCharges", "avg_spend_per_month"],
                "top_features": [["Contract_Two_year", 0.15]],
                "model_name": "Random Forest",
            }
            df = pd.DataFrame([{
                "tenure": 2, "MonthlyCharges": 95.0,
                "TotalCharges": 190.0, "avg_spend_per_month": 95.0,
            }])
            result = predict_churn(df)
            # prob 0.8 should be HIGH
            assert result["risk_level"] == "HIGH"
