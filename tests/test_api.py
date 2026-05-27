import pytest


def test_root_returns_200(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "message" in res.json()


def test_health_check(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["status"] == "ok"
    # model_loaded can be false if no pkl exists yet, thats fine
    assert "model_loaded" in data


def test_predict_valid_high_risk(client, high_risk_customer):
    # only runs if model is trained, skip if not
    from pathlib import Path
    if not Path("app/model/churn_model.pkl").exists():
        pytest.skip("model not trained yet")

    res = client.post("/api/v1/predict", json=high_risk_customer)
    assert res.status_code == 200

    data = res.json()
    assert "will_churn" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_valid_low_risk(client, low_risk_customer):
    from pathlib import Path
    if not Path("app/model/churn_model.pkl").exists():
        pytest.skip("model not trained yet")

    res = client.post("/api/v1/predict", json=low_risk_customer)
    assert res.status_code == 200

    data = res.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_missing_field(client):
    # sending incomplete data should fail validation
    res = client.post("/api/v1/predict", json={"tenure": 5})
    assert res.status_code == 422  # pydantic validation error


def test_predict_negative_tenure(client):
    # tenure cant be negative
    bad_data = {
        "tenure": -1,
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0,
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "PaymentMethod_Credit_card": 0,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0,
    }
    res = client.post("/api/v1/predict", json=bad_data)
    assert res.status_code == 422


def test_model_info_endpoint(client):
    from pathlib import Path
    if not Path("app/model/model_metadata.json").exists():
        pytest.skip("model metadata not found, train first")

    res = client.get("/api/v1/model-info")
    assert res.status_code == 200

    data = res.json()
    assert "model_name" in data
    assert "metrics" in data
    assert "top_churn_factors" in data
