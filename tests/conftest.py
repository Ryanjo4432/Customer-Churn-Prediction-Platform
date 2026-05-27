import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

# reusable test client for all api tests
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# a fake customer that should be high risk (new + expensive plan)
@pytest.fixture
def high_risk_customer():
    return {
        "tenure": 2,
        "MonthlyCharges": 95.5,
        "TotalCharges": 191.0,
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
        "PaymentMethod_Credit_card": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0,
    }


# a fake customer that should be low risk (long tenure + cheap plan)
@pytest.fixture
def low_risk_customer():
    return {
        "tenure": 60,
        "MonthlyCharges": 25.0,
        "TotalCharges": 1500.0,
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 1,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "PaymentMethod_Credit_card": 1,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0,
    }


# raw dataframe that looks like the telco dataset
@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "customerID": ["abc-123", "def-456"],
        "tenure": [2, 60],
        "MonthlyCharges": [95.5, 25.0],
        "TotalCharges": ["191.0", "1500.0"],  # intentionally string like real data
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 0],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        "Churn": ["Yes", "No"],
    })
