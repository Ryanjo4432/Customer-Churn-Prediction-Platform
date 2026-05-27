import json
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

from app.model.predict import predict_churn

router = APIRouter()

METADATA_PATH = Path("app/model/model_metadata.json")


# what the user sends in the request body
class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="months with the company")
    MonthlyCharges: float = Field(..., gt=0, description="monthly bill amount")
    TotalCharges: float = Field(..., ge=0, description="total amount paid so far")
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen")
    Partner: int = Field(..., ge=0, le=1, description="1 if has partner")
    Dependents: int = Field(..., ge=0, le=1, description="1 if has dependents")
    PhoneService: int = Field(..., ge=0, le=1, description="1 if has phone service")
    PaperlessBilling: int = Field(..., ge=0, le=1, description="1 if paperless billing")
    Contract_One_year: int = Field(0, ge=0, le=1, description="1 if one-year contract")
    Contract_Two_year: int = Field(0, ge=0, le=1, description="1 if two-year contract")
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1)
    InternetService_No: int = Field(0, ge=0, le=1)
    PaymentMethod_Credit_card: int = Field(0, ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1)


@router.get("/health")
def health_check():
    # quick ping to check the api is alive
    model_ready = Path("app/model/churn_model.pkl").exists()
    return {
        "status": "ok",
        "model_loaded": model_ready,
    }


@router.get("/model-info")
def model_info():
    # returns what model is running + how it performed
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="model not trained yet, run train.py first")

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    return {
        "model_name": metadata["model_name"],
        "metrics": metadata["metrics"],
        "top_churn_factors": metadata["top_features"][:10],
    }


@router.post("/predict")
def predict(customer: CustomerData):
    try:
        # dump pydantic model to dict then into a dataframe
        data = customer.model_dump()

        # compute derived features the model expects
        data["avg_spend_per_month"] = data["TotalCharges"] / (data["tenure"] + 1)
        data["high_risk"] = int(data["tenure"] < 12 and data["MonthlyCharges"] > 65)

        # tenure group one-hot (model was trained with these)
        data["tenure_group_1-2yr"] = int(12 <= data["tenure"] < 24)
        data["tenure_group_2-4yr"] = int(24 <= data["tenure"] < 48)
        data["tenure_group_4+yr"] = int(data["tenure"] >= 48)

        input_df = pd.DataFrame([data])
        result = predict_churn(input_df)
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")
