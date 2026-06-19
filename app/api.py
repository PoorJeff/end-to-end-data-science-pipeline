from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model_bundle, predict_churn


class CustomerFeatures(BaseModel):
    gender: str
    senior_citizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: float = Field(ge=0)
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predict customer churn probability from Telco customer attributes.",
)


@lru_cache(maxsize=1)
def get_model_bundle():
    return load_model_bundle()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerFeatures) -> dict:
    try:
        return predict_churn(get_model_bundle(), payload.model_dump())
    except AttributeError:
        return predict_churn(get_model_bundle(), payload.dict())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
