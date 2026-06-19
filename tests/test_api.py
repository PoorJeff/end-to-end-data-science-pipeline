from fastapi.testclient import TestClient

from app import api


def test_predict_endpoint_returns_stable_response_structure(monkeypatch):
    def fake_bundle():
        return {
            "model": None,
            "numeric_features": ["tenure", "monthly_charges", "total_charges", "senior_citizen"],
            "categorical_features": [
                "gender",
                "partner",
                "dependents",
                "phone_service",
                "multiple_lines",
                "internet_service",
                "online_security",
                "online_backup",
                "device_protection",
                "tech_support",
                "streaming_tv",
                "streaming_movies",
                "contract",
                "paperless_billing",
                "payment_method",
            ],
            "top_features": [],
        }

    def fake_predict(bundle, payload):
        return {
            "churn_probability": 0.73,
            "decision_threshold": 0.45,
            "prediction": 1,
            "prediction_label": "Churn",
            "risk_band": "High",
            "recommended_action": "High-priority retention outreach.",
            "top_contributing_features": [],
        }

    monkeypatch.setattr(api, "get_model_bundle", fake_bundle)
    monkeypatch.setattr(api, "predict_churn", fake_predict)
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure": 1,
            "phone_service": "No",
            "multiple_lines": "No phone service",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 29.85,
            "total_charges": 29.85,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.73,
        "decision_threshold": 0.45,
        "prediction": 1,
        "prediction_label": "Churn",
        "risk_band": "High",
        "recommended_action": "High-priority retention outreach.",
        "top_contributing_features": [],
    }
