# API Examples

Run the API locally:

```bash
uvicorn app.api:app --reload
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "total_charges": 29.85
  }'
```

Response fields:

- `churn_probability`: model-estimated churn probability
- `decision_threshold`: operating threshold selected from threshold analysis
- `prediction`: `1` for churn, `0` for no churn
- `prediction_label`: readable label
- `risk_band`: low, medium, or high
- `recommended_action`: retention action suggestion
- `top_contributing_features`: global top features from the model explanation layer
