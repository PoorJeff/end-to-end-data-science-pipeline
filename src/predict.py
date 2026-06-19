from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.business import classify_risk, recommend_retention_action
from src.evaluate import prediction_scores


@lru_cache(maxsize=1)
def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def required_features(bundle: dict[str, Any]) -> list[str]:
    return [*bundle.get("numeric_features", []), *bundle.get("categorical_features", [])]


def default_customer(bundle: dict[str, Any]) -> dict[str, Any]:
    defaults = bundle.get("default_input", {})
    if defaults:
        return dict(defaults)
    return {feature: 0 for feature in required_features(bundle)}


def payload_to_frame(bundle: dict[str, Any], payload: dict[str, Any]) -> pd.DataFrame:
    features = required_features(bundle)
    missing = [feature for feature in features if feature not in payload]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    row = {feature: payload[feature] for feature in features}
    frame = pd.DataFrame([row])
    for feature in bundle.get("numeric_features", []):
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def predict_churn(bundle: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    model = bundle["model"]
    frame = payload_to_frame(bundle, payload)
    probability = float(prediction_scores(model, frame)[0])
    threshold = float(bundle.get("decision_threshold", 0.5))
    prediction = int(probability >= threshold)
    return {
        "churn_probability": round(probability, 4),
        "decision_threshold": round(threshold, 4),
        "prediction": prediction,
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "risk_band": classify_risk(probability, threshold),
        "recommended_action": recommend_retention_action(probability, threshold),
        "top_contributing_features": bundle.get("top_features", [])[:5],
    }
