import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.predict import predict_churn


def test_predict_churn_returns_probability_label_and_top_features():
    X = pd.DataFrame(
        {
            "tenure": [1, 20, 3, 40],
            "contract": ["Month-to-month", "Two year", "Month-to-month", "One year"],
        }
    )
    y = [1, 0, 1, 0]
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["contract"]),
                        ("numeric", "passthrough", ["tenure"]),
                    ]
                ),
            ),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )
    pipeline.fit(X, y)
    bundle = {
        "model": pipeline,
        "numeric_features": ["tenure"],
        "categorical_features": ["contract"],
        "decision_threshold": 0.45,
        "top_features": [
            {"feature": "contract_Month-to-month", "importance": 0.42},
            {"feature": "tenure", "importance": 0.31},
        ],
    }

    result = predict_churn(bundle, {"tenure": 2, "contract": "Month-to-month"})

    assert set(result) == {
        "churn_probability",
        "decision_threshold",
        "prediction",
        "prediction_label",
        "risk_band",
        "recommended_action",
        "top_contributing_features",
    }
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["decision_threshold"] == 0.45
    assert result["prediction_label"] in {"Churn", "No Churn"}
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert isinstance(result["recommended_action"], str)
    assert result["top_contributing_features"][0]["feature"] == "contract_Month-to-month"
