import numpy as np
import pandas as pd

from src.features import prepare_modeling_data
from src.preprocessing import build_preprocessor, make_train_test_split


def sample_telco_data():
    return pd.DataFrame(
        {
            "customer_id": ["1", "2", "3", "4", "5", "6"],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "senior_citizen": [0, 1, 0, 1, 0, 1],
            "partner": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "dependents": ["No", "No", "Yes", "No", "No", "Yes"],
            "tenure": [1, 34, 2, 45, np.nan, 12],
            "phone_service": ["No", "Yes", "Yes", "Yes", "No", "Yes"],
            "multiple_lines": ["No phone service", "No", "No", "Yes", "No phone service", "No"],
            "internet_service": ["DSL", "Fiber optic", "DSL", "Fiber optic", "No", "DSL"],
            "online_security": ["No", "Yes", "No", "Yes", "No internet service", "No"],
            "online_backup": ["Yes", "No", "No", "Yes", "No internet service", "Yes"],
            "device_protection": ["No", "Yes", "No", "Yes", "No internet service", "No"],
            "tech_support": ["No", "No", "Yes", "Yes", "No internet service", "No"],
            "streaming_tv": ["No", "Yes", "No", "Yes", "No internet service", "No"],
            "streaming_movies": ["No", "Yes", "No", "Yes", "No internet service", "No"],
            "contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month", "One year"],
            "paperless_billing": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "payment_method": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer (automatic)", "Mailed check", "Credit card (automatic)"],
            "monthly_charges": [29.85, 56.95, 53.85, 42.3, 20.0, 65.5],
            "total_charges": ["29.85", "1889.5", "108.15", "1840.75", " ", "786.0"],
            "churn": ["No", "No", "Yes", "No", "Yes", "No"],
        }
    )


def test_prepare_modeling_data_converts_total_charges_and_target():
    X, y, metadata = prepare_modeling_data(sample_telco_data())

    assert "customer_id" not in X.columns
    assert y.tolist() == [0, 0, 1, 0, 1, 0]
    assert pd.api.types.is_numeric_dtype(X["total_charges"])
    assert "total_charges" in metadata.numeric_features


def test_preprocessor_outputs_no_missing_values_after_fit_transform():
    X, y, metadata = prepare_modeling_data(sample_telco_data())
    X_train, _, y_train, _ = make_train_test_split(X, y, test_size=0.34, random_state=7)
    preprocessor = build_preprocessor(metadata.numeric_features, metadata.categorical_features)

    transformed = preprocessor.fit_transform(X_train, y_train)

    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert dense.shape[0] == X_train.shape[0]
    assert not np.isnan(dense).any()
