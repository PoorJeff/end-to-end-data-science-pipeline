import pandas as pd

from src.data_loader import build_data_quality_summary, normalize_column_names


def test_normalize_column_names_converts_telco_headers_to_snake_case():
    df = pd.DataFrame(
        {
            "CustomerID": ["0001"],
            "Total Charges": ["29.85"],
            "PaymentMethod": ["Electronic check"],
        }
    )

    normalized = normalize_column_names(df)

    assert list(normalized.columns) == ["customer_id", "total_charges", "payment_method"]


def test_build_data_quality_summary_reports_missing_duplicates_and_target_distribution():
    df = pd.DataFrame(
        {
            "customer_id": ["1", "1", "2"],
            "monthly_charges": [10.0, 10.0, None],
            "contract": ["Month-to-month", "Month-to-month", "Two year"],
            "churn": ["Yes", "Yes", "No"],
        }
    )

    summary = build_data_quality_summary(df, target_column="churn")

    assert summary["row_count"] == 3
    assert summary["column_count"] == 4
    assert summary["duplicate_count"] == 1
    assert summary["missing_values"].loc["monthly_charges", "missing_count"] == 1
    assert summary["target_distribution"].loc["Yes", "count"] == 2
