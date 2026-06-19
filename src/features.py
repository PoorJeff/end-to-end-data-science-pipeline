from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ID_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class FeatureMetadata:
    numeric_features: list[str]
    categorical_features: list[str]
    target_column: str
    positive_label: str
    id_columns: tuple[str, ...]


def encode_churn_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    encoded = series.astype(str).str.strip().str.lower().map(mapping)
    if encoded.isna().any():
        invalid = sorted(series[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected target labels for churn: {invalid}")
    return encoded.astype(int)


def prepare_modeling_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: tuple[str, ...] = ID_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, FeatureMetadata]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    data = df.copy()
    if "total_charges" in data.columns:
        data["total_charges"] = pd.to_numeric(data["total_charges"], errors="coerce")
    if "senior_citizen" in data.columns:
        data["senior_citizen"] = pd.to_numeric(data["senior_citizen"], errors="coerce")

    y = encode_churn_target(data[target_column])
    drop_columns = [target_column, *[column for column in id_columns if column in data.columns]]
    X = data.drop(columns=drop_columns)

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    metadata = FeatureMetadata(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
        positive_label="Churn",
        id_columns=id_columns,
    )
    return X, y, metadata
