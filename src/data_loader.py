from __future__ import annotations

import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd

from src.config import RAW_DATA_PATH, RAW_DATA_DIR, TARGET_COLUMN, TELCO_DATA_URLS


def to_snake_case(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").lower()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [to_snake_case(column) for column in normalized.columns]
    return normalized


def download_raw_data(path: Path = RAW_DATA_PATH, urls: tuple[str, ...] = TELCO_DATA_URLS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            urlretrieve(url, path)
            return path
        except (OSError, URLError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Telco churn data to {path}") from last_error


def load_raw_data(path: Path = RAW_DATA_PATH, download_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Raw data file not found: {path}")
        download_raw_data(path)

    df = pd.read_csv(path)
    return normalize_column_names(df)


def build_data_quality_summary(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, object]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset")

    quality_df = df.replace(r"^\s*$", pd.NA, regex=True)
    missing = pd.DataFrame(
        {
            "missing_count": quality_df.isna().sum(),
            "missing_percent": (quality_df.isna().mean() * 100).round(2),
        }
    ).sort_values(["missing_count", "missing_percent"], ascending=False)

    target_distribution = (
        quality_df[target_column]
        .value_counts(dropna=False)
        .rename_axis(target_column)
        .reset_index(name="count")
    )
    target_distribution["percent"] = (
        target_distribution["count"] / len(df) * 100
    ).round(2)
    target_distribution = target_distribution.set_index(target_column)

    numeric_summary = quality_df.select_dtypes(include="number").describe().T.round(3)
    numeric_columns = set(quality_df.select_dtypes(include="number").columns)
    categorical_columns = [column for column in quality_df.columns if column not in numeric_columns]
    categorical_cardinality = (
        quality_df[categorical_columns]
        .nunique(dropna=False)
        .sort_values(ascending=False)
        .rename("unique_values")
        .to_frame()
    )

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_count": int(df.duplicated().sum()),
        "missing_values": missing,
        "target_distribution": target_distribution,
        "numeric_summary": numeric_summary,
        "categorical_cardinality": categorical_cardinality,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_markdown()


def save_data_quality_report(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Quality Report",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Columns: {summary['column_count']}",
        f"- Duplicate rows: {summary['duplicate_count']}",
        "",
        "## Missing Value Table",
        "",
        dataframe_to_markdown(summary["missing_values"]),
        "",
        "## Target Distribution",
        "",
        dataframe_to_markdown(summary["target_distribution"]),
        "",
        "## Numeric Feature Summary",
        "",
        dataframe_to_markdown(summary["numeric_summary"]),
        "",
        "## Categorical Feature Cardinality",
        "",
        dataframe_to_markdown(summary["categorical_cardinality"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_and_profile_raw_data(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    df = load_raw_data(path)
    return df, build_data_quality_summary(df)
