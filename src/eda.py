from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="churn", hue="churn", palette="Set2", legend=False)
    plt.title("Target Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close()


def save_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_columns = [column for column in ["tenure", "monthly_charges", "total_charges"] if column in df.columns]
    if not numeric_columns:
        return

    plot_df = df.copy()
    if "total_charges" in plot_df.columns:
        plot_df["total_charges"] = pd.to_numeric(plot_df["total_charges"], errors="coerce")

    fig, axes = plt.subplots(1, len(numeric_columns), figsize=(5 * len(numeric_columns), 4))
    if len(numeric_columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, numeric_columns):
        sns.histplot(data=plot_df, x=column, hue="churn", kde=True, ax=axis)
        axis.set_title(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_distributions.png", dpi=160)
    plt.close()


def save_categorical_churn_rates(df: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in ["contract", "internet_service", "payment_method"] if column in df.columns]
    if not columns:
        return

    plot_df = df.copy()
    plot_df["churn_flag"] = plot_df["churn"].astype(str).str.lower().map({"yes": 1, "no": 0})
    fig, axes = plt.subplots(len(columns), 1, figsize=(9, 4 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns):
        rates = (
            plot_df.groupby(column, dropna=False)["churn_flag"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        sns.barplot(data=rates, y=column, x="churn_flag", ax=axis, color="#4c78a8")
        axis.set_xlabel("Churn Rate")
        axis.set_ylabel("")
        axis.set_title(f"Churn Rate by {column.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_dir / "categorical_churn_rates.png", dpi=160)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    if "total_charges" in plot_df.columns:
        plot_df["total_charges"] = pd.to_numeric(plot_df["total_charges"], errors="coerce")
    if "churn" in plot_df.columns:
        plot_df["churn_flag"] = plot_df["churn"].astype(str).str.lower().map({"yes": 1, "no": 0})
    numeric = plot_df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("Numeric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_eda_figures(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_target_distribution(df, output_dir)
    save_numeric_distributions(df, output_dir)
    save_categorical_churn_rates(df, output_dir)
    save_correlation_heatmap(df, output_dir)
