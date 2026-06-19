# Project Design

## Problem Statement

This project predicts whether a telecom customer is likely to churn. It is a binary classification problem where the target variable is `churn`. The task is suitable for data science because churn risk can be estimated from historical customer attributes, service usage, contract choices, billing information, and support-related signals.

## Portfolio Goal

The project is designed for graduate application review. It demonstrates the full data science lifecycle: raw data ingestion, data-quality assessment, EDA, leakage-safe preprocessing, model benchmarking, model interpretation, deployment-ready prediction, and clear documentation.

## Dataset

The project uses the Telco Customer Churn dataset. It contains customer demographics, account information, service subscriptions, billing attributes, and a churn label.

## Pipeline

1. Load raw data from `data/raw/`.
2. Normalize column names and inspect schema.
3. Generate a data-quality report.
4. Clean dataset-specific fields such as `total_charges`.
5. Split data into stratified train and test sets.
6. Fit preprocessing only on training data.
7. Train and compare multiple classifiers.
8. Save the best model bundle.
9. Generate model comparison and explanation artifacts.
10. Serve predictions through Streamlit and FastAPI.

## Modeling Strategy

The project compares interpretable, ensemble, kernel, and boosting models using the same metrics. The selected metrics are Accuracy, Precision, Recall, F1, and ROC-AUC. Accuracy is not used alone because churn data may be imbalanced and business decisions often depend on catching likely churners without creating too many false positives.

## Explainability

The model explanation layer reports the most influential features globally. SHAP is attempted when compatible with the trained model; otherwise, permutation importance provides a stable fallback.

## Limitations

The dataset is historical and relatively small. It does not include customer interaction history, marketing campaign exposure, competitor offers, or time-series behavior. A production churn system would need drift monitoring, retraining policies, and cost-sensitive decision thresholds.
