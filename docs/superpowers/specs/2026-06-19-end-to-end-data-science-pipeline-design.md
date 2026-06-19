# End-to-End Data Science Pipeline Design

## Context

This repository is intended to demonstrate a complete data science lifecycle for graduate program applications. The project will use the Telco Customer Churn dataset because it is compact, business-readable, and includes numerical fields, categorical fields, data-quality issues, binary classification, and explainability opportunities.

## Chosen Approach

The project will be a runnable MVP rather than a notebook-only portfolio or a heavy MLOps project. It will include reusable source modules, tests, reports, a saved model, a Streamlit dashboard, and a FastAPI endpoint. Optional production extensions such as Docker, DVC, MLflow, and CI will be documented as future work instead of being part of the first implementation.

## Architecture

The code is split into small modules with clear responsibilities:

- `src/data_loader.py` loads raw data, normalizes columns, and creates data-quality summaries.
- `src/features.py` performs dataset-specific type cleaning and target conversion.
- `src/preprocessing.py` builds leakage-safe scikit-learn preprocessing pipelines.
- `src/evaluate.py` calculates shared metrics and creates evaluation plots.
- `src/train.py` trains and compares models, tunes a compact parameter grid, and saves the best bundle.
- `src/explain.py` creates model interpretation artifacts.
- `src/predict.py` loads the saved bundle and returns prediction results for apps.
- `app/streamlit_app.py` provides the interactive dashboard.
- `app/api.py` exposes `/health` and `/predict` endpoints.

## Data Flow

Raw Telco CSV data lives in `data/raw/`. The loader keeps raw data unchanged and returns a normalized DataFrame. Feature preparation converts `total_charges` to numeric, removes customer identifiers from modeling features, maps churn labels to integers, and leaves preprocessing to scikit-learn transformers. Train, validation, and test splits are stratified. Preprocessing is fit only on the training split through scikit-learn `Pipeline` and `ColumnTransformer`.

## Modeling

The project will compare Logistic Regression, Random Forest, SVM, Gradient Boosting, and XGBoost when XGBoost is installed. Each model uses the same train/test split and shared metrics: Accuracy, Precision, Recall, F1, and ROC-AUC. A compact `GridSearchCV` tunes the strongest tree-based candidate to avoid making the portfolio project slow or brittle.

## Explainability

The primary explanation artifact is a feature-importance report. SHAP is used when compatible with the selected model and installed dependencies. If SHAP fails because of environment or model constraints, permutation importance is used as a reliable fallback while preserving the required output artifacts.

## Error Handling

Data loading validates that files exist and that the target column is present. Prediction validates that a trained model bundle exists and returns clear errors for missing required fields. The API uses structured Pydantic input and returns stable JSON keys so it can be tested without running the full Streamlit app.

## Testing

Tests cover data loading and column normalization, preprocessing output shape and missing-value handling, prediction output structure, and API response structure. Unit tests use small in-memory sample data so they remain fast and do not depend on the external dataset.

## Deliverables

The completed project will include `README.md`, `requirements.txt`, reusable source code, tests, notebooks, generated reports, figures, a model card, a final report, and application entry points. The README will explain project value, how to run it, model comparison results, limitations, and future work.
