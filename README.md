# End-to-End Data Science Pipeline: Customer Churn Prediction

[![CI](https://github.com/PoorJeff/end-to-end-data-science-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/PoorJeff/end-to-end-data-science-pipeline/actions/workflows/ci.yml)

## Project Overview

This project is a resume-ready, end-to-end data science pipeline for telecom customer churn prediction. It starts from raw data, performs data-quality assessment and EDA, builds leakage-safe preprocessing pipelines, compares multiple models, explains the selected model, adds threshold and lift analysis for business decisions, and serves predictions through Streamlit and FastAPI.

## Why This Project Matters

Customer churn is a practical business problem: retaining existing customers is often cheaper than acquiring new ones. This project shows both sides of applied data science: model development and decision support. The final system does not stop at ROC-AUC; it converts churn probabilities into risk bands, operating thresholds, and retention actions.

## Portfolio Highlights

- End-to-end workflow: raw data, quality checks, EDA, feature engineering, modeling, explainability, API, dashboard, reports, tests, and CI.
- Best model: **XGBoost (tuned)** with **0.8467 ROC-AUC** on the held-out test set.
- Operating threshold: **0.30**, selected to balance recall and precision for retention outreach.
- Business impact framing: captures **78.1%** of churners while contacting **38.8%** of customers.
- Lift analysis: top-risk decile has **75.9% churn rate** and **2.86x lift** over the base churn rate.

## Dataset

- Dataset: Telco Customer Churn
- Rows: 7,043
- Target: `churn`
- Task type: binary classification
- Raw file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Data-quality note: `total_charges` contains 11 blank values, handled through train-only preprocessing imputation.
- More detail: `docs/data_card.md`

## Pipeline

1. Load raw data from `data/raw/`.
2. Normalize column names to snake_case.
3. Generate data-quality summary tables.
4. Create EDA figures for target distribution, numeric distributions, categorical churn rates, and correlations.
5. Prepare modeling features and encode the churn target.
6. Split data with stratification.
7. Fit preprocessing with `ColumnTransformer` only on training data.
8. Train and compare Logistic Regression, Random Forest, Gradient Boosting, SVM, and XGBoost.
9. Tune the best model with compact grid search.
10. Generate model metrics, threshold analysis, lift analysis, SHAP summary, and feature importance.
11. Save the model bundle and serve predictions with Streamlit and FastAPI.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.7835 | 0.5848 | 0.6364 | 0.6095 | 0.8331 |
| Gradient Boosting | 0.8062 | 0.6735 | 0.5241 | 0.5895 | 0.8434 |
| SVM | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 |
| XGBoost | 0.8055 | 0.6678 | 0.5321 | 0.5923 | 0.8452 |
| XGBoost (tuned) | 0.8055 | 0.6678 | 0.5321 | 0.5923 | 0.8467 |

## Threshold and Lift Analysis

The default 0.50 threshold is not always the best operating point for churn retention. This project selects a **0.30 threshold** by maximizing F1 among thresholds with recall at least 0.65.

| Threshold | Precision | Recall | F1 | Contact Rate | Churn Capture Rate |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.5338 | 0.7807 | 0.6341 | 0.3882 | 0.7807 |

Top lift segments:

| Decile | Customers | Churn Rate | Lift | Cumulative Churn Capture |
|---:|---:|---:|---:|---:|
| 1 | 141 | 0.7589 | 2.8589 | 0.2861 |
| 2 | 141 | 0.6170 | 2.3246 | 0.5187 |
| 3 | 141 | 0.3901 | 1.4695 | 0.6658 |

Detailed report: `reports/threshold_analysis.md`

## Screenshots and Artifacts

![Confusion matrix](reports/figures/confusion_matrix.png)

![ROC curve](reports/figures/roc_curve.png)

![SHAP summary](reports/figures/shap_summary.png)

Additional outputs:

- `reports/data_quality_report.md`
- `reports/model_comparison.md`
- `reports/threshold_analysis.md`
- `reports/final_report.md`
- `docs/model_card.md`
- `docs/data_card.md`
- `docs/resume_bullets.md`
- `models/best_model.joblib`

## How To Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Train models and regenerate reports:

```bash
python -m src.train
```

Generate explainability artifacts:

```bash
python -m src.explain
```

Run the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

Run the FastAPI service:

```bash
uvicorn app.api:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation. API examples are in `docs/api_examples.md`.

If `make` is available:

```bash
make all
```

## Repository Structure

```text
end-to-end-data-science-pipeline/
├── .github/workflows/ci.yml
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
├── docs/
│   ├── api_examples.md
│   ├── data_card.md
│   ├── model_card.md
│   ├── project_design.md
│   └── resume_bullets.md
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── business.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
└── tests/
```

## Resume Summary

Built an end-to-end customer churn prediction pipeline covering data quality assessment, EDA, leakage-safe preprocessing, model benchmarking, explainability, threshold analysis, and Streamlit/FastAPI deployment. Compared five classifiers and selected tuned XGBoost with 0.8467 ROC-AUC; added lift analysis to identify a top-risk decile with 2.86x lift for retention targeting.

## Limitations

- The dataset is static and may not represent current telecom behavior.
- The operating threshold should be recalibrated for real retention campaign budgets.
- The model does not include customer interaction history, campaign exposure, or competitor data.
- The app is a local demo, not a production deployment.

## Future Work

- Add MLflow experiment tracking.
- Add Docker for one-command environment setup.
- Add data drift checks and scheduled retraining.
- Calibrate probabilities and tune thresholds against real contact cost and retention value.
