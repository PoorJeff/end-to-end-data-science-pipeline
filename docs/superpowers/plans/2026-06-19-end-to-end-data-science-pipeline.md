# End-to-End Data Science Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a runnable customer churn data science portfolio project with reusable code, tests, reports, a saved model, and demo applications.

**Architecture:** Use small Python modules under `src/` for loading, features, preprocessing, evaluation, training, explanation, and prediction. Keep apps thin by loading the saved prediction bundle from `models/`.

**Tech Stack:** Python, pandas, numpy, scikit-learn, xgboost when available, matplotlib, seaborn, shap when available, joblib, Streamlit, FastAPI, pytest.

---

## File Structure

- Create: `.gitignore` for Python caches, virtual environments, generated local files, and notebook checkpoints.
- Create: `requirements.txt` with project runtime and testing dependencies.
- Create: `data/README.md`, `data/raw/.gitkeep`, `data/processed/.gitkeep`.
- Create: `src/__init__.py`, `src/config.py`, `src/data_loader.py`, `src/features.py`, `src/preprocessing.py`, `src/evaluate.py`, `src/train.py`, `src/explain.py`, `src/predict.py`.
- Create: `app/__init__.py`, `app/streamlit_app.py`, `app/api.py`.
- Create: `tests/test_data_loader.py`, `tests/test_preprocessing.py`, `tests/test_predict.py`, `tests/test_api.py`.
- Create: `notebooks/01_data_understanding.ipynb`, `notebooks/02_eda.ipynb`, `notebooks/03_model_experiments.ipynb`.
- Create or update: `README.md`, `docs/model_card.md`, `reports/model_comparison.md`, `reports/final_report.md`, `reports/data_quality_report.md`, `reports/figures/.gitkeep`, `models/.gitkeep`.

### Task 1: Test Harness and Expected Behavior

**Files:**
- Create: `tests/test_data_loader.py`
- Create: `tests/test_preprocessing.py`
- Create: `tests/test_predict.py`
- Create: `tests/test_api.py`

- [x] **Step 1: Write failing tests**

Add tests for column normalization, target conversion, preprocessing output with no missing values, prediction output keys, and API response shape.

- [x] **Step 2: Run tests to verify failure**

Run: `python -m pytest -q`

Expected: tests fail because the `src` and `app` modules do not exist yet.

### Task 2: Core Data Pipeline

**Files:**
- Create: `src/config.py`
- Create: `src/data_loader.py`
- Create: `src/features.py`
- Create: `src/preprocessing.py`

- [x] **Step 1: Implement data and preprocessing modules**

Implement raw CSV loading, snake_case column normalization, data-quality summaries, Telco-specific cleaning, target encoding, train/test splitting, and a `ColumnTransformer` with median imputation, standard scaling, most-frequent categorical imputation, and one-hot encoding.

- [x] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_data_loader.py tests/test_preprocessing.py -q`

Expected: data-loader and preprocessing tests pass.

### Task 3: Training, Evaluation, and Model Bundle

**Files:**
- Create: `src/evaluate.py`
- Create: `src/train.py`
- Create: `models/.gitkeep`
- Create: `reports/figures/.gitkeep`

- [x] **Step 1: Implement shared metrics and plots**

Implement metrics for Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve, and precision-recall curve.

- [x] **Step 2: Implement training script**

Train Logistic Regression, Random Forest, SVM, Gradient Boosting, and XGBoost if installed. Save a model bundle containing the best pipeline, feature lists, metrics, and prediction defaults.

- [x] **Step 3: Download dataset and run training**

Run: `python -m src.train`

Expected: `models/best_model.joblib`, `reports/model_comparison.md`, `reports/data_quality_report.md`, and evaluation figures are generated.

### Task 4: Prediction, API, and Streamlit

**Files:**
- Create: `src/predict.py`
- Create: `app/api.py`
- Create: `app/streamlit_app.py`

- [x] **Step 1: Implement prediction interface**

Implement bundle loading, single-record prediction, probability output, predicted class label, and top contributing global features.

- [x] **Step 2: Implement app entry points**

Implement FastAPI `/health` and `/predict`, and a Streamlit dashboard that lets users edit customer attributes and run predictions.

- [x] **Step 3: Run focused tests**

Run: `python -m pytest tests/test_predict.py tests/test_api.py -q`

Expected: prediction and API tests pass.

### Task 5: Reports, Notebooks, README, and Verification

**Files:**
- Create: `src/explain.py`
- Create: notebooks under `notebooks/`
- Modify: `README.md`
- Create or update: `docs/model_card.md`, `reports/final_report.md`

- [x] **Step 1: Implement explainability script**

Generate feature importance and SHAP-compatible summary artifacts, with permutation importance fallback.

- [x] **Step 2: Generate final artifacts**

Run: `python -m src.explain`

Expected: explanation figure and model-card feature list are available.

- [x] **Step 3: Run full verification**

Run: `python -m pytest -q`

Expected: all tests pass.

Run: `python -m src.train`

Expected: training completes and regenerates reports without errors.

Run: `python -m src.explain`

Expected: explanation artifacts regenerate without errors.
