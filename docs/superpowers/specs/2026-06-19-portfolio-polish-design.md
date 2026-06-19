# Portfolio Polish Design

## Goal

Upgrade the churn pipeline from a runnable project to a resume-ready portfolio artifact. The improvement should make the project easier to evaluate, reproduce, and discuss in interviews.

## Current Assessment

The project already has a complete machine learning lifecycle: data loading, EDA, preprocessing, model comparison, a saved model, explainability, tests, Streamlit, FastAPI, and generated reports. The main gaps are portfolio polish: no CI, limited one-command workflow, limited business decision analysis, and documentation that does not yet include data-card or resume-ready storytelling.

## Chosen Improvements

1. Add threshold analysis and lift analysis so the model supports business tradeoff discussion, not only metric reporting.
2. Add risk bands and recommended retention actions to prediction output.
3. Add a GitHub Actions workflow and Makefile for reproducible checks.
4. Add a data card, API examples, and resume bullets.
5. Refresh README and final reports so they highlight measurable outcomes and practical deployment.

## Architecture

Add a small `src/business.py` module for threshold metrics, lift table generation, risk bands, and retention action recommendations. Keep model training in `src/train.py`, but make it call the business module after the best model is selected. Keep app and API layers thin by reading decision thresholds and action text from the saved model bundle.

## Testing

Tests will cover threshold selection, lift table shape, risk-band classification, and prediction output fields. Existing tests must continue to pass.

## Deliverables

- `src/business.py`
- `tests/test_business.py`
- Updated `src/train.py`, `src/predict.py`, `app/api.py`, and `app/streamlit_app.py`
- `.github/workflows/ci.yml`
- `Makefile`
- `docs/data_card.md`, `docs/api_examples.md`, `docs/resume_bullets.md`
- Updated `README.md`, `reports/final_report.md`, `reports/model_comparison.md`, and model bundle
