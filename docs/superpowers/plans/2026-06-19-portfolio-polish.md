# Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the churn project more resume-ready through business decision analysis, reproducibility automation, and stronger documentation.

**Architecture:** Add a focused business-analysis module and reuse it from training and prediction. Keep CI and local commands lightweight by running tests rather than full training on every push.

**Tech Stack:** Python, pandas, scikit-learn, pytest, GitHub Actions, Makefile.

---

### Task 1: Business Decision Layer

**Files:**
- Create: `src/business.py`
- Create: `tests/test_business.py`
- Modify: `src/train.py`
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [x] **Step 1: Write tests for threshold metrics, lift table, risk bands, and prediction action fields.**
- [x] **Step 2: Run tests and verify the new tests fail before implementation.**
- [x] **Step 3: Implement `src/business.py` with threshold analysis, lift analysis, risk bands, and retention recommendations.**
- [x] **Step 4: Wire selected threshold and top lift deciles into the model bundle and prediction output.**
- [x] **Step 5: Run focused tests and verify they pass.**

### Task 2: Portfolio Documentation

**Files:**
- Create: `docs/data_card.md`
- Create: `docs/api_examples.md`
- Create: `docs/resume_bullets.md`
- Modify: `README.md`
- Modify: `reports/final_report.md`

- [x] **Step 1: Add data-card and API examples.**
- [x] **Step 2: Add resume-ready English and Chinese bullets.**
- [x] **Step 3: Update README with CI badge, one-command workflow, threshold analysis, and resume highlights.**

### Task 3: Reproducibility Automation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `Makefile`

- [x] **Step 1: Add CI that installs dependencies and runs tests on pushes to `main`.**
- [x] **Step 2: Add local make targets for install, test, train, explain, app, api, and all.**

### Task 4: Regenerate, Verify, Commit, Push

**Files:**
- Update generated model/report artifacts.

- [x] **Step 1: Run `python -m pytest -q`.**
- [x] **Step 2: Run `python -m src.train`.**
- [x] **Step 3: Run `python -m src.explain`.**
- [x] **Step 4: Run final `python -m pytest -q`.**
- [ ] **Step 5: Commit and push `HEAD:main` only.**
