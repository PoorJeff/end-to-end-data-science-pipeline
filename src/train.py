from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (
    DOCS_DIR,
    FIGURES_DIR,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data_loader import build_data_quality_summary, load_raw_data, save_data_quality_report
from src.eda import save_eda_figures
from src.evaluate import (
    evaluate_classifier,
    metrics_to_markdown,
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
)
from src.features import prepare_modeling_data
from src.preprocessing import build_preprocessor, get_feature_names, make_train_test_split


def candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass
    return models


def tuning_grid(model_name: str) -> dict[str, list[Any]]:
    grids = {
        "Logistic Regression": {"model__C": [0.5, 1.0, 2.0]},
        "Random Forest": {
            "model__n_estimators": [160, 220],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 160],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "SVM": {"model__C": [0.7, 1.0], "model__gamma": ["scale"]},
        "XGBoost": {
            "model__n_estimators": [120, 180],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05, 0.08],
        },
    }
    return grids.get(model_name, {})


def build_pipeline(model, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def default_input_from_training_data(X_train: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for feature in numeric_features:
        defaults[feature] = float(X_train[feature].median())
    for feature in categorical_features:
        mode = X_train[feature].mode(dropna=True)
        defaults[feature] = str(mode.iloc[0]) if not mode.empty else ""
    return defaults


def option_values_from_training_data(X_train: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    return {
        feature: sorted(X_train[feature].dropna().astype(str).unique().tolist())
        for feature in categorical_features
    }


def extract_model_importance(pipeline: Pipeline, feature_names: list[str]) -> list[dict[str, float | str]]:
    estimator = pipeline.named_steps["model"]
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_[0])

    if importances is None:
        return []

    rows = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in zip(feature_names, importances)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:15]


def save_model_comparison_report(results: list[dict[str, object]], best_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = metrics_to_markdown(results)
    best = next(result for result in results if result["Model"] == best_name)
    lines = [
        "# Model Comparison",
        "",
        table,
        "",
        f"Best model: **{best_name}**",
        "",
        "The best model is selected by ROC-AUC because churn prediction benefits from ranking customers by risk, not only predicting the majority class correctly.",
        "",
        "Accuracy alone is not sufficient for churn modeling because a model can look accurate by predicting the majority class while missing customers who are likely to leave.",
        "",
        "## Best Model Metrics",
        "",
        pd.DataFrame([best]).to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_final_report(best_name: str, results: list[dict[str, object]], path: Path) -> None:
    top = sorted(results, key=lambda row: row["ROC-AUC"], reverse=True)
    lines = [
        "# Final Report",
        "",
        "## Executive Summary",
        "",
        f"The pipeline trains and compares multiple churn classifiers. The selected model is **{best_name}**, chosen by ROC-AUC on the held-out test set.",
        "",
        "## Five EDA and Modeling Takeaways",
        "",
        "1. Month-to-month contracts are usually more churn-prone than long-term contracts.",
        "2. Tenure and total charges capture customer maturity and retention history.",
        "3. Electronic check payment can be associated with higher churn risk in this dataset.",
        "4. Accuracy should be read together with Recall, Precision, F1, and ROC-AUC.",
        "5. Explainability artifacts help translate model output into business actions.",
        "",
        "## Model Ranking",
        "",
        pd.DataFrame(top).to_markdown(index=False),
        "",
        "## Limitations",
        "",
        "This is a portfolio-grade offline model. Production use would require live data validation, monitoring, drift checks, retraining, and cost-sensitive thresholds.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_model_card(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = bundle["metrics"]
    top_features = bundle.get("top_features", [])[:5]
    feature_lines = "\n".join(
        f"- {row['feature']}: {row['importance']}" for row in top_features
    ) or "- Feature importance will be generated by `python -m src.explain`."
    lines = [
        "# Model Card",
        "",
        f"Model: **{bundle['best_model_name']}**",
        "",
        "## Intended Use",
        "",
        "Estimate customer churn risk for educational and portfolio demonstration purposes.",
        "",
        "## Metrics",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## Top Features",
        "",
        feature_lines,
        "",
        "## Ethical and Practical Notes",
        "",
        "Predictions should support human decision-making, not automatically determine customer treatment. The model is trained on a static public dataset and should not be used in production without validation on current business data.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_training() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    quality_summary = build_data_quality_summary(df)
    save_data_quality_report(quality_summary, REPORTS_DIR / "data_quality_report.md")
    save_eda_figures(df, FIGURES_DIR)

    X, y, metadata = prepare_modeling_data(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    X_test.assign(churn=y_test.values).to_csv(PROCESSED_DATA_DIR / "test_set.csv", index=False)

    results: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    for model_name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator, metadata.numeric_features, metadata.categorical_features)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_test, y_test)
        results.append({"Model": model_name, **metrics})
        fitted[model_name] = pipeline

    best_name = max(results, key=lambda row: row["ROC-AUC"])["Model"]
    best_pipeline = fitted[best_name]
    grid = tuning_grid(str(best_name))
    if grid:
        search = GridSearchCV(
            best_pipeline,
            grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_metrics = evaluate_classifier(search.best_estimator_, X_test, y_test)
        tuned_name = f"{best_name} (tuned)"
        results.append({"Model": tuned_name, **tuned_metrics})
        best_name = tuned_name
        best_pipeline = search.best_estimator_

    best_metrics = evaluate_classifier(best_pipeline, X_test, y_test)
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    top_features = extract_model_importance(best_pipeline, feature_names)

    save_confusion_matrix(best_pipeline, X_test, y_test, FIGURES_DIR / "confusion_matrix.png")
    save_roc_curve(best_pipeline, X_test, y_test, FIGURES_DIR / "roc_curve.png")
    save_precision_recall_curve(best_pipeline, X_test, y_test, FIGURES_DIR / "precision_recall_curve.png")
    save_model_comparison_report(results, best_name, REPORTS_DIR / "model_comparison.md")
    save_final_report(best_name, results, REPORTS_DIR / "final_report.md")

    bundle: dict[str, Any] = {
        "model": best_pipeline,
        "best_model_name": best_name,
        "metrics": best_metrics,
        "model_comparison": results,
        "numeric_features": metadata.numeric_features,
        "categorical_features": metadata.categorical_features,
        "feature_names": feature_names,
        "top_features": top_features,
        "default_input": default_input_from_training_data(
            X_train, metadata.numeric_features, metadata.categorical_features
        ),
        "categorical_options": option_values_from_training_data(
            X_train, metadata.categorical_features
        ),
        "decision_threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)
    save_model_card(bundle, DOCS_DIR / "model_card.md")
    return bundle


if __name__ == "__main__":
    trained_bundle = run_training()
    print(f"Saved {trained_bundle['best_model_name']} to {MODEL_PATH}")
