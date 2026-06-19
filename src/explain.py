from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import DOCS_DIR, FIGURES_DIR, MODEL_PATH, RANDOM_STATE, REPORTS_DIR
from src.data_loader import load_raw_data
from src.features import prepare_modeling_data
from src.preprocessing import make_train_test_split


def save_importance_plot(importances: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importances.head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#2f6f9f")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def permutation_importance_report(bundle: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        bundle["model"],
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    frame = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    )
    return frame.sort_values("importance", ascending=False)


def try_save_shap_summary(bundle: dict, X_test: pd.DataFrame, path: Path) -> bool:
    try:
        import shap

        pipeline = bundle["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        sample = X_test.head(200)
        transformed = preprocessor.transform(sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = bundle.get("feature_names", [])
        try:
            masker = shap.maskers.Independent(transformed, max_samples=transformed.shape[0])
            explainer = shap.Explainer(estimator, masker, feature_names=feature_names)
            values = explainer(transformed)
            shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=15)
        except NotImplementedError:
            explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
            values = explainer.shap_values(transformed)
            shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        plt.close()
        return False


def update_model_card(bundle: dict, importances: pd.DataFrame, used_shap: bool) -> None:
    top_features = importances.head(5)
    feature_lines = "\n".join(
        f"- {row.feature}: {row.importance:.4f}" for row in top_features.itertuples()
    )
    explainability = "SHAP summary plot" if used_shap else "Permutation importance fallback"
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
        pd.DataFrame([bundle["metrics"]]).to_markdown(index=False),
        "",
        "## Explainability Method",
        "",
        explainability,
        "",
        "## Top 5 Features",
        "",
        feature_lines,
        "",
        "## Business Interpretation",
        "",
        "The top features should be read as churn risk signals. Contract type, tenure, charges, and service choices can guide retention outreach, but they should be paired with customer context and business cost analysis.",
        "",
        "## Limitations",
        "",
        "This public dataset is static and may not represent a current telecom customer base. Production use would require monitoring, periodic retraining, fairness review, and threshold calibration.",
        "",
    ]
    (DOCS_DIR / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def run_explainability() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train` before generating explanations.")

    bundle = joblib.load(MODEL_PATH)
    df = load_raw_data()
    X, y, _ = prepare_modeling_data(df)
    _, X_test, _, y_test = make_train_test_split(X, y)

    importances = permutation_importance_report(bundle, X_test, y_test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importances.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    shap_path = FIGURES_DIR / "shap_summary.png"
    used_shap = try_save_shap_summary(bundle, X_test, shap_path)
    if not used_shap:
        save_importance_plot(importances, shap_path, "Permutation Importance Summary")
    save_importance_plot(importances, FIGURES_DIR / "feature_importance.png", "Global Feature Importance")

    bundle["top_features"] = [
        {"feature": row.feature, "importance": round(float(row.importance), 6)}
        for row in importances.head(15).itertuples()
    ]
    joblib.dump(bundle, MODEL_PATH)
    update_model_card(bundle, importances, used_shap)
    return importances


if __name__ == "__main__":
    report = run_explainability()
    print(report.head(10).to_string(index=False))
