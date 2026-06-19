from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def prediction_scores(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return scores
        return (scores - min_score) / (max_score - min_score)
    return model.predict(X)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = prediction_scores(model, X_test)
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, y_score)), 4),
    }


def save_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["No Churn", "Churn"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_score = prediction_scores(model, X_test)
    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_precision_recall_curve(model, X_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_score = prediction_scores(model, X_test)
    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def metrics_to_markdown(results: list[dict[str, object]]) -> str:
    table = pd.DataFrame(results)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    for column in metric_columns:
        table[column] = table[column].map(lambda value: f"{float(value):.4f}")
    return table[["Model", *metric_columns]].to_markdown(index=False)
