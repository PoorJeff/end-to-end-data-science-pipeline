from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


DEFAULT_THRESHOLDS = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]


def threshold_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    y_true = pd.Series(y_true).reset_index(drop=True).astype(int)
    y_score = pd.Series(y_score).reset_index(drop=True).astype(float)
    total_customers = len(y_true)
    total_churners = int(y_true.sum())
    rows: list[dict[str, float | int]] = []

    for threshold in thresholds:
        y_pred = y_score >= threshold
        captured_churners = int(((y_true == 1) & y_pred).sum())
        flagged_customers = int(y_pred.sum())
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "flagged_customers": flagged_customers,
                "contact_rate": round(flagged_customers / total_customers, 4),
                "captured_churners": captured_churners,
                "churn_capture_rate": round(captured_churners / total_churners, 4) if total_churners else 0.0,
            }
        )

    return pd.DataFrame(rows)


def select_operating_threshold(
    table: pd.DataFrame,
    min_recall: float = 0.65,
) -> dict[str, float | int]:
    eligible = table[table["Recall"] >= min_recall]
    candidates = eligible if not eligible.empty else table
    selected = candidates.sort_values(["F1", "Precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return selected.to_dict()


def lift_table(y_true: pd.Series, y_score: pd.Series, buckets: int = 10) -> pd.DataFrame:
    scored = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True).astype(int),
            "score": pd.Series(y_score).reset_index(drop=True).astype(float),
        }
    ).sort_values("score", ascending=False)
    scored["decile"] = pd.qcut(scored["score"].rank(method="first", ascending=False), q=buckets, labels=False) + 1
    base_rate = float(scored["actual"].mean())
    total_churners = float(scored["actual"].sum())

    rows = []
    cumulative_churners = 0.0
    for decile, group in scored.groupby("decile", sort=True):
        churners = float(group["actual"].sum())
        cumulative_churners += churners
        churn_rate = float(group["actual"].mean())
        rows.append(
            {
                "decile": int(decile),
                "customers": int(len(group)),
                "churners": int(churners),
                "churn_rate": round(churn_rate, 4),
                "average_score": round(float(group["score"].mean()), 4),
                "lift": round(churn_rate / base_rate, 4) if base_rate else 0.0,
                "cumulative_churn_capture_rate": round(cumulative_churners / total_churners, 4) if total_churners else 0.0,
            }
        )

    return pd.DataFrame(rows)


def classify_risk(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= threshold * 0.6:
        return "Medium"
    return "Low"


def recommend_retention_action(probability: float, threshold: float) -> str:
    band = classify_risk(probability, threshold)
    if band == "High":
        return "High-priority retention outreach: offer contract review, service support, or targeted incentive."
    if band == "Medium":
        return "Monitor and nurture: send proactive education, usage tips, or satisfaction check-in."
    return "Low immediate risk: continue standard engagement and monitor for future changes."
