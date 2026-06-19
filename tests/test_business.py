import pandas as pd

from src.business import (
    classify_risk,
    lift_table,
    recommend_retention_action,
    select_operating_threshold,
    threshold_metrics,
)


def test_threshold_metrics_selects_highest_f1_threshold_that_meets_recall_floor():
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    y_score = pd.Series([0.91, 0.72, 0.41, 0.67, 0.30, 0.10])

    table = threshold_metrics(y_true, y_score, thresholds=[0.3, 0.5, 0.7])
    selected = select_operating_threshold(table, min_recall=0.6)

    assert list(table["threshold"]) == [0.3, 0.5, 0.7]
    assert selected["threshold"] == 0.7
    assert selected["Recall"] >= 0.6
    assert selected["F1"] == table.loc[table["threshold"] == 0.7, "F1"].iloc[0]


def test_threshold_metrics_ignores_non_matching_series_indexes():
    y_true = pd.Series([1, 0, 1], index=[10, 20, 30])
    y_score = pd.Series([0.8, 0.7, 0.2], index=[100, 200, 300])

    table = threshold_metrics(y_true, y_score, thresholds=[0.5])

    assert table.loc[0, "captured_churners"] == 1
    assert table.loc[0, "flagged_customers"] == 2


def test_lift_table_ranks_highest_risk_customers_first():
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_score = pd.Series([0.95, 0.10, 0.80, 0.40, 0.70, 0.30])

    table = lift_table(y_true, y_score, buckets=3)

    assert list(table["decile"]) == [1, 2, 3]
    assert table.loc[0, "churners"] == 2
    assert table.loc[0, "churn_rate"] > table.loc[1, "churn_rate"]
    assert table.loc[0, "lift"] > 1.0


def test_risk_band_and_retention_action_follow_selected_threshold():
    assert classify_risk(0.75, threshold=0.5) == "High"
    assert classify_risk(0.40, threshold=0.5) == "Medium"
    assert classify_risk(0.10, threshold=0.5) == "Low"

    action = recommend_retention_action(0.75, threshold=0.5)

    assert "priority" in action.lower()
    assert "retention" in action.lower()
