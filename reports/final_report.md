# Final Report

## Executive Summary

The pipeline trains and compares multiple churn classifiers. The selected model is **XGBoost (tuned)**, chosen by ROC-AUC on the held-out test set.

## Five EDA and Modeling Takeaways

1. Month-to-month contracts are usually more churn-prone than long-term contracts.
2. Tenure and total charges capture customer maturity and retention history.
3. Electronic check payment can be associated with higher churn risk in this dataset.
4. Accuracy should be read together with Recall, Precision, F1, and ROC-AUC.
5. Explainability artifacts help translate model output into business actions.

## Model Ranking

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| XGBoost (tuned)     |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8467 |
| XGBoost             |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8452 |
| Gradient Boosting   |     0.8062 |      0.6735 |   0.5241 | 0.5895 |    0.8434 |
| Logistic Regression |     0.7381 |      0.5043 |   0.7834 | 0.6136 |    0.8413 |
| Random Forest       |     0.7835 |      0.5848 |   0.6364 | 0.6095 |    0.8331 |
| SVM                 |     0.7438 |      0.5115 |   0.7727 | 0.6155 |    0.8211 |

## Limitations

This is a portfolio-grade offline model. Production use would require live data validation, monitoring, drift checks, retraining, and cost-sensitive thresholds.
