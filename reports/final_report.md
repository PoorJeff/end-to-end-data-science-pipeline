# Final Report

## Executive Summary

The pipeline trains and compares multiple churn classifiers. The selected model is **XGBoost (tuned)**, chosen by ROC-AUC on the held-out test set. The recommended operating threshold is **0.30**, which supports a higher-recall retention workflow.

## Five EDA and Modeling Takeaways

1. Month-to-month contracts are usually more churn-prone than long-term contracts.
2. Tenure and total charges capture customer maturity and retention history.
3. Electronic check payment can be associated with higher churn risk in this dataset.
4. Accuracy should be read together with Recall, Precision, F1, and ROC-AUC.
5. Threshold and lift analysis translate probability scores into a practical outreach list.

## Model Ranking

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| XGBoost (tuned)     |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8467 |
| XGBoost             |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8452 |
| Gradient Boosting   |     0.8062 |      0.6735 |   0.5241 | 0.5895 |    0.8434 |
| Logistic Regression |     0.7381 |      0.5043 |   0.7834 | 0.6136 |    0.8413 |
| Random Forest       |     0.7835 |      0.5848 |   0.6364 | 0.6095 |    0.8331 |
| SVM                 |     0.7438 |      0.5115 |   0.7727 | 0.6155 |    0.8211 |

## Operating Threshold

|   threshold |   Precision |   Recall |     F1 |   flagged_customers |   contact_rate |   captured_churners |   churn_capture_rate |
|------------:|------------:|---------:|-------:|--------------------:|---------------:|--------------------:|---------------------:|
|         0.3 |      0.5338 |   0.7807 | 0.6341 |                 547 |         0.3882 |                 292 |               0.7807 |

## Top Lift Segments

|   decile |   customers |   churners |   churn_rate |   average_score |   lift |   cumulative_churn_capture_rate |
|---------:|------------:|-----------:|-------------:|----------------:|-------:|--------------------------------:|
|        1 |         141 |        107 |       0.7589 |          0.7549 | 2.8589 |                          0.2861 |
|        2 |         141 |         87 |       0.617  |          0.5735 | 2.3246 |                          0.5187 |
|        3 |         141 |         55 |       0.3901 |          0.4489 | 1.4695 |                          0.6658 |

## Business Recommendation

Use the selected threshold to create a high-priority retention queue. Start with the top lift deciles, pair outreach with contract and support interventions, and tune the threshold when campaign capacity or contact cost changes.

## Limitations

This is a portfolio-grade offline model. Production use would require live data validation, monitoring, drift checks, retraining, and cost-sensitive thresholds.
