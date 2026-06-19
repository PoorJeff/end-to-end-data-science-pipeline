# Model Card

Model: **XGBoost (tuned)**

## Intended Use

Estimate customer churn risk for educational and portfolio demonstration purposes.

## Metrics

|   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|-----------:|------------:|---------:|-------:|----------:|
|     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8467 |

## Operating Threshold

- Decision threshold: 0.30
- Threshold strategy: Maximize F1 among thresholds with recall >= 0.65

## Explainability Method

SHAP summary plot

## Top 5 Features

- contract: 0.0872
- tenure: 0.0467
- internet_service: 0.0095
- online_security: 0.0069
- monthly_charges: 0.0054

## Business Interpretation

The top features should be read as churn risk signals. Contract type, tenure, charges, and service choices can guide retention outreach, but they should be paired with customer context and business cost analysis.

## Limitations

This public dataset is static and may not represent a current telecom customer base. Production use would require monitoring, periodic retraining, fairness review, and threshold calibration.
