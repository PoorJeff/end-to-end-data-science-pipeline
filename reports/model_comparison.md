# Model Comparison

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression |     0.7381 |      0.5043 |   0.7834 | 0.6136 |    0.8413 |
| Random Forest       |     0.7835 |      0.5848 |   0.6364 | 0.6095 |    0.8331 |
| Gradient Boosting   |     0.8062 |      0.6735 |   0.5241 | 0.5895 |    0.8434 |
| SVM                 |     0.7438 |      0.5115 |   0.7727 | 0.6155 |    0.8211 |
| XGBoost             |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8452 |
| XGBoost (tuned)     |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8467 |

Best model: **XGBoost (tuned)**

The best model is selected by ROC-AUC because churn prediction benefits from ranking customers by risk, not only predicting the majority class correctly.

Accuracy alone is not sufficient for churn modeling because a model can look accurate by predicting the majority class while missing customers who are likely to leave.

## Best Model Metrics

| Model           |   Accuracy |   Precision |   Recall |     F1 |   ROC-AUC |
|:----------------|-----------:|------------:|---------:|-------:|----------:|
| XGBoost (tuned) |     0.8055 |      0.6678 |   0.5321 | 0.5923 |    0.8467 |
