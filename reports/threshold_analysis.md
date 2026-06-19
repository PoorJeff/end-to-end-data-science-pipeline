# Threshold Analysis

The model outputs churn probabilities. The operating threshold converts those probabilities into action decisions.

## Selected Operating Threshold

|   threshold |   Precision |   Recall |     F1 |   flagged_customers |   contact_rate |   captured_churners |   churn_capture_rate |
|------------:|------------:|---------:|-------:|--------------------:|---------------:|--------------------:|---------------------:|
|         0.3 |      0.5338 |   0.7807 | 0.6341 |                 547 |         0.3882 |                 292 |               0.7807 |

This threshold is selected by maximizing F1 among thresholds that meet the recall floor. The goal is to catch a useful share of likely churners without contacting every customer.

## Threshold Tradeoff Table

|   threshold |   Precision |   Recall |     F1 |   flagged_customers |   contact_rate |   captured_churners |   churn_capture_rate |
|------------:|------------:|---------:|-------:|--------------------:|---------------:|--------------------:|---------------------:|
|        0.2  |      0.4731 |   0.8449 | 0.6065 |                 668 |         0.4741 |                 316 |               0.8449 |
|        0.25 |      0.5092 |   0.8128 | 0.6262 |                 597 |         0.4237 |                 304 |               0.8128 |
|        0.3  |      0.5338 |   0.7807 | 0.6341 |                 547 |         0.3882 |                 292 |               0.7807 |
|        0.35 |      0.5616 |   0.7193 | 0.6307 |                 479 |         0.34   |                 269 |               0.7193 |
|        0.4  |      0.5851 |   0.6524 | 0.6169 |                 417 |         0.296  |                 244 |               0.6524 |
|        0.45 |      0.6311 |   0.5856 | 0.6075 |                 347 |         0.2463 |                 219 |               0.5856 |
|        0.5  |      0.6678 |   0.5321 | 0.5923 |                 298 |         0.2115 |                 199 |               0.5321 |
|        0.55 |      0.6975 |   0.4439 | 0.5425 |                 238 |         0.1689 |                 166 |               0.4439 |
|        0.6  |      0.7273 |   0.3422 | 0.4655 |                 176 |         0.1249 |                 128 |               0.3422 |

## Lift Table

|   decile |   customers |   churners |   churn_rate |   average_score |   lift |   cumulative_churn_capture_rate |
|---------:|------------:|-----------:|-------------:|----------------:|-------:|--------------------------------:|
|        1 |         141 |        107 |       0.7589 |          0.7549 | 2.8589 |                          0.2861 |
|        2 |         141 |         87 |       0.617  |          0.5735 | 2.3246 |                          0.5187 |
|        3 |         141 |         55 |       0.3901 |          0.4489 | 1.4695 |                          0.6658 |
|        4 |         141 |         46 |       0.3262 |          0.3386 | 1.2291 |                          0.7888 |
|        5 |         141 |         27 |       0.1915 |          0.2252 | 0.7214 |                          0.861  |
|        6 |         140 |         28 |       0.2    |          0.1461 | 0.7535 |                          0.9358 |
|        7 |         141 |         14 |       0.0993 |          0.0877 | 0.3741 |                          0.9733 |
|        8 |         141 |          8 |       0.0567 |          0.0471 | 0.2138 |                          0.9947 |
|        9 |         141 |          0 |       0      |          0.0261 | 0      |                          0.9947 |
|       10 |         141 |          2 |       0.0142 |          0.0127 | 0.0534 |                          1      |
