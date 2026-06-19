# Data Quality Report

- Rows: 7043
- Columns: 21
- Duplicate rows: 0

## Missing Value Table

|                   |   missing_count |   missing_percent |
|:------------------|----------------:|------------------:|
| total_charges     |              11 |              0.16 |
| customer_id       |               0 |              0    |
| gender            |               0 |              0    |
| senior_citizen    |               0 |              0    |
| partner           |               0 |              0    |
| dependents        |               0 |              0    |
| tenure            |               0 |              0    |
| phone_service     |               0 |              0    |
| multiple_lines    |               0 |              0    |
| internet_service  |               0 |              0    |
| online_security   |               0 |              0    |
| online_backup     |               0 |              0    |
| device_protection |               0 |              0    |
| tech_support      |               0 |              0    |
| streaming_tv      |               0 |              0    |
| streaming_movies  |               0 |              0    |
| contract          |               0 |              0    |
| paperless_billing |               0 |              0    |
| payment_method    |               0 |              0    |
| monthly_charges   |               0 |              0    |
| churn             |               0 |              0    |

## Target Distribution

| churn   |   count |   percent |
|:--------|--------:|----------:|
| No      |    5174 |     73.46 |
| Yes     |    1869 |     26.54 |

## Numeric Feature Summary

|                 |   count |   mean |    std |   min |   25% |   50% |   75% |    max |
|:----------------|--------:|-------:|-------:|------:|------:|------:|------:|-------:|
| senior_citizen  |    7043 |  0.162 |  0.369 |  0    |   0   |  0    |  0    |   1    |
| tenure          |    7043 | 32.371 | 24.559 |  0    |   9   | 29    | 55    |  72    |
| monthly_charges |    7043 | 64.762 | 30.09  | 18.25 |  35.5 | 70.35 | 89.85 | 118.75 |

## Categorical Feature Cardinality

|                   |   unique_values |
|:------------------|----------------:|
| customer_id       |            7043 |
| total_charges     |            6531 |
| payment_method    |               4 |
| contract          |               3 |
| multiple_lines    |               3 |
| online_backup     |               3 |
| device_protection |               3 |
| tech_support      |               3 |
| online_security   |               3 |
| internet_service  |               3 |
| streaming_tv      |               3 |
| streaming_movies  |               3 |
| gender            |               2 |
| phone_service     |               2 |
| dependents        |               2 |
| partner           |               2 |
| paperless_billing |               2 |
| churn             |               2 |
