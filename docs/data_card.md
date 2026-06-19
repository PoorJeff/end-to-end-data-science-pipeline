# Data Card

## Dataset

Telco Customer Churn public dataset.

## Task

Predict whether a telecom customer will churn. The target variable is `churn`, mapped to `1` for churn and `0` for no churn.

## Dataset Shape

- Rows: 7,043
- Columns: 21
- Target distribution: 73.46% no churn, 26.54% churn
- Raw file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Feature Groups

- Demographics: gender, senior citizen, partner, dependents
- Account history: tenure, contract, payment method, paperless billing
- Services: phone, internet, online security, online backup, device protection, tech support, streaming TV, streaming movies
- Billing: monthly charges, total charges

## Data Quality Notes

- `total_charges` contains 11 blank values, treated as missing values.
- Raw data is preserved unchanged.
- Missing values are imputed inside a scikit-learn pipeline fit only on training data.
- Customer ID is excluded from modeling to avoid identifier leakage.

## Intended Use

This dataset is used for educational portfolio demonstration of an end-to-end data science workflow. It is not current production data.

## Limitations

- No time-series customer behavior.
- No marketing campaign history.
- No customer support ticket text.
- No acquisition channel or competitor offer data.
- Public historical data may not represent a current telecom population.
