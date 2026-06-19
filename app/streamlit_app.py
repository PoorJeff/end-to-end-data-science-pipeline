from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import default_customer, load_model_bundle, predict_churn


st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Customer Churn Predictor")

try:
    bundle = load_model_bundle()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

defaults = default_customer(bundle)
categorical_options = bundle.get("categorical_options", {})

left, right = st.columns([1, 1])
payload = {}

with left:
    st.subheader("Customer Profile")
    for feature in bundle.get("categorical_features", []):
        options = categorical_options.get(feature, [])
        default = str(defaults.get(feature, options[0] if options else ""))
        if default not in options and options:
            options = [default, *options]
        payload[feature] = st.selectbox(
            feature.replace("_", " ").title(),
            options or [default],
            index=(options.index(default) if default in options else 0),
        )

with right:
    st.subheader("Account Values")
    for feature in bundle.get("numeric_features", []):
        value = float(defaults.get(feature, 0.0))
        payload[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=0.0,
            value=value,
            step=1.0 if feature in {"tenure", "senior_citizen"} else 5.0,
        )

if st.button("Predict churn", type="primary"):
    result = predict_churn(bundle, payload)
    st.metric("Churn probability", f"{result['churn_probability']:.1%}")
    st.metric("Prediction", result["prediction_label"])

    top_features = result["top_contributing_features"]
    if top_features:
        st.subheader("Top contributing global features")
        st.dataframe(pd.DataFrame(top_features), hide_index=True, use_container_width=True)

st.caption(
    "Educational demo only. Predictions should support retention analysis, not automated customer decisions."
)
