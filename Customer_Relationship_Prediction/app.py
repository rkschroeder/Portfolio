import streamlit as st

from customer_behavior import app_support, config

st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="\U0001F4CA",
    layout="wide",
)

st.title("Customer Behavior Prediction")
st.caption("Churn · Appetency · Upselling — KDD Cup 2009 (Orange) dataset")

st.markdown(
    """
This app predicts three customer behaviors for a French telecom operator (Orange):

- **Churn** — the likelihood a customer leaves for a competitor
- **Appetency** — the likelihood a customer buys additional products or services
- **Upselling** — the likelihood a customer buys an upgrade or add-on

The goal is to let a business target retention offers and upsell campaigns at the
customers most likely to respond, instead of marketing to everyone.

The data comes from the [KDD Cup 2009](https://kdd.org/kdd-cup/view/kdd-cup-2009/Intro)
dataset: 50,000 customers, 230 anonymized features, heavy class imbalance, and a lot
of missing data. Raw files are extracted and loaded into a DuckDB warehouse, then
the pipeline handles cleaning with column/row filtering, frequency encoding for
categorical features, undersampling for class imbalance, and compares Random
Forest against Logistic Regression using ROC AUC.
"""
)

if not app_support.warehouse_available():
    st.warning(
        "No warehouse found yet. From the project root, run:\n\n"
        "```\npoetry install\npoetry run elt\npoetry run train\n```\n\n"
        "`elt` extracts the raw files into a DuckDB warehouse; `train` reads from "
        "it and writes the model artifacts the other pages need to `models/`.",
        icon="⚠️",
    )
elif not app_support.models_available():
    st.warning(
        "Warehouse found, but no trained models yet. From the project root, run:\n\n"
        "```\npoetry run train\n```\n\n"
        "This trains all six models (Random Forest + Logistic Regression × 3 targets) "
        "and writes the artifacts the other pages need to `models/`.",
        icon="⚠️",
    )
else:
    metrics = app_support.load_metrics()
    cols = st.columns(3)
    for col, target in zip(cols, config.TARGETS):
        best_auc = max(metrics[target].values())
        col.metric(
            f"{target.capitalize()} — model quality (ROC AUC)",
            f"{best_auc:.3f}",
            help=(
                "How well the model ranks customers overall, on a 0.5 (random guessing) to 1.0 "
                "(perfect) scale — measured once across the whole held-out test set. This is not "
                "a probability, and it won't match the per-customer percentages on the Predict "
                "page: a single customer's predicted probability can be anywhere from 0% to 100% "
                "even for a model with a modest ROC AUC like this."
            ),
        )

st.markdown(
    """
### Explore

Use the sidebar to navigate:

1. **Exploratory Data Analysis** — label distribution, missing data, and cardinality
   in the raw dataset.
2. **Model Performance** — Random Forest vs Logistic Regression ROC AUC per target,
   plus feature importances.
3. **Predict Customer Behavior** — adjust a customer's top features (or load a random
   held-out example) and get a live prediction from both models.
"""
)