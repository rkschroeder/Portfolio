import pandas as pd
import plotly.express as px
import streamlit as st

from customer_behavior import app_support, config, palette

st.set_page_config(page_title="Model Performance · Customer Behavior", page_icon="\U0001F916", layout="wide")
st.title("Model Performance")

if not app_support.models_available():
    st.warning(
        "No trained models found yet. Run `poetry run train` from the project root first.",
        icon="⚠️",
    )
    st.stop()

metrics = app_support.load_metrics()

rows = []
for target in config.TARGETS:
    rows.append({"Target": target.capitalize(), "Model": "Random Forest", "ROC AUC": metrics[target]["random_forest"]})
    rows.append(
        {
            "Target": target.capitalize(),
            "Model": "Logistic Regression",
            "ROC AUC": metrics[target]["logistic_regression"],
        }
    )
metrics_df = pd.DataFrame(rows)

st.header("Random Forest vs Logistic Regression")
st.caption(
    "ROC AUC is measured once across the whole held-out test set (0.5 = random guessing, "
    "1.0 = perfect ranking). It describes the model overall, not any single customer — it won't "
    "match the per-customer percentages on the Predict page."
)
fig = px.bar(
    metrics_df,
    x="Target",
    y="ROC AUC",
    color="Model",
    barmode="group",
    color_discrete_map={"Random Forest": palette.CATEGORICAL[0], "Logistic Regression": palette.CATEGORICAL[1]},
    text_auto=".3f",
    range_y=[0, 1],
)
fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, height=450, legend_title_text="")
st.plotly_chart(fig, width="stretch")

with st.expander("View ROC AUC as a table"):
    st.dataframe(
        metrics_df.pivot(index="Target", columns="Model", values="ROC AUC").style.format("{:.4f}"),
        width="stretch",
    )

st.header("Feature importance")
st.caption(
    "Random Forest importances used to rank features. All features are still used for "
    "training — this ranking also drives which inputs appear on the Predict page."
)

target_choice = st.selectbox("Target", config.TARGETS, format_func=str.capitalize)
importance_data = app_support.load_feature_importance(target_choice)

if importance_data is not None:
    top_n = st.slider("Number of features to show", 5, min(30, len(importance_data["features"])), 15)
    importance_df = pd.DataFrame(
        {
            "Feature": importance_data["features"][:top_n],
            "Importance": importance_data["importances"][:top_n],
        }
    ).sort_values("Importance")

    importance_fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[palette.SEQUENTIAL_BLUE[3]],
    )
    importance_fig.update_layout(**palette.PLOTLY_LAYOUT_DEFAULTS, height=max(350, top_n * 28))
    st.plotly_chart(importance_fig, width="stretch")