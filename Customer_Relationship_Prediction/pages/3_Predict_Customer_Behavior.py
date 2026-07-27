import pandas as pd
import streamlit as st

from customer_behavior import app_support, config

st.set_page_config(page_title="Predict · Customer Behavior", page_icon="\U0001F52E", layout="wide")
st.title("Predict Customer Behavior")

if not app_support.models_available():
    st.warning(
        "No trained models found yet. Run `poetry run elt` then `poetry run train` from the "
        "project root first.",
        icon="⚠️",
    )
    st.stop()

column_metadata = app_support.load_column_metadata()
feature_stats = app_support.load_feature_stats()
all_columns = column_metadata["all_columns"]


def input_key(column: str) -> str:
    return f"input_{column}"


# Seed every feature's widget state once, on first load, so nothing resets on rerun.
for col in all_columns:
    st.session_state.setdefault(input_key(col), feature_stats[col]["median"])
st.session_state.setdefault("loaded_label", None)
st.session_state.setdefault("loaded_for_target", None)

st.markdown("#### 1. Choose what to predict")
target = st.selectbox("Target", config.TARGETS, format_func=str.capitalize, label_visibility="collapsed")

importance_data = app_support.load_feature_importance(target)
top_features = importance_data["features"][:12]

st.markdown("#### 2. Set the customer's features")
st.caption(
    f"The features are anonymized (`Var6`, `Var13`, ...), so there's no real-world label to tell "
    "you what a sensible value looks like. Use one of the two buttons below to fill the fields "
    "for you, or edit them by hand. Whatever you do, the prediction at the bottom updates "
    "instantly — there's no separate submit step."
)

button_cols = st.columns([1, 1, 3])
load_clicked = button_cols[0].button(
    "\U0001F3B2 Load a real customer",
    help=(
        "Fills every field with one actual customer's data, picked at random from the data held "
        "out during training (the models never saw it). Also reveals what really happened to that "
        "customer, so you can check the prediction against reality."
    ),
)
reset_clicked = button_cols[1].button(
    "\U0001F504 Reset to typical customer",
    help=(
        "Sets every field to its median value across the whole dataset — a synthetic 'average' "
        "customer. Useful as a neutral starting point before you tweak individual fields by hand."
    ),
)

if load_clicked:
    test_data = app_support.load_test_data(target)
    label_col = test_data.columns[-1]
    sample = test_data.sample(1).iloc[0]
    for col in all_columns:
        st.session_state[input_key(col)] = float(sample[col])
    st.session_state.loaded_label = int(sample[label_col])
    st.session_state.loaded_for_target = target
    st.rerun()

if reset_clicked:
    for col in all_columns:
        st.session_state[input_key(col)] = feature_stats[col]["median"]
    st.session_state.loaded_label = None
    st.session_state.loaded_for_target = None
    st.rerun()

if st.session_state.loaded_label is not None and st.session_state.loaded_for_target == target:
    st.caption("Showing a real customer's data, loaded from the held-out test set.")
else:
    st.caption(
        f"Showing the typical (median) customer, with the {len(top_features)} features most "
        f"important for predicting {target} editable below."
    )

input_cols = st.columns(3)
for i, feature in enumerate(top_features):
    stats = feature_stats[feature]
    input_cols[i % 3].number_input(
        feature,
        min_value=float(stats["min"]),
        max_value=float(stats["max"]),
        key=input_key(feature),
        help=f"Observed range: {stats['min']:.3g} to {stats['max']:.3g} (median {stats['median']:.3g})",
    )

row = {col: st.session_state[input_key(col)] for col in all_columns}
X_input = pd.DataFrame([[row[col] for col in all_columns]], columns=all_columns)

rf_model = app_support.load_model(target, "random_forest")
lr_model = app_support.load_model(target, "logistic_regression")
rf_proba = float(rf_model.predict_proba(X_input)[0, 1])
lr_proba = float(lr_model.predict_proba(X_input)[0, 1])


def likelihood_label(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"


st.markdown("#### 3. Prediction")
pred_cols = st.columns(2)
pred_cols[0].metric("Random Forest", f"{rf_proba:.1%}", likelihood_label(rf_proba), delta_color="off")
pred_cols[1].metric(
    "Logistic Regression", f"{lr_proba:.1%}", likelihood_label(lr_proba), delta_color="off"
)

if st.session_state.loaded_label is not None and st.session_state.loaded_for_target == target:
    actual = "Yes" if st.session_state.loaded_label == 1 else "No"
    st.info(f"Loaded example's actual outcome (held-out test data): **{actual}**", icon="\U0001F3AF")