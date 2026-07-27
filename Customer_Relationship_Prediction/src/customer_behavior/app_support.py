"""Cached loaders shared by the Streamlit pages: raw data, trained model
artifacts, and the metadata produced by `poetry run train`.
"""

import json

import joblib
import pandas as pd
import streamlit as st

from customer_behavior import config, data


def models_available() -> bool:
    """Whether `poetry run train` has been run at least once."""
    return (config.MODELS_DIR / "metrics.json").exists()


def warehouse_available() -> bool:
    """Whether `poetry run elt` has been run at least once."""
    return data.warehouse_available()


@st.cache_data(show_spinner="Loading data from the warehouse...")
def load_raw_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return data.read_warehouse()


@st.cache_data
def load_metrics() -> dict | None:
    path = config.MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_importance(target: str) -> dict | None:
    path = config.MODELS_DIR / f"{target}_feature_importance.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_feature_stats() -> dict | None:
    path = config.MODELS_DIR / "feature_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_column_metadata() -> dict | None:
    path = config.MODELS_DIR / "column_metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_test_data(target: str) -> pd.DataFrame | None:
    path = config.MODELS_DIR / f"{target}_test_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(target: str, model_type: str):
    """model_type is 'random_forest' or 'logistic_regression'."""
    path = config.MODELS_DIR / f"{target}_{model_type}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)