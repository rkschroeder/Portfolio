"""Frequency encoding of categorical columns and Random-Forest-based feature ranking."""

import numpy as np
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.ensemble import RandomForestClassifier

from customer_behavior import config


def apply_frequency_encoding(
    df: pd.DataFrame,
    categorical_cols: pd.Index,
    encoder: CountFrequencyEncoder | None = None,
) -> tuple[pd.DataFrame, CountFrequencyEncoder]:
    """Replace each categorical column with how often its value occurs in `df`.

    If `encoder` is None, a new CountFrequencyEncoder is fit on `df`. Passing an
    already-fit encoder lets you apply the same frequencies to new data.
    """
    if encoder is None:
        encoder = CountFrequencyEncoder(encoding_method="frequency")
        encoded = encoder.fit_transform(df[categorical_cols])
    else:
        encoded = encoder.transform(df[categorical_cols])

    df_encoded = df.copy()
    df_encoded[categorical_cols] = encoded
    return df_encoded, encoder


def rank_features_by_importance(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_estimators: int = config.FEATURE_SELECTION_N_ESTIMATORS,
    random_state: int = config.MODEL_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank columns of X_train by Random Forest feature importance, descending.

    Returns (feature_names_sorted, importances_sorted).
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train.values.ravel())
    importances = clf.feature_importances_

    variables = np.array(X_train.columns)
    indices = np.argsort(importances)[::-1]
    return variables[indices], importances[indices]