"""Model training: Random Forest and Logistic Regression, both with undersampling
to counter the heavy class imbalance in churn/appetency/upselling labels.

Each trainer returns a fitted imblearn Pipeline. imblearn pipelines only apply the
resampling step during `fit`; `predict`/`predict_proba` skip straight to the
estimator, so the returned pipeline is directly reusable at inference time.
"""

import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from customer_behavior import config


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
    random_state: int = config.MODEL_RANDOM_STATE,
) -> Pipeline:
    """Undersample then fit a class-weighted Random Forest."""
    pipeline = Pipeline(
        steps=[
            ("undersample", RandomUnderSampler(random_state=random_state)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators"),
                    max_features=params.get("max_features"),
                    min_samples_leaf=params.get("min_samples_leaf"),
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train.values.ravel())
    return pipeline


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
    random_state: int = config.MODEL_RANDOM_STATE,
) -> Pipeline:
    """Scale, undersample, then fit a class-weighted Logistic Regression."""
    pipeline = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
            ("undersample", RandomUnderSampler(random_state=random_state)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_state,
                    **params,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train.values.ravel())
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """ROC AUC score of the positive class on held-out data."""
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)