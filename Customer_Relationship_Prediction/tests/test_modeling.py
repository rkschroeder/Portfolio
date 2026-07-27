import numpy as np
import pandas as pd

from customer_behavior import modeling


def _make_imbalanced_classification_data(n=400, n_features=5, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n, n_features))
    # Imbalanced: positive class driven by feature 0, ~10% positive rate.
    logits = X[:, 0] * 2 - 2
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)

    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_df = pd.DataFrame({"label": y})
    return X_df, y_df


def test_train_random_forest_fits_and_evaluates():
    X, y = _make_imbalanced_classification_data()
    X_train, X_test = X.iloc[:300], X.iloc[300:]
    y_train, y_test = y.iloc[:300], y.iloc[300:]

    pipeline = modeling.train_random_forest(
        X_train, y_train, params={"n_estimators": 50, "max_features": 3, "min_samples_leaf": 2}
    )
    auc = modeling.evaluate_model(pipeline, X_test, y_test)

    assert 0.0 <= auc <= 1.0
    assert hasattr(pipeline, "predict_proba")


def test_train_logistic_regression_fits_and_evaluates():
    X, y = _make_imbalanced_classification_data()
    X_train, X_test = X.iloc[:300], X.iloc[300:]
    y_train, y_test = y.iloc[:300], y.iloc[300:]

    pipeline = modeling.train_logistic_regression(X_train, y_train, params={"C": 0.8, "max_iter": 100})
    auc = modeling.evaluate_model(pipeline, X_test, y_test)

    assert 0.0 <= auc <= 1.0
    assert hasattr(pipeline, "predict_proba")