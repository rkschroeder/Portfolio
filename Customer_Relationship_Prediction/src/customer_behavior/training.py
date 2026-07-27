"""End-to-end training pipeline: load raw data, clean it, encode categoricals,
train Random Forest + Logistic Regression for churn/appetency/upselling, and
persist every artifact the Streamlit app needs under `models/`.

Run with: `poetry run train`
"""

import json

import joblib
from sklearn.model_selection import train_test_split

from customer_behavior import config, data, features, modeling, preprocessing


def compute_feature_stats(df) -> dict:
    """Per-column min/median/max, used by the app to build sensible input widgets."""
    return {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in df.columns
    }


def main() -> None:
    if not data.warehouse_available():
        raise SystemExit(
            f"No warehouse found at {config.WAREHOUSE_PATH}. Run `poetry run elt` first."
        )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from the warehouse...")
    raw_features, raw_labels = data.read_warehouse()

    print(f"Raw shape: {raw_features.shape}")
    df_clean, labels_clean, categorical_cols = preprocessing.clean_dataset(raw_features, raw_labels)
    print(f"Shape after cleaning: {df_clean.shape} ({len(categorical_cols)} categorical columns)")

    # Frequency encoder is fit on the full cleaned dataset before splitting, matching
    # the original notebook. This is a mild train/test leak (see KNOWN_ISSUES.md) that
    # we're preserving intentionally to keep parity with the notebook's reported scores.
    df_encoded, encoder = features.apply_frequency_encoding(df_clean, categorical_cols)
    joblib.dump(encoder, config.MODELS_DIR / "frequency_encoder.joblib")

    feature_stats = compute_feature_stats(df_encoded)
    with open(config.MODELS_DIR / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    column_metadata = {
        "all_columns": df_encoded.columns.tolist(),
        "categorical_columns": list(categorical_cols),
    }
    with open(config.MODELS_DIR / "column_metadata.json", "w") as f:
        json.dump(column_metadata, f, indent=2)

    metrics: dict[str, dict[str, float]] = {}

    for target in config.TARGETS:
        print(f"\n=== {target} ===")
        y = labels_clean[target]

        X_train, X_test, y_train, y_test = train_test_split(
            df_encoded,
            y,
            random_state=config.SPLIT_RANDOM_STATE,
            test_size=config.TEST_SIZE,
            stratify=y,
        )

        features_sorted, importances_sorted = features.rank_features_by_importance(X_train, y_train)
        with open(config.MODELS_DIR / f"{target}_feature_importance.json", "w") as f:
            json.dump(
                {
                    "features": features_sorted.tolist(),
                    "importances": [float(x) for x in importances_sorted],
                },
                f,
                indent=2,
            )

        rf_pipeline = modeling.train_random_forest(
            X_train, y_train, config.RANDOM_FOREST_PARAMS[target]
        )
        auc_rf = modeling.evaluate_model(rf_pipeline, X_test, y_test)
        joblib.dump(rf_pipeline, config.MODELS_DIR / f"{target}_random_forest.joblib")
        print(f"Random Forest ROC AUC: {auc_rf:.4f}")

        lr_pipeline = modeling.train_logistic_regression(
            X_train, y_train, config.LOGISTIC_REGRESSION_PARAMS
        )
        auc_lr = modeling.evaluate_model(lr_pipeline, X_test, y_test)
        joblib.dump(lr_pipeline, config.MODELS_DIR / f"{target}_logistic_regression.joblib")
        print(f"Logistic Regression ROC AUC: {auc_lr:.4f}")

        metrics[target] = {"random_forest": auc_rf, "logistic_regression": auc_lr}

        test_data = X_test.copy()
        test_data[y.columns[0]] = y_test.values.ravel()
        test_data.to_csv(config.MODELS_DIR / f"{target}_test_data.csv", index=False)

    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Artifacts written to", config.MODELS_DIR)


if __name__ == "__main__":
    main()