import numpy as np
import pandas as pd

from customer_behavior import preprocessing


def test_drop_empty_columns_removes_all_nan_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "empty": [np.nan, np.nan, np.nan], "b": [4, np.nan, 6]})
    result = preprocessing.drop_empty_columns(df)
    assert "empty" not in result.columns
    assert list(result.columns) == ["a", "b"]


def test_drop_high_missing_columns_keeps_columns_above_threshold():
    df = pd.DataFrame(
        {
            "mostly_full": [1, 2, 3, 4, np.nan],  # 80% non-null
            "mostly_empty": [1, np.nan, np.nan, np.nan, np.nan],  # 20% non-null
        }
    )
    result = preprocessing.drop_high_missing_columns(df, threshold=0.8)
    assert "mostly_full" in result.columns
    assert "mostly_empty" not in result.columns


def test_get_column_types_splits_numerical_and_categorical():
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["x", "y", "z"]})
    numerical_cols, categorical_cols = preprocessing.get_column_types(df)
    assert list(numerical_cols) == ["num"]
    assert list(categorical_cols) == ["cat"]


def test_drop_rows_with_missing_aligns_labels():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4]})
    labels = {"churn": pd.DataFrame({"Label_Churn": [1, -1, 1, -1]})}

    df_clean, labels_clean = preprocessing.drop_rows_with_missing(df, labels)

    assert list(df_clean.index) == [0, 2, 3]
    assert list(labels_clean["churn"].index) == [0, 2, 3]
    assert len(df_clean) == len(labels_clean["churn"])


def test_drop_high_cardinality_columns_drops_above_max_unique():
    df = pd.DataFrame(
        {
            "low_card": ["a", "b", "a", "b"],
            "high_card": ["w", "x", "y", "z"],
        }
    )
    categorical_cols = df.columns
    df_updated, categorical_cols_updated = preprocessing.drop_high_cardinality_columns(
        df, categorical_cols, max_unique=2
    )
    assert "high_card" not in df_updated.columns
    assert "low_card" in df_updated.columns
    assert list(categorical_cols_updated) == ["low_card"]


def test_missing_value_percentage_sorted_descending():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, np.nan]})
    result = preprocessing.missing_value_percentage(df)
    assert result.iloc[0] >= result.iloc[1]
    assert result["a"] == 75.0
    assert result["b"] == 25.0


def test_cardinality_by_column_sorted_descending():
    df = pd.DataFrame({"a": ["x", "y", "z", "w"], "b": ["x", "x", "y", "y"]})
    result = preprocessing.cardinality_by_column(df, df.columns)
    assert result.index[0] == "a"
    assert result["a"] == 4
    assert result["b"] == 2


def test_clean_dataset_end_to_end():
    # cardinality threshold defaults to 1500, so it isn't exercised here (see the
    # dedicated drop_high_cardinality_columns test above) - this checks the rest
    # of the pipeline: empty columns, high-missing columns, and incomplete rows.
    df = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan, np.nan],
            "keep_num": [1, 2, 3, 4],
            "keep_cat": ["a", "b", "a", "b"],
            "has_missing": [1, np.nan, 3, 4],
        }
    )
    labels = {"churn": pd.DataFrame({"Label_Churn": [1, -1, 1, -1]})}

    df_clean, labels_clean, categorical_cols = preprocessing.clean_dataset(df, labels)

    assert "all_nan" not in df_clean.columns
    assert not df_clean.isnull().values.any()
    assert len(df_clean) == len(labels_clean["churn"])
    assert list(categorical_cols) == ["keep_cat"]
