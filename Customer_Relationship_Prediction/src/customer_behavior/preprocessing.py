"""Cleaning steps ported from the notebook: drop empty/high-missing columns, drop
incomplete rows, and drop high-cardinality categorical columns."""

import pandas as pd

from customer_behavior import config


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely NaN."""
    return df.dropna(how="all", axis=1)


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = config.COLUMN_NON_NULL_THRESHOLD
) -> pd.DataFrame:
    """Keep only columns with at least `threshold` fraction of non-null values."""
    min_non_null = int(len(df) * threshold)
    return df.dropna(axis=1, thresh=min_non_null)


def get_column_types(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Split columns into (numerical_columns, categorical_columns)."""
    numerical_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    return numerical_cols, categorical_cols


def drop_rows_with_missing(
    df: pd.DataFrame, labels: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Drop rows containing any NaN, dropping the same row indices from every label frame."""
    rows_to_drop = df[df.isnull().any(axis=1)].index
    df_clean = df.drop(rows_to_drop, axis=0)
    labels_clean = {
        target: label_df.drop(rows_to_drop, axis=0) for target, label_df in labels.items()
    }
    return df_clean, labels_clean


def drop_high_cardinality_columns(
    df: pd.DataFrame,
    categorical_cols: pd.Index,
    max_unique: int = config.HIGH_CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Index]:
    """Drop categorical columns with more than `max_unique` unique values."""
    cardinality = df[categorical_cols].nunique()
    columns_to_drop = cardinality[cardinality > max_unique].index
    df_updated = df.drop(columns=columns_to_drop)
    categorical_cols_updated = df_updated.select_dtypes(include=["object", "category"]).columns
    return df_updated, categorical_cols_updated


def missing_value_percentage(df: pd.DataFrame) -> pd.Series:
    """Percentage of missing values per column, descending."""
    return (df.isna().mean() * 100).sort_values(ascending=False)


def cardinality_by_column(df: pd.DataFrame, categorical_cols: pd.Index) -> pd.Series:
    """Number of unique values per categorical column, descending."""
    return df[categorical_cols].nunique().sort_values(ascending=False)


def clean_dataset(
    df: pd.DataFrame, labels: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Index]:
    """Run the full cleaning pipeline: empty columns -> high-missing columns ->
    incomplete rows -> high-cardinality categorical columns.

    Returns the cleaned dataframe, the row-aligned labels, and the surviving
    categorical column names.
    """
    df_step1 = drop_empty_columns(df)
    df_step2 = drop_high_missing_columns(df_step1)
    _, categorical_cols = get_column_types(df_step2)
    df_step3, labels_clean = drop_rows_with_missing(df_step2, labels)
    df_step4, categorical_cols_updated = drop_high_cardinality_columns(df_step3, categorical_cols)
    return df_step4, labels_clean, categorical_cols_updated