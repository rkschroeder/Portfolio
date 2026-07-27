"""Data access for the KDD Cup 2009 (Orange) dataset.

Two layers:
- `*_from_files`: Extract primitives that read the original tab/comma-delimited
  files under `data/raw/`. Only used by `elt.py`.
- `write_warehouse` / `read_warehouse`: the Load side of ELT — everything
  downstream (training, EDA) reads from the DuckDB warehouse these produce,
  not from the raw files directly.
"""

from pathlib import Path

import duckdb
import pandas as pd

from customer_behavior import config


def load_features_from_files(path: Path = config.FEATURES_FILE) -> pd.DataFrame:
    """Extract: load the tab-delimited raw feature matrix."""
    return pd.read_csv(path, delimiter="\t")


def load_label_from_file(path: Path, column_name: str) -> pd.DataFrame:
    """Extract: load a single target's label file as a one-column DataFrame."""
    return pd.read_csv(path, header=None, names=[column_name])


def load_labels_from_files(label_files: dict[str, Path] = config.LABEL_FILES) -> dict[str, pd.DataFrame]:
    """Extract: load all target label files, keyed by target name (e.g. 'churn')."""
    return {
        target: load_label_from_file(path, f"Label_{target.capitalize()}")
        for target, path in label_files.items()
    }


def warehouse_available(db_path: Path = config.WAREHOUSE_PATH) -> bool:
    """Whether `poetry run elt` has been run at least once."""
    return db_path.exists()


def write_warehouse(
    features_df: pd.DataFrame,
    labels: dict[str, pd.DataFrame],
    db_path: Path = config.WAREHOUSE_PATH,
) -> None:
    """Load: write the raw feature matrix and labels into DuckDB tables.

    The source files have no explicit customer identifier — rows across the
    feature file and each label file are aligned purely by position. A
    `customer_id` surrogate key (row position) is added so that alignment is
    an explicit, joinable column in the warehouse rather than an implicit
    assumption.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    features_with_id = features_df.reset_index(drop=True)
    features_with_id.insert(0, config.CUSTOMER_ID_COLUMN, range(len(features_with_id)))

    labels_wide = pd.DataFrame({config.CUSTOMER_ID_COLUMN: range(len(features_df))})
    for label_df in labels.values():
        label_df = label_df.reset_index(drop=True)
        labels_wide[label_df.columns[0]] = label_df.iloc[:, 0]

    con = duckdb.connect(str(db_path))
    try:
        con.register("features_with_id", features_with_id)
        con.execute(f"CREATE OR REPLACE TABLE {config.RAW_FEATURES_TABLE} AS SELECT * FROM features_with_id")
        con.unregister("features_with_id")

        con.register("labels_wide", labels_wide)
        con.execute(f"CREATE OR REPLACE TABLE {config.RAW_LABELS_TABLE} AS SELECT * FROM labels_wide")
        con.unregister("labels_wide")
    finally:
        con.close()


def read_warehouse(db_path: Path = config.WAREHOUSE_PATH) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read the raw feature matrix and labels back out of the DuckDB warehouse.

    Returns the same shape as the old file-based loader: a features DataFrame
    and a dict of target -> single-column label DataFrame, both indexed by
    `customer_id` so they stay row-aligned exactly as before.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        features_df = con.execute(f"SELECT * FROM {config.RAW_FEATURES_TABLE}").df()
        labels_wide = con.execute(f"SELECT * FROM {config.RAW_LABELS_TABLE}").df()
    finally:
        con.close()

    features_df = features_df.set_index(config.CUSTOMER_ID_COLUMN)
    labels_wide = labels_wide.set_index(config.CUSTOMER_ID_COLUMN)

    labels = {
        target: labels_wide[[f"Label_{target.capitalize()}"]] for target in config.TARGETS
    }
    return features_df, labels
