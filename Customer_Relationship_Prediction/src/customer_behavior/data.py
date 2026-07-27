"""Loading the raw KDD Cup 2009 (Orange) feature and label files."""

from pathlib import Path

import pandas as pd

from customer_behavior import config


def load_features(path: Path = config.FEATURES_FILE) -> pd.DataFrame:
    """Load the tab-delimited raw feature matrix."""
    return pd.read_csv(path, delimiter="\t")


def load_label(path: Path, column_name: str) -> pd.DataFrame:
    """Load a single target's label file as a one-column DataFrame."""
    return pd.read_csv(path, header=None, names=[column_name])


def load_labels(label_files: dict[str, Path] = config.LABEL_FILES) -> dict[str, pd.DataFrame]:
    """Load all target label files, keyed by target name (e.g. 'churn')."""
    return {
        target: load_label(path, f"Label_{target.capitalize()}")
        for target, path in label_files.items()
    }


def load_raw_dataset(
    features_path: Path = config.FEATURES_FILE,
    label_files: dict[str, Path] = config.LABEL_FILES,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load the raw feature matrix together with all target label files."""
    return load_features(features_path), load_labels(label_files)