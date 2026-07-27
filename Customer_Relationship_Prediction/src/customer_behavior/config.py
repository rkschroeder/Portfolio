"""Paths and hyperparameters shared across the pipeline, mirroring the original notebook's choices."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_FILE = DATA_DIR / "orange_small_train.data"
LABEL_FILES = {
    "churn": DATA_DIR / "orange_small_train_churn.labels",
    "appetency": DATA_DIR / "orange_small_train_appetency.labels",
    "upselling": DATA_DIR / "orange_small_train_upselling.labels",
}

# DuckDB warehouse populated by `poetry run elt` (Extract raw files -> Load into tables).
# Everything downstream (training, EDA) reads from here, not from data/raw/ directly.
WAREHOUSE_PATH = PROJECT_ROOT / "data" / "warehouse.duckdb"
RAW_FEATURES_TABLE = "raw_features"
RAW_LABELS_TABLE = "raw_labels"
CUSTOMER_ID_COLUMN = "customer_id"

TARGETS = ["churn", "appetency", "upselling"]

# Fraction of non-null values a column must have to survive the missingness filter.
COLUMN_NON_NULL_THRESHOLD = 0.8

# Categorical columns with more unique values than this are dropped (too noisy/high-cardinality).
HIGH_CARDINALITY_THRESHOLD = 1500

# train_test_split uses a different seed than the models, matching the notebook.
SPLIT_RANDOM_STATE = 1
TEST_SIZE = 0.2

MODEL_RANDOM_STATE = 42
FEATURE_SELECTION_N_ESTIMATORS = 200

RANDOM_FOREST_PARAMS = {
    "churn": {"n_estimators": 300, "max_features": 40, "min_samples_leaf": 2},
    "appetency": {"n_estimators": 300, "max_features": 16, "min_samples_leaf": 5},
    "upselling": {"n_estimators": 300, "max_features": 18, "min_samples_leaf": 5},
}

# penalty="l2" is scikit-learn's default; omitted to avoid its deprecated explicit-penalty warning.
LOGISTIC_REGRESSION_PARAMS = {"C": 0.8, "max_iter": 100}