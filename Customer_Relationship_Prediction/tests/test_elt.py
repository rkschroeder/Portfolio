import pandas as pd

from customer_behavior import data


def test_warehouse_available_false_when_missing(tmp_path):
    assert data.warehouse_available(tmp_path / "does_not_exist.duckdb") is False


def test_write_then_read_warehouse_round_trip(tmp_path):
    db_path = tmp_path / "warehouse.duckdb"

    features_df = pd.DataFrame({"Var1": [1.0, 2.0, 3.0], "Var2": ["a", "b", "c"]})
    labels = {
        "churn": pd.DataFrame({"Label_Churn": [-1, 1, -1]}),
        "appetency": pd.DataFrame({"Label_Appetency": [-1, -1, 1]}),
        "upselling": pd.DataFrame({"Label_Upselling": [1, -1, -1]}),
    }

    data.write_warehouse(features_df, labels, db_path)

    assert data.warehouse_available(db_path) is True

    read_features, read_labels = data.read_warehouse(db_path)

    # customer_id becomes the index, row order/alignment preserved.
    assert list(read_features.index) == [0, 1, 2]
    assert read_features["Var1"].tolist() == [1.0, 2.0, 3.0]
    assert read_features["Var2"].tolist() == ["a", "b", "c"]

    assert set(read_labels.keys()) == {"churn", "appetency", "upselling"}
    assert read_labels["churn"]["Label_Churn"].tolist() == [-1, 1, -1]
    assert read_labels["appetency"]["Label_Appetency"].tolist() == [-1, -1, 1]
    assert read_labels["upselling"]["Label_Upselling"].tolist() == [1, -1, -1]

    # Features and labels share the same index, so downstream row-dropping-by-index
    # (preprocessing.drop_rows_with_missing) stays valid.
    assert list(read_features.index) == list(read_labels["churn"].index)


def test_write_warehouse_preserves_row_order_as_customer_id(tmp_path):
    db_path = tmp_path / "warehouse.duckdb"
    n = 50
    features_df = pd.DataFrame({"Var1": range(n)})
    labels = {
        "churn": pd.DataFrame({"Label_Churn": [1 if i % 10 == 0 else -1 for i in range(n)]}),
        "appetency": pd.DataFrame({"Label_Appetency": [-1] * n}),
        "upselling": pd.DataFrame({"Label_Upselling": [-1] * n}),
    }

    data.write_warehouse(features_df, labels, db_path)
    read_features, read_labels = data.read_warehouse(db_path)

    assert read_features["Var1"].tolist() == list(range(n))
    assert list(read_features.index) == list(range(n))
    assert read_labels["churn"]["Label_Churn"].tolist() == labels["churn"]["Label_Churn"].tolist()