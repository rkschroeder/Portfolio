import numpy as np
import pandas as pd
import pytest

from customer_behavior import features


@pytest.fixture
def categorical_df():
    # "a" appears 4/8 times, "b" 3/8, "c" 1/8
    return pd.DataFrame({"cat": ["a", "a", "a", "a", "b", "b", "b", "c"]})


def test_apply_frequency_encoding_maps_known_frequencies(categorical_df):
    encoded, encoder = features.apply_frequency_encoding(categorical_df, ["cat"])

    assert encoded.loc[categorical_df["cat"] == "a", "cat"].unique().tolist() == [4 / 8]
    assert encoded.loc[categorical_df["cat"] == "b", "cat"].unique().tolist() == [3 / 8]
    assert encoded.loc[categorical_df["cat"] == "c", "cat"].unique().tolist() == [1 / 8]
    assert encoded["cat"].dtype.kind == "f"


def test_apply_frequency_encoding_reuses_fitted_encoder(categorical_df):
    _, encoder = features.apply_frequency_encoding(categorical_df, ["cat"])

    new_data = pd.DataFrame({"cat": ["a", "b"]})
    encoded_new, returned_encoder = features.apply_frequency_encoding(new_data, ["cat"], encoder=encoder)

    assert returned_encoder is encoder
    assert encoded_new.loc[0, "cat"] == pytest.approx(4 / 8)
    assert encoded_new.loc[1, "cat"] == pytest.approx(3 / 8)


def test_rank_features_by_importance_returns_all_columns_sorted_desc():
    rng = np.random.default_rng(0)
    n = 200
    strong_signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (strong_signal > 0).astype(int)

    X = pd.DataFrame({"strong": strong_signal, "noise": noise})
    y_df = pd.DataFrame({"label": y})

    sorted_features, sorted_importances = features.rank_features_by_importance(
        X, y_df, n_estimators=50
    )

    assert set(sorted_features) == {"strong", "noise"}
    assert list(sorted_importances) == sorted(sorted_importances, reverse=True)
    assert sorted_features[0] == "strong"