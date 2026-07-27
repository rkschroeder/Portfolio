"""Extract-Load step: read the raw KDD Cup 2009 files and load them into the
DuckDB warehouse. Everything downstream (training, EDA) reads from there.

Run with: `poetry run elt`
"""

from customer_behavior import config, data


def main() -> None:
    print("Extracting raw files...")
    features_df = data.load_features_from_files()
    labels = data.load_labels_from_files()
    print(f"  features: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    for target, label_df in labels.items():
        print(f"  {target}: {label_df.shape[0]:,} rows")

    print(f"\nLoading into {config.WAREHOUSE_PATH} ...")
    data.write_warehouse(features_df, labels, config.WAREHOUSE_PATH)

    print(f"Done. Tables written: {config.RAW_FEATURES_TABLE}, {config.RAW_LABELS_TABLE}")


if __name__ == "__main__":
    main()