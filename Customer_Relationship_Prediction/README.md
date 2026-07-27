# Customer Relationship Prediction

A Streamlit dashboard that predicts three customer behaviors for a telecom
operator (Orange) using the [KDD Cup 2009](https://kdd.org/kdd-cup/view/kdd-cup-2009/Intro)
dataset:

- **Churn** — will this customer leave for a competitor?
- **Appetency** — will this customer buy additional products or services?
- **Upselling** — will this customer buy an upgrade or add-on?

The project started as a single exploratory notebook (kept in `notebooks/` for
history) and has since been refactored into installable modules, a Poetry
project, and a multipage Streamlit app.

## Project structure

```
app.py                        # Streamlit entry point (Overview page)
pages/                        # Streamlit multipage app
  1_Exploratory_Data_Analysis.py
  2_Model_Performance.py
  3_Predict_Customer_Behavior.py
src/customer_behavior/        # Installable package with the core pipeline
  config.py                   # paths, thresholds, hyperparameters
  data.py                     # raw data loading
  preprocessing.py            # column/row cleaning
  features.py                 # frequency encoding, RF feature ranking
  modeling.py                 # Random Forest / Logistic Regression training
  training.py                 # orchestrates the full pipeline (CLI: `train`)
  app_support.py              # cached loaders used by the Streamlit pages
data/raw/                     # the original KDD Cup 2009 data files
models/                       # generated at training time (gitignored)
notebooks/                    # the original exploratory notebook
tests/                        # pytest unit tests for the pipeline modules
```

## Setup

Requires [Poetry](https://python-poetry.org/) and Python 3.10–3.13.

```bash
poetry install
```

## Train the models

The Streamlit app reads pre-trained artifacts from `models/` rather than
training on every launch. Generate them once (takes a few minutes — six models
across three targets):

```bash
poetry run train
```

This writes trained Random Forest / Logistic Regression pipelines, the fitted
frequency encoder, per-target feature importances, held-out test sets, and a
`metrics.json` summary to `models/`.

## Run the app

```bash
poetry run streamlit run app.py
```

## Run the tests

```bash
poetry run pytest
```

## Challenges with this dataset

- **Anonymized features**: columns are named `Var1`...`Var230` with no
  real-world labels — Orange redacted what each one actually measures. This
  rules out domain-informed feature engineering (there's no way to know
  "this is call volume" vs "this is contract length") and shapes the Predict
  page's UI: it can't show meaningful field labels or sensible defaults, so it
  leans on loading real customer records instead of asking you to guess
  plausible values for `Var6`.
- **Heavy missingness**: many of the 230 raw columns are >50% NaN; the
  cleaning pipeline drops columns below an 80% non-null threshold and then
  drops any remaining incomplete rows, which shrinks 50,000 rows to ~36,700.
- **Severe class imbalance**: all three targets (churn, appetency, upselling)
  have a small minority positive class, addressed with undersampling.
- **Mixed, high-cardinality categoricals**: some categorical columns have
  over 15,000 unique values; those are dropped outright, and the rest are
  frequency-encoded rather than one-hot encoded to avoid an explosion of
  features.

## Approach

- **Missing data**: columns that are empty or >20% missing are dropped; rows
  with any remaining missing value are dropped.
- **Categorical features**: frequency encoding (each category replaced by how
  often it appears), chosen over one-hot (too many categories) or target
  encoding (biased by the same class imbalance we're trying to predict).
- **Class imbalance**: random undersampling of the majority class before
  fitting.
- **Models**: Random Forest and Logistic Regression, compared by ROC AUC.
  Random Forest outperforms Logistic Regression on appetency and upselling;
  the two are comparable on churn.