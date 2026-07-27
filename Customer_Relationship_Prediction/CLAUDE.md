# Customer Relationship Prediction — notes for Claude

- The dataset's features are **anonymized** (`Var1`...`Var230`, no real-world
  labels — Orange redacted what each one measures). This isn't just a data
  quirk, it drives UI decisions: the Predict page can't show meaningful field
  labels or explain what a "sensible" value is, so it deliberately favors
  loading real customer records (via "Load a real customer") over asking
  users to hand-type values for fields like `Var6`. Keep this in mind for any
  future UI work on the Predict page — err on the side of more explanatory
  copy/tooltips, not less, since users have already found the anonymized
  fields and the ROC-AUC-vs-probability distinction confusing without it.
  See the "Challenges with this dataset" section in `README.md`.
- `src/customer_behavior/training.py` fits the `CountFrequencyEncoder` on the
  full cleaned dataset before splitting into train/test per target. This is a
  mild train/test leak, but it's **intentional** — it mirrors the original
  notebook exactly so ROC AUC scores stay comparable to what the notebook
  reported. Don't "fix" this without checking with the user first; if asked to
  fix it, the change is: fit the encoder per-target on `X_train` only inside
  the per-target loop, then `encoder.transform` on `X_test`.
- Poetry's venv on this machine needs `C:\Users\OWNER\miniconda3` prepended to
  `PATH` before running `poetry env use` / `poetry install` — otherwise Poetry
  shells out to the broken Microsoft Store Python alias
  (`WindowsApps\python.exe`) and fails with exit code 9009. The project venv
  is already pinned to Python 3.13 (conda's `python.exe`); this only matters
  if the venv needs to be recreated.
- **ELT layer**: `poetry run elt` extracts `data/raw/*`
  into a DuckDB warehouse (`data/warehouse.duckdb`, gitignored, regenerated on
  demand); `poetry run train` and the EDA page read from that warehouse
  instead of the raw files directly. 
  - **DuckDB** — embedded/file-based, zero infrastructure, no
    Docker needed to run it.
  - **Transform stays in pandas** — `preprocessing.py`/`features.py`
  - **Docker and Airflow are explicitly deferred**, not in scope yet. 