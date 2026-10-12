"""Build an Evidently drift report: API requests vs. the training data.

Run it from the project folder (after sending some requests) with:

    uv run python -m src.generate_report            # all logged requests
    uv run python -m src.generate_report --last 50  # only the most recent 50
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.api import REQUEST_LOG_PATH
from src.preprocess import (
    CATEGORICAL_FEATURES,
    ROOT_DIR,
    download_data,
    load_data,
    preprocess,
    to_records,
)
from src.train import MODEL_PATH

REPORT_PATH = ROOT_DIR / "monitoring" / "reports" / "drift_report.html"

# Tell Evidently which columns are categories and which are numbers, so it
# picks a suitable statistical test for each one.
DATA_DEFINITION = DataDefinition(
    categorical_columns=CATEGORICAL_FEATURES,
    numerical_columns=["trip_distance", "prediction"],
)
COLUMNS = CATEGORICAL_FEATURES + ["trip_distance", "prediction"]

# The zone IDs have ~260 possible values, so a small batch of requests can never
# match the training mix exactly. With Evidently's default threshold (0.1) they
# are flagged as drifted even for perfectly normal traffic. 0.2 still catches a
# real change in where trips start and end.
DRIFT_PRESET = DataDriftPreset(cat_threshold=0.2)


def load_reference() -> pd.DataFrame:
    """Training data (January) plus the model's predictions on it."""
    df = preprocess(load_data(download_data()))
    model = joblib.load(MODEL_PATH)
    df["prediction"] = model.predict(to_records(df.drop(columns="duration")))
    return df[COLUMNS]


def load_current(last: int | None) -> pd.DataFrame:
    """Requests the API has logged, optionally only the most recent ones."""
    if not REQUEST_LOG_PATH.exists():
        raise FileNotFoundError(
            "No request log yet. Start the API and send some requests to /predict first."
        )
    df = pd.read_csv(REQUEST_LOG_PATH)
    if last:
        df = df.tail(last)
    # Same types as in training: zone IDs are categories (strings).
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return df[COLUMNS]


def generate_report(last: int | None) -> Path:
    reference = load_reference()
    current = load_current(last)
    print(f"Comparing {len(current)} requests with {len(reference)} training rows")

    report = Report(metrics=[DRIFT_PRESET, DataSummaryPreset()])
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference, data_definition=DATA_DEFINITION),
        current_data=Dataset.from_pandas(current, data_definition=DATA_DEFINITION),
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    return REPORT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=int, help="only use the most recent N requests")
    args = parser.parse_args()

    path = generate_report(args.last)
    print(f"Report written to {path.relative_to(ROOT_DIR)}  (open it in your browser)")
