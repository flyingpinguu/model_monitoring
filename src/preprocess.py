"""Download, load and clean the NYC green taxi data.

This module holds everything that happens to the data *before* training,
so both the training script and the API can reuse the same logic.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# --- Settings -----------------------------------------------------------------
# In the notebook these were loose variables in cells. Here they are named
# constants at the top of the file, so there is one obvious place to change them.

ROOT_DIR = Path(__file__).resolve().parents[1]  # the project folder
DATA_DIR = ROOT_DIR / "data"

COLOR = "green"
YEAR = 2024
MONTH = 1

FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET = "duration"
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]


# --- Functions ----------------------------------------------------------------


def get_data_path(color: str = COLOR, year: int = YEAR, month: int = MONTH) -> Path:
    """Return where the parquet file for one month is stored locally."""
    return DATA_DIR / f"{color}_tripdata_{year}-{month:02d}.parquet"


def download_data(color: str = COLOR, year: int = YEAR, month: int = MONTH) -> Path:
    """Download one month of taxi data, unless it is already on disk."""
    data_path = get_data_path(color, year, month)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        url = (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"{color}_tripdata_{year}-{month:02d}.parquet"
        )
        print(f"Downloading {url}")
        urlretrieve(url, data_path)

    return data_path


def load_data(data_path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(data_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add the target, drop outliers, and keep only the model columns.

    Same logic as the notebook's preprocess(), but it works on a copy so the
    DataFrame passed in is never modified by accident.
    """
    df = df.copy()

    # Target: trip duration in minutes.
    df[TARGET] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Remove unrealistic trips.
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]

    # Location IDs are categories, not numbers (zone 200 is not "more" than zone 100).
    # As strings, DictVectorizer one-hot encodes them.
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)

    return df[FEATURES + [TARGET]]


def to_records(X: pd.DataFrame) -> list[dict]:
    """Turn feature rows into the list-of-dicts format DictVectorizer expects."""
    return X.to_dict(orient="records")


# --- Quick manual check -------------------------------------------------------
# This block only runs when you execute the file directly
# (`uv run python -m src.preprocess`), not when another file imports it.

if __name__ == "__main__":
    path = download_data()
    raw = load_data(path)
    clean = preprocess(raw)

    print(f"Raw rows:   {len(raw):>7}")
    print(f"Clean rows: {len(clean):>7}")
    print(clean.head())
    print(clean[TARGET].describe())
