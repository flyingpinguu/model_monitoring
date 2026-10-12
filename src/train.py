"""Train the taxi trip-duration model and track the run with MLflow.

Run it from the project folder with:

    uv run python -m src.train
"""

import getpass
import json
import logging
import os

import joblib
import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from src.preprocess import (
    COLOR,
    FEATURES,
    MONTH,
    ROOT_DIR,
    TARGET,
    YEAR,
    download_data,
    load_data,
    preprocess,
    to_records,
)

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.ERROR)

# --- Settings -----------------------------------------------------------------

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = ROOT_DIR / "reports" / "train_metrics.json"

EXPERIMENT_NAME = "green-taxi-monitoring"
REGISTERED_MODEL_NAME = "green-taxi-ride-duration"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# --- Functions ----------------------------------------------------------------


def setup_mlflow() -> None:
    """Point MLflow at the tracking store and select the experiment."""
    load_dotenv(ROOT_DIR / ".env")
    # Default to an absolute path so it works no matter which folder you run from.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    # Creates the experiment on the first run, reuses it afterwards.
    mlflow.set_experiment(EXPERIMENT_NAME)


def split_data(df):
    """Split the cleaned data into train and test sets (same split as the notebook)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(X_train, y_train) -> Pipeline:
    """Fit DictVectorizer + LinearRegression as one pipeline."""
    pipeline = make_pipeline(DictVectorizer(), LinearRegression())
    pipeline.fit(to_records(X_train), y_train)
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> float:
    """Return the RMSE on the test set, in minutes."""
    y_pred = pipeline.predict(to_records(X_test))
    return root_mean_squared_error(y_test, y_pred)


def save_model(pipeline: Pipeline, metadata: dict) -> None:
    """Save the model to models/ so the API can load it without MLflow."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def save_metrics(metrics: dict) -> None:
    """Write the metrics to a small JSON file that `dvc metrics show` can read."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def register_model(model_uri: str, run_id: str) -> str:
    """Register the model in MLflow and point the 'production' alias at it."""
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            tags={"framework": "scikit-learn"},
            description="Model to predict ride duration in NYC",
        )
    except MlflowException as exc:
        # Reuse the model only if it already exists; re-raise any other MLflow error.
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise

    version = client.create_model_version(
        name=REGISTERED_MODEL_NAME, source=model_uri, run_id=run_id
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME, version=version.version, alias="production"
    )
    return version.version


def main() -> None:
    """Run the full training pipeline: data -> model -> metrics -> MLflow."""
    # 1. Data
    data_path = download_data()
    df = preprocess(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # 2. MLflow run: everything inside the `with` block is logged to one run.
    setup_mlflow()
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "developer": getpass.getuser(),
                "dataset": f"{COLOR}-taxi",
                "year": YEAR,
                "month": MONTH,
                "features": FEATURES,
                "target": TARGET,
            }
        )
        mlflow.log_params(
            {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "train_rows": len(X_train)}
        )

        # 3. Train and evaluate
        pipeline = train_model(X_train, y_train)
        rmse = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metric("rmse", rmse)

        # 4. Store the model in MLflow and in the registry
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        version = register_model(model_info.model_uri, run.info.run_id)

    # 5. Save a local copy for the API
    save_model(
        pipeline,
        {
            "run_id": run.info.run_id,
            "model_version": version,
            "rmse": rmse,
            "features": FEATURES,
            "data_file": data_path.name,
        },
    )
    save_metrics({"rmse": rmse, "train_rows": len(X_train), "test_rows": len(X_test)})

    print(f"RMSE: {rmse:.3f} minutes")
    print(f"MLflow run: {run.info.run_id}  (registered as version {version}, alias 'production')")
    print(f"Saved model to {MODEL_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
