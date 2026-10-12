"""Prediction API for the taxi trip-duration model.

Start it from the project folder with:

    uv run uvicorn src.api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import csv
import json
import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.preprocess import ROOT_DIR, prepare_features, to_records
from src.train import METADATA_PATH, MODEL_PATH

REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"

# --- Load the model once, when the API starts --------------------------------
# Loading on every request would be slow, so we do it a single time here.


def load_model() -> tuple[Any, dict]:
    """Load the trained pipeline and its metadata from models/."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run `uv run dvc repro` first."
        )
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


MODEL, MODEL_METADATA = load_model()


# --- Prometheus metrics -------------------------------------------------------
# Each metric is a number the API keeps in memory. Prometheus reads them all
# from /metrics every few seconds and stores the history.
#   Counter   = only goes up (e.g. number of requests)
#   Histogram = counts values in buckets, so Grafana can show averages and
#               distributions (e.g. how long requests take)

PREDICTION_REQUESTS = Counter(
    "api_prediction_requests_total",
    "Number of prediction requests handled by the API.",
)
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds",
    "Time spent handling a prediction request.",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
PREDICTED_DURATION = Histogram(
    "model_predicted_duration_minutes",
    "Trip durations predicted by the model.",
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 60],
)
INPUT_TRIP_DISTANCE = Histogram(
    "model_input_trip_distance_miles",
    "Trip distances sent to the model.",
    buckets=[0.5, 1, 2, 3, 5, 7, 10, 15, 20, 50, 100],
)


# --- Request log for Evidently ------------------------------------------------
# Prometheus keeps numbers, not the individual requests. Evidently needs the
# actual inputs and predictions, so we also append every request to a CSV file.

REQUEST_LOG_LOCK = threading.Lock()  # stops two requests writing at the same moment


def log_request(features: dict, prediction: float) -> None:
    """Append one request and its prediction to monitoring/requests.csv."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **features,
        "prediction": prediction,
    }
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG_LOCK:
        write_header = not REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# --- Request and response schemas ---------------------------------------------
# Pydantic checks every incoming request against these classes. Wrong types or
# out-of-range values are rejected with a 422 error before reaching the model.


class TripRequest(BaseModel):
    """One completed taxi trip."""

    PULocationID: int = Field(..., ge=1, le=265, examples=[74], description="Pickup zone ID")
    DOLocationID: int = Field(..., ge=1, le=265, examples=[243], description="Dropoff zone ID")
    trip_distance: float = Field(
        ...,
        gt=0,
        le=100,
        examples=[5.14],
        description="Taximeter distance in miles, measured during the trip",
    )


class PredictionResponse(BaseModel):
    """The model's answer for one trip."""

    predicted_duration_minutes: float
    model_version: str


# --- The app and its endpoints ------------------------------------------------

app = FastAPI(
    title="Green Taxi Trip-Duration API",
    description=(
        "Predicts NYC green taxi trip duration. Uses the measured trip distance, "
        "so it replays completed trips; it is not a forecast made at pickup time."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check: answers 'ok' if the API is running."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    """Show which model is being served (MLflow run, version, RMSE)."""
    return MODEL_METADATA


@app.get("/metrics")
def metrics() -> Response:
    """Expose all metrics in the text format Prometheus understands."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(trip: TripRequest) -> PredictionResponse:
    """Predict the duration of one trip in minutes."""
    start_time = perf_counter()

    # Turn the request into a one-row DataFrame and reuse the training preprocessing.
    features = prepare_features(pd.DataFrame([trip.model_dump()]))
    prediction = round(float(MODEL.predict(to_records(features))[0]), 2)

    # Monitoring: update the Prometheus metrics and write the request log.
    log_request(trip.model_dump(), prediction)
    PREDICTION_REQUESTS.inc()
    PREDICTED_DURATION.observe(prediction)
    INPUT_TRIP_DISTANCE.observe(trip.trip_distance)
    PREDICTION_LATENCY.observe(perf_counter() - start_time)

    return PredictionResponse(
        predicted_duration_minutes=prediction,
        model_version=str(MODEL_METADATA["model_version"]),
    )
