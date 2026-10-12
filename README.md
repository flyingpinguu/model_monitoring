# Green Taxi Trip-Duration: MLOps Project

A small, end-to-end MLOps workflow around a model that predicts how long an NYC green taxi trip takes.
The model is deliberately simple (linear regression on three features). The point is the workflow around it:

```text
download data ──► train ──► serve via API ──► monitor (Prometheus, Grafana, Evidently)
   (DVC)        (DVC + MLflow)   (FastAPI)
```

> **Scope:** `trip_distance` is the taximeter distance measured *during* the trip, so the model replays
> completed trips. It is not a forecast made at pickup time.

## Quick start

All commands run from the project folder.

```bash
uv sync                                          # 1. install dependencies into .venv
uv run dvc repro                                 # 2. download data + train the model
docker compose up -d --build                     # 3. start API, Prometheus, Grafana
#                                                  4. send requests (see "Using the API")
uv run python -m src.generate_report             # 5. build the Evidently drift report
open monitoring/reports/drift_report.html
docker compose down                              # stop everything
```

| Service | URL |
|---|---|
| API docs (try requests in the browser) | <http://127.0.0.1:8000/docs> |
| Grafana dashboard (`admin` / `admin`) | <http://127.0.0.1:3000/d/green-taxi-api> |
| Prometheus targets | <http://127.0.0.1:9090/targets> |
| MLflow UI (start with the command below) | <http://127.0.0.1:5001> |

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Port 5000 is taken by macOS AirPlay, hence 5001.

## Using the API

`POST /predict` with one trip:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 74, "DOLocationID": 243, "trip_distance": 5.14}'
```

```json
{"predicted_duration_minutes": 20.94, "model_version": "5"}
```

| Field | Meaning | Allowed values |
|---|---|---|
| `PULocationID` | Pickup taxi zone ([zone list](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv)) | 1–265 |
| `DOLocationID` | Dropoff taxi zone | 1–265 |
| `trip_distance` | Distance in miles | > 0 and ≤ 100 |

Other endpoints: `GET /health` (is it running?), `GET /model-info` (which model is served),
`GET /metrics` (numbers for Prometheus).

## What each file does

### Pipeline code: `src/`

| File | Role |
|---|---|
| `preprocess.py` | Shared data logic: download, load, clean (`preprocess`), and `prepare_features`, which training **and** the API use, so both see identical inputs. Settings (month, features) live at the top. |
| `download_data.py` | DVC stage 1: downloads the January 2024 data into `data/`. |
| `train.py` | DVC stage 2: splits, trains, evaluates (RMSE), logs to MLflow, registers the model with alias `production`, saves `models/model.joblib`. |
| `api.py` | FastAPI app. Loads the model once at startup, validates requests, predicts, updates Prometheus metrics and appends each request to `monitoring/requests.csv`. |
| `generate_report.py` | Evidently report comparing logged requests (current) with the training data (reference). |

### Data and experiment tracking

| File | Role |
|---|---|
| `dvc.yaml` | The pipeline recipe: stages, their commands, inputs (`deps`) and outputs (`outs`). |
| `dvc.lock` | Auto-generated fingerprints (md5) of every input and output. This is the data version; commit it. |
| `.dvc/` | DVC's own configuration. |
| `mlflow.db`, `mlruns/` | Local MLflow store with all training runs and registered models (not in Git). |
| `reports/train_metrics.json` | Latest RMSE, shown by `uv run dvc metrics show`. |

### Serving and monitoring

| File | Role |
|---|---|
| `Dockerfile` | Builds the API container image. |
| `docker-compose.yaml` | Starts API, Prometheus and Grafana together. Mounts `models/` (read-only) and `monitoring/` into the API. |
| `monitoring/prometheus.yaml` | Tells Prometheus to scrape the API's `/metrics` every 5 s. |
| `monitoring/grafana/` | Data source and dashboard definitions, loaded automatically when Grafana starts. |
| `monitoring/requests.csv` | Request log written by the API (generated, not in Git). |
| `monitoring/reports/` | Evidently HTML reports (generated, not in Git). |

### Generated or local only (not in Git)

`data/` (downloaded parquet files), `models/` (trained model + metadata, tracked by DVC), `.venv/`, `.env`.

### From the original template

| File | Role |
|---|---|
| `01-train-ml-model.ipynb` | The original notebook the `src/` code was refactored from. |
| `project-description.md` | The project brief. |
| `solution/` | Reference implementation of the same workflow on the Iris dataset. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies and the exact locked versions. |

## Monitoring at a glance

- **Prometheus/Grafana (system and model behaviour over time):** request rate, latency, average predicted
  duration, average input distance, prediction distribution.
- **Evidently (data drift):** compares the logged requests with the training data, column by column.
  Needs a reasonable number of requests (dozens at least) to say anything meaningful. The zone-ID drift
  threshold is raised to 0.2 because ~260 categories cannot match the training mix exactly with a small
  batch of requests (avoids false alarms).
