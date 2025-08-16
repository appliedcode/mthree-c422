# Capstone Project with the Iris Dataset (Classification)

**Main takeaway:**
Implement a complete MLOps pipeline for multiclass classification on the Iris dataset, covering data versioning, experiment tracking, API deployment, CI/CD automation, and monitoring—ideal for hands-on learning on a local system.

***

## Project Overview \& Architecture

- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Task:** Multiclass classification
- **Tools:**
    - Git + GitHub
    - DVC (optional)
    - MLflow (tracking \& model registry)
    - FastAPI + Pydantic
    - Docker
    - GitHub Actions
    - Python logging + SQLite
    - Prometheus + Grafana (optional bonus)

***

## Architecture Diagram

1. **Data \& Code Versioning:** Git for code; DVC to track `iris.csv`
2. **Experiment Tracking \& Registry:** MLflow local server
3. **API Service:** FastAPI with Pydantic validation
4. **Containerization:** Docker image
5. **CI/CD:** GitHub Actions workflows
6. **Logging \& Monitoring:** Python logging → file/SQLite; optional Prometheus/Grafana

***

## Part 1: Repository \& Data Versioning

- **Repository Structure:**

```
.
├── data/
│   └── iris.csv       # exported via pandas
├── src/
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   └── schemas.py
├── models/            # model binaries
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/ci.yml
└── README.md
```

- **Data Preparation:**
    - In `data_prep.py`, load via `sklearn.datasets.load_iris()`, split into train/test, save as `data/iris.csv`.
- **DVC (optional):**
    - Initialize DVC and track `data/iris.csv` with `dvc add data/iris.csv`, keeping large artifacts out of Git.

***

## Part 2: Model Development \& Experiment Tracking

- **Models:** LogisticRegression and RandomForestClassifier.
- **MLflow Integration:**
    - Log hyperparameters (e.g., `C`, `n_estimators`), metrics (accuracy, precision, recall, F1), and feature importance.
    - Compare runs in the MLflow UI and register the best-performing model in the Model Registry.
- **Reproducibility:** Commit all training code, `mlruns/` (ignored in Git, configured locally), and parameter settings.

***

## Part 3: API \& Docker Packaging

- **FastAPI Service:**
    - `GET /health` → returns status `"OK"`.
    - `POST /predict` → accepts `IrisFeatures` JSON (sepal/petal lengths and widths); returns predicted class and probabilities.
- **Pydantic Schema:**

```python
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
```

***

## Part 4: CI/CD with GitHub Actions

- **Workflow (`ci.yml`):**

1. **Lint \& Test:** Run `flake8` and `pytest` on `src/` and `api/`.
2. **Docker Build:** Build Docker image `iris-api`.
3. **Docker Push:** Push to Docker Hub (`dockerhub_user/iris-api:latest`) using stored secrets.
4. **Local Deploy (optional self-hosted runner):** Stop existing container, pull new image, and start service locally.
- **Security:** Store `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as GitHub Secrets.

***

## Part 5: Logging \& Monitoring

- **Logging:**
    - Use Python’s `logging` to record each request’s payload, timestamp, predicted class, probability, and latency.
    - Persist logs to a daily rotating file (`logs/iris_api.log`) and to a SQLite table (`predictions`).
- **Monitoring (Optional Bonus):**
    - Add `/metrics` endpoint exporting Prometheus metrics: request count, error count, latency histogram, model version.
    - Launch Prometheus and Grafana locally via Docker Compose; create dashboard panels for:
        - Total requests over time
        - P95 latency
        - Error rate
        - Prediction distribution by class

***

## Part 6: Summary \& Demo 

- **1-Page Summary PDF:**
    - Diagram illustrating the pipeline
    - Tool choices and rationale
    - CI/CD workflow snapshot
    - MLflow run comparison screenshot
    - Monitoring dashboard snapshots
- **5-Minute Video Walkthrough:**
    - Show data preparation and versioning
    - Train models and demonstrate MLflow UI
    - Build and run the Dockerized API, perform sample predictions
    - Display logs in SQLite and Grafana dashboard

***

## Bonus Features

1. **Enhanced Input Validation:**
    - Constrain numeric ranges in Pydantic schema; custom error messages.
2. **Prometheus \& Grafana Dashboard:**
    - Pre-built JSON dashboard with four panels (requests, latency, errors, class distribution).
3. **Automated Retraining Trigger:**
    - Upon new data commit via DVC, run `dvc repro && python src/train.py`, register a new model, rebuild image, and redeploy using an additional GitHub Actions job.

***

## Deliverables

- **GitHub Repository:** Includes all source code, DVC files (if used), MLflow scripts, GitHub Actions workflows, Docker configuration, and detailed `README.md` with instructions.
- **Docker Hub Image:** Publicly accessible `dockerhub_user/iris-api:latest`.
- **Summary PDF:** Architecture and pipeline overview.
