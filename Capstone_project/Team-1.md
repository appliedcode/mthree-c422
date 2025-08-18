# Capstone Project with the Wine Quality Dataset (Classification)

**Main takeaway:**
Implement a complete end-to-end MLOps pipeline for multiclass classification (predicting wine quality score) using the Wine Quality dataset, incorporating data versioning, experiment tracking, API deployment, CI/CD, and monitoring—perfect for hands-on learning on a local system.

***

## Project Overview \& Architecture

- **Dataset:** Wine Quality (Red, 1,599 samples; White, 4,898 samples; 11 physicochemical features, target: quality score 0–10)
- **Task:** Multiclass classification (quality groupings)
*(Optionally, treat as regression for bonus)*
- **Tools:**
    - Git + GitHub
    - Data Version Control (DVC, optional)
    - MLflow (tracking \& model registry)
    - FastAPI + Pydantic
    - Docker
    - GitHub Actions
    - Python logging + SQLite
    - Prometheus + Grafana (optional bonus)

***

## Architecture Diagram

1. **Data \& Code Versioning:** Git for source code; DVC to track `winequality-red.csv`/`winequality-white.csv`
2. **Experiment Tracking \& Registry:** MLflow local server
3. **API Service:** FastAPI server with Pydantic validation
4. **Containerization:** Docker image
5. **CI/CD:** GitHub Actions workflows
6. **Logging \& Monitoring:** Python logging to file/SQLite; optional Prometheus/Grafana

***

## Part 1: Repository \& Data Versioning

- **Repository Structure:**

```
.
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
├── src/
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   └── schemas.py
├── models/
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/ci.yml
└── README.md
```

- **Data Preparation:**
    - In `data_prep.py`, combine and preprocess both CSVs, stratify split into train/test, save processed dataset.
- **DVC (optional):**
    - Track raw and processed datasets via DVC.

***

## Part 2: Model Development \& Experiment Tracking

- **Models:** LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- **MLflow Integration:**
    - Log hyperparameters, metrics (accuracy, precision, recall, F1), and feature importance.
    - Compare runs, register best model in MLflow Registry.
- **Reproducibility:** Commit all code, `mlruns/` (local only, gitignore), parameter configs.

***

## Part 3: API \& Docker Packaging

- **FastAPI Service:**
    - `GET /health` → returns status `"OK"`
    - `POST /predict` → accepts `WineFeatures` JSON (see below); returns predicted quality class and probabilities.
- **Pydantic Schema:**

```python
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    type: str  # "red" or "white"
```


***

## Part 4: CI/CD with GitHub Actions

- **Workflow (`ci.yml`):**

1. **Lint \& Test:** Run `flake8` and `pytest` on `src/` and `api/`
2. **Docker Build:** Build image `wine-api`
3. **Docker Push:** Push to Docker Hub (`dockerhub_user/wine-api:latest`)
4. **Deploy (optional self-hosted runner):** Stop existing container, pull new image, restart service locally
- **Secrets:** Store `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as secrets

***

## Part 5: Logging \& Monitoring

- **Logging:**
    - Record each request’s payload, timestamp, predicted class, probability, latency
    - Persist to daily rotating file (`logs/wine_api.log`) and a SQLite table (`predictions`)
- **Monitoring (Optional):**
    - `/metrics` endpoint exporting Prometheus metrics: request count, error count, latency histogram, model version
    - Use Docker Compose for Prometheus + Grafana dashboards (requests, P95 latency, error rate, class prediction dist.)

***

## Part 6: Summary \& Demo

- **1-Page Summary PDF:**
    - Architecture diagram
    - Tool choices/rationale
    - CI/CD workflow snapshot
    - MLflow run comparison screenshot
    - Monitoring dashboard screenshot

***

## Bonus Features

1. **Enhanced Input Validation:**
    - Numeric constraints (e.g., pH range), custom error messages in Pydantic
2. **Prometheus \& Grafana Dashboard:**
    - JSON dashboard ready for quality class monitoring
3. **Automated Retraining Trigger:**
    - On new data commit (DVC), auto-run training, register model, rebuild image, redeploy via GitHub Actions

***

## Deliverables

- **GitHub Repository:** All code, DVC files, MLflow scripts, GitHub Actions workflow, Docker config, README
- **Docker Hub Image:** `dockerhub_user/wine-api:latest`
- **Summary PDF:** Architecture overview and results

***