# Capstone Project with the California Housing Dataset (Regression)

**Main takeaway:**
Develop a full MLOps pipeline to predict house prices, applying data versioning, experiment tracking, API deployment, CI/CD, logging, and monitoring. This project is ideal for hands-on mastery of model ops and deployment for real-world tabular regression.

***

## Project Overview \& Architecture

- **Dataset:** California Housing (20,000+ samples; features: longitude, latitude, housing median age, rooms, population, median income, etc.)
- **Task:** Regression (`target`: median house value)
- **Tools:**
    - Git + GitHub
    - DVC (optional, for data versioning)
    - MLflow (experiment tracking \& model registry)
    - FastAPI + Pydantic
    - Docker
    - GitHub Actions
    - Python logging + SQLite
    - Prometheus + Grafana (optional)

***

## Architecture Diagram

1. **Data \& Code Versioning:** Git for code, DVC for dataset files
2. **Experiment Tracking:** MLflow server for metrics/models
3. **API Service:** FastAPI with Pydantic validation
4. **Containerization:** Docker image
5. **CI/CD:** GitHub Actions workflows
6. **Logging \& Monitoring:** Logs to file/SQLite; Prometheus/Grafana dashboards (optional)

***

## Part 1: Repository \& Data Versioning

- **Suggested Structure:**

```
.
├── data/
│   └── cal_housing.csv
├── src/
│   ├── prep_data.py
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
    - Use `prep_data.py` to load and clean data, handle missing values, scale features, and output processed data for modeling.
- **DVC (optional):**
    - Track raw/engineered datasets.

***

## Part 2: Model Development \& Experiment Tracking

- **Models:** LinearRegression, RandomForestRegressor, XGBRegressor
- **MLflow Integration:** Log hyperparams, metrics (MAE, RMSE, R²), feature importance
- **Reproducibility:** Versioned scripts/configs; log all parameters

***

## Part 3: API \& Docker Packaging

- **FastAPI Service:**
    - `GET /health` → status
    - `POST /predict` → accepts housing features, returns predicted price
- **Pydantic Schema:**

```python
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
```


***

## Part 4: CI/CD with GitHub Actions

- **Workflow (`ci.yml`):**

1. Lint/test with pytest \& flake8
2. Build Docker image
3. Push image to Docker Hub
4. Deploy container via self-hosted runner (optional)
- **Secrets:** Docker credentials

***

## Part 5: Logging \& Monitoring

- **Logging:** Track feature payloads, prediction output, request latency, errors; save to file and SQLite.
- **Monitoring (optional):** `/metrics` endpoint with Prometheus; Grafana dashboards for API and prediction stats.

***

## Part 6: Summary \& Demo

- **Summary PDF:**
    - Architecture diagram
    - CI/CD workflow
    - MLflow run comparisons
    - Monitoring screenshots

***

## Bonus Features

1. **Robust validation:** Ensure realistic ranges for all features; custom warnings for outliers.
2. **Prometheus/Grafana:** Plug-and-play dashboards for request volume, latency, error rate, price distribution.
3. **Automated retraining:** On new data, the pipeline retrains, registers, and redeploys via CI.

***

## Deliverables

- GitHub repo with all code/configs
- Docker Hub public API image
- PDF summary and instructions

***