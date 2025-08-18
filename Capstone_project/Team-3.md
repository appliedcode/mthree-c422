# Capstone Project with the Heart Disease UCI Dataset (Classification)

**Main takeaway:**
Build a complete MLOps pipeline for binary classification to detect heart disease, demonstrating skills in data management, experiment tracking, API deployment, CI/CD, logging, and monitoring. Ideal for hands-on experience on your local system.

***

## Project Overview \& Architecture

- **Dataset:** Heart Disease (Cleveland Clinic, 303 rows, multiple features: age, sex, chest pain type, resting blood pressure, etc.)
- **Task:** Binary classification (`target`: 1 = heart disease present, 0 = not present)
- **Tools:**
    - Git + GitHub
    - DVC (optional, for data versioning)
    - MLflow (experiment tracking \& model registry)
    - FastAPI + Pydantic
    - Docker
    - GitHub Actions
    - Python logging + SQLite
    - Prometheus + Grafana (optional/bonus)

***

## Architecture Diagram

1. **Data \& Code Versioning:** Use Git for code, optionally DVC for the CSV
2. **Experiment Tracking:** MLflow server for run logs and models
3. **API Service:** FastAPI with Pydantic for validation
4. **Containerization:** Docker image
5. **CI/CD:** GitHub Actions workflows
6. **Logging \& Monitoring:** Log requests/results; optional Prometheus/Grafana for live metrics

***

## Part 1: Repository \& Data Versioning

- **Suggested Structure:**

```
.
├── data/
│   └── heart.csv
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
    - Use `prep_data.py` to handle missing values, scale features, encode categoricals (`cp`, `sex`, `thal`, etc.), and save clean set for modeling.
- **DVC (optional):**
    - Track data/feature engineering.

***

## Part 2: Model Development \& Experiment Tracking

- **Models:** LogisticRegression, RandomForestClassifier, XGBClassifier
- **MLflow Integration:** Log hyperparams, metrics (accuracy, ROC AUC, precision, recall), feature importance, best model registry.
- **Reproducibility:** Version scripts/configs/parameters.

***

## Part 3: API \& Docker Packaging

- **FastAPI Service:**
    - `GET /health` → status
    - `POST /predict` → accept features, return prediction \& probability
- **Pydantic Schema:**

```python
class HeartFeatures(BaseModel):
    age: int
    sex: int
    cp: int               # chest pain type: 0-3
    trestbps: int         # resting blood pressure
    chol: int             # serum cholesterol
    fbs: int              # fasting blood sugar
    restecg: int          # resting ECG results
    thalach: int          # max heart rate
    exang: int            # exercise induced angina
    oldpeak: float        # ST depression induced by exercise
    slope: int            # slope of ST segment
    ca: int               # # of major vessels colored
    thal: int             # thalassemia: 0-3
```


***

## Part 4: CI/CD with GitHub Actions

- **Workflow (`ci.yml`):**

1. Lint/test
2. Build Docker image
3. Push image
4. (Optional) Deploy new container
- **Secrets:** Docker credentials

***

## Part 5: Logging \& Monitoring

- **Logging:** Record each prediction request, model version, timestamp, result, latency. Store in file and SQLite.
- **Monitoring (optional):** `/metrics` endpoint, Prometheus \& Grafana dashboards for live API health and prediction stats.

***

## Part 6: Summary \& Demo

- **Output:** Architecture diagram, summary PDF with screenshots

***

## Bonus Features

1. **Input validation:** check ranges and logical relationships
2. **Prometheus/Grafana dashboard:** plug-and-play
3. **Automated retraining via CI/CD:** on new data, retrain, publish, redeploy

***

## Deliverables

- GitHub Repo (`README.md` with full instructions)
- Docker Hub public image
- PDF summary

***