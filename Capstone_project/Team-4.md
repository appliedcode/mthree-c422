# Capstone Project with the Titanic Dataset (Survivability Classification)

**Main takeaway:**
Build a fully working MLOps pipeline for binary classification (“Survived”/“Not survived”) using the Titanic dataset. Cover data versioning, experiment tracking, API deployment, CI/CD, logging, and monitoring—ideal for hands-on learning on your local machine.

***

## Project Overview \& Architecture

- **Dataset:** Titanic (`train.csv` — 891 samples, features like age, fare, class, sex, siblings/spouses, parents/children)
- **Task:** Binary classification (`Survived`: 0 or 1)
- **Tools:**
    - Git + GitHub
    - DVC (optional)
    - MLflow (tracking \& model registry)
    - FastAPI + Pydantic
    - Docker
    - GitHub Actions
    - Python logging + SQLite
    - Prometheus + Grafana (optional/bonus)

***

## Architecture Diagram

1. **Data \& Code Versioning:** Git for code, DVC for data artifacts (e.g., CSVs)
2. **Experiment Tracking:** MLflow local server for logging and model registry
3. **API Service:** FastAPI with Pydantic input validation
4. **Containerization:** Dockerized API
5. **CI/CD:** GitHub Actions for build/test/deploy
6. **Logging \& Monitoring:** Logging to file/SQLite; Prometheus/Grafana (optional)

***

## Part 1: Repository \& Data Versioning

- **Suggested Structure:**

```
.
├── data/
│   └── train.csv
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
    - Use `prep_data.py` to load `train.csv`, handle missing values (e.g., fill/flag NA for Age, Cabin), engineer features (e.g., `IsChild`, ticket groupings), encode categoricals, and write cleaned data as `data/titanic_clean.csv`.
- **DVC (optional):**
    - Initialize and track datasets, keeping large files out of git.

***

## Part 2: Model Development \& Experiment Tracking

- **Models:** LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- **MLflow Integration:** Log model parameters, training metrics (accuracy, precision, recall, F1), and feature importance. Compare runs, register best model.
- **Reproducibility:** Track all parameters and commits. Version training scripts and configurations.

***

## Part 3: API \& Docker Packaging

- **FastAPI Service:**
    - `GET /health` → returns status `"OK"`
    - `POST /predict` → accepts `TitanicFeatures` JSON; returns predicted survival (0/1) and probability.
- **Pydantic Schema:**

```python
class TitanicFeatures(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    # Optionally: IsChild: int, Title: str, FamilySize: int
```


***

## Part 4: CI/CD with GitHub Actions

- **Workflow (`ci.yml`):**

1. **Lint/Test:** Run `flake8` and `pytest` on `src/` and `api/`
2. **Docker Build:** Build image `titanic-api`
3. **Docker Push:** Push to Docker Hub
4. **Deploy (optional self-hosted runner):** Update container with new image
- **Secrets:** Securely store Docker credentials for publishing images

***

## Part 5: Logging \& Monitoring

- **Logging:**
    - Record each request’s features, timestamp, prediction, probability, and latency
    - Persist to daily rotating log file, update `predictions` table in SQLite
- **Monitoring (Optional):**
    - `/metrics` endpoint for Prometheus; dashboard charts for request volume, latency, error rate, prediction ratio

***

## Part 6: Summary \& Demo

- **Summary PDF:**
    - Architecture diagram
    - CI/CD workflow snapshot
    - MLflow comparison screenshot
    - Monitoring dashboard screenshot

***

## Bonus Features

1. **Advanced validation:** Add logic for realistic Age ranges, Fare values, and custom error messages
2. **Pre-built monitoring dashboard:** Prometheus \& Grafana JSON configuration for prediction stats
3. **Automated retraining:** On new clean dataset, retrain model, register, build/push/redeploy via CI

***

## Deliverables

- Professional GitHub repo with all code/configs
- Docker Hub image for API server
- Detailed README and 1-page pipeline summary

***
