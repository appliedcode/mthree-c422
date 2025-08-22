# 🚀 Lab Exercise: Multi-Stage MLOps Pipeline with Docker and FastAPI for Predicting Housing Prices using the Boston Housing Dataset


***

## Objective

Build a multi-stage MLOps pipeline that includes:

- Data processing and feature preparation
- Regression model building and training
- Containerizing training and serving applications in Docker
- Pushing Docker images to Docker Hub
- Serving the model via a FastAPI regression API
- Automating the entire workflow with sequential GitHub Actions jobs

***

## Dataset

Use the **Boston Housing Dataset** (available from scikit-learn or UCI repository). Input features include properties of houses and the target is the median house price.

***

## Project Structure (Example)

```
.
├── app
│   ├── data_processing.py
│   ├── model_training.py
│   ├── serve_model.py
│   ├── requirements.txt
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── .github
│   └── workflows
│       └── mlops-boston-pipeline.yml
```


***

## Step 1: Data Processing and Feature Engineering

### File: `app/data_processing.py`

- Load Boston housing dataset via scikit-learn.
- Convert to pandas DataFrame.
- Perform any necessary feature scaling or transformation.
- Save processed features and target as CSV files (`features.csv` and `target.csv`).

***

## Step 2: Model Training

### File: `app/model_training.py`

- Load processed data CSV files.
- Split into train/test sets.
- Train a regression model (e.g., RandomForestRegressor).
- Save the trained model as pickle or joblib (`boston_model.joblib`).
- Save evaluation metrics (e.g., RMSE) to JSON.

***

## Step 3: FastAPI Model Serving

### File: `app/serve_model.py`

- Load the serialized model artifact.
- Create FastAPI app with `/predict` POST endpoint accepting feature inputs.
- Return predicted house price.

***

## Step 4: Docker Image for Training

### File: `app/Dockerfile.train`

- Base image: Python 3.10-slim
- Install dependencies
- Copy processing and training scripts
- Execute data processing and model training in build steps

***

## Step 5: Docker Image for Serving

### File: `app/Dockerfile.serve`

- Base image: Python 3.10-slim
- Install dependencies including FastAPI and Uvicorn
- Copy the serving app and saved model artifact
- Default command runs FastAPI server

***

## Step 6: GitHub Actions Multi-Stage Workflow

### File: `.github/workflows/mlops-boston-pipeline.yml`

- Job 1: Data processing and artifact upload
- Job 2: Model training consuming artifacts from Job 1, saving model artifact
- Job 3: Build \& push training and serving Docker images
- Job 4: Deploy container as service and test prediction API on localhost

***

## Sample Key Code Snippets


***

### `data_processing.py`

```python
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

def process_data():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name='PRICE')

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_scaled.to_csv('features.csv', index=False)
    y.to_csv('target.csv', index=False)
    print("Processed features and target saved.")

if __name__ == '__main__':
    process_data()
```


***

### `model_training.py`

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import json

def train():
    X = pd.read_csv('features.csv')
    y = pd.read_csv('target.csv').squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE: {rmse:.4f}")

    joblib.dump(model, 'boston_model.joblib')
    with open('metrics.json', 'w') as f:
        json.dump({'rmse': rmse}, f)
    print("Model and metrics saved.")

if __name__ == '__main__':
    train()
```


***

### `serve_model.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('boston_model.joblib')

class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    X = pd.DataFrame([features.dict()])
    pred_price = model.predict(X)[0]
    return {"predicted_price": pred_price}
```


***

## Instructions

- Use multi-stage GitHub Actions workflow to link each job as dependency
- Secure Docker Hub credentials as GitHub secrets
- Automate Docker image builds and pushes for both training and serving containers
- Deploy the serving container as a service in the final GitHub Actions job and run inference tests via curl or HTTP client

