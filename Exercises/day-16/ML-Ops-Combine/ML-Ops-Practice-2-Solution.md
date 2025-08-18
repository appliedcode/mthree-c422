# Solution: Bike Sharing Demand Prediction with MLflow \& FastAPI


***

## Step 1: Setup

Install dependencies:

```bash
pip install pandas scikit-learn mlflow fastapi uvicorn matplotlib
```


***

## Step 2: Data Preparation \& Experiment Tracking

Create a Python script `mlflow_bike_demand.py`:

```python
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
df = pd.read_csv("hour.csv", compression='zip')

# Select features and target
features = ['season', 'holiday', 'workingday', 'weathersit', 'temp',
            'atemp', 'hum', 'windspeed', 'hr']
target = 'cnt'

X = df[features]
y = df[target]

# Train/test split (last 21 days as test approx)
train_size = len(df) - 24*21
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

mlflow.set_experiment("bike_demand_experiments")
model_name = "bike_demand_predictor"

models = [
    ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

for model_label, model in models:
    with mlflow.start_run(run_name=model_label) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_type", model_label)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        # Plot predictions vs actual
        plt.figure(figsize=(10,5))
        plt.plot(y_test.values, label="Actual")
        plt.plot(preds, label="Predicted")
        plt.title(f"{model_label} Predictions vs Actual")
        plt.legend()
        plt.tight_layout()
        plot_path = "pred_vs_actual.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        print(f"Run {run.info.run_id} logged with RMSE={rmse:.3f}, R2={r2:.3f}")
```


***

## Step 3: Promote Best Model

- Run the above script multiple times (possibly tweak model hyperparameters).
- Start MLflow UI for Model Registry:

```bash
mlflow ui
```

- Access http://localhost:5000, navigate to **Models** tab.
- Promote the best model (lowest RMSE) to “Production” stage.

***

## Step 4: Prediction API with FastAPI

Create `api_bike_demand.py`:

```python
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Features order must match training features
FEATURE_NAMES = ['season', 'holiday', 'workingday', 'weathersit', 'temp',
                 'atemp', 'hum', 'windspeed', 'hr']

class BikeDemandInput(BaseModel):
    features: List[float]  # expects 9 features

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/bike_demand_predictor/Production")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: BikeDemandInput):
    if len(data.features) != len(FEATURE_NAMES):
        return {"error": f"Expected {len(FEATURE_NAMES)} features."}
    pred = model.predict([data.features])[0]
    return {"predicted_bike_count": pred}
```

Run the API service:

```bash
uvicorn api_bike_demand:app --reload
```


***

## Step 5: Test the API

Example curl request replacing feature values accordingly:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [1,0,1,1,0.22,0.21,0.38,0.12,14]}'
```


***

## Summary Table

| Step | Command/File | Purpose |
| :-- | :-- | :-- |
| Run experiments | `python mlflow_bike_demand.py` | Train, track, and register models |
| Promote best model | via MLflow UI (http://localhost:5000) | Promote to Production |
| Serve API | `uvicorn api_bike_demand:app --reload` | Model prediction service |
| Test API | curl command | Validate API returns predictions |


***
