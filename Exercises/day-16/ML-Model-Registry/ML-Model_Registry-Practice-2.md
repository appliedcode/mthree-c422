# Advanced Lab Exercise: Managing Model Versions in MLflow Model Registry with Heart Disease Dataset (Local Setup)

This lab guides students step-by-step through training, logging, registering, versioning, promoting, and serving machine learning models using MLflow Model Registry with the Heart Disease dataset. It includes directory structure, file contents, and runnable code designed to be run on a **local system**.

***

## Dataset

- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) (you can download and save `heart.csv` locally)
- Features include age, sex, chest pain type, resting blood pressure, cholesterol, etc.
- Target: Heart disease presence (1 = disease, 0 = no disease)

***

## Directory Structure

```
heart_disease_mlflow/
│
├── data/
│   └── heart.csv                              # Dataset CSV file
│
├── src/
│   ├── data_processing.py                     # Data loading and preprocessing
│   ├── train_models.py                        # Training, logging, and registration code
│   ├── inference.py                          # Inference code loading model by stage
│
├── mlruns/                                   # MLflow local tracking artifact (auto)
│
├── models/                                   # Saved models (optional)
│
├── README.md                                 # Project description and instructions
│
├── requirements.txt                         # Required libraries
│
└── config.yaml                             # Optional config for hyperparameters
```


***

## File Contents

### 1. `data/heart.csv`

You should download the Heart Disease `heart.csv` file from the UCI repository or Kaggle and place it here.

***

### 2. `requirements.txt`

```plaintext
pandas
scikit-learn
mlflow
pyyaml
```


***

### 3. `src/data_processing.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path='data/heart.csv'):
    df = pd.read_csv(data_path)
    # Example: drop rows with missing values if any
    df = df.dropna()

    # Assume target column is named 'target'
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/val/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    # Feature Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
```


***

### 4. `src/train_models.py`

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_processing import load_and_preprocess

experiment_name = "heart_disease_classification"
mlflow.set_experiment(experiment_name)

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    auc = roc_auc_score(y_val, probs) if probs is not None else None
    return acc, f1, auc

def train_and_log():
    X_train, y_train, X_val, y_val, _, _, scaler = load_and_preprocess()

    models_to_train = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver='liblinear')),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ]

    for name, model in models_to_train:
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            acc, f1, auc = evaluate_model(model, X_val, y_val)

            # Log params and metrics
            mlflow.log_param("model_type", name)
            mlflow.log_param("hyperparameters", model.get_params())
            mlflow.log_metric("val_accuracy", acc)
            mlflow.log_metric("val_f1", f1)
            if auc is not None:
                mlflow.log_metric("val_auc", auc)

            # Log model and scaler
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")

            print(f"Trained {name} - Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}" if auc else f"Trained {name} - Acc: {acc:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    train_and_log()
```


***

### 5. `src/inference.py`

```python
import mlflow.pyfunc
import pandas as pd
import numpy as np

def load_model(stage="Staging", model_name="heart_disease_classification"):
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

if __name__ == "__main__":
    # Example inference with dummy data or outside loaded CSV
    model = load_model(stage="Staging")

    sample_data = pd.DataFrame({
        "age": [^55],
        "sex": [^1],
        "cp": [^2],
        "trestbps": ,
        "chol": ,
        "fbs": ,
        "restecg": [^1],
        "thalach": ,
        "exang": ,
        "oldpeak": [1.5],
        "slope": [^3],
        "ca": ,
        "thal": [^3]
    })

    preds = model.predict(sample_data)
    print(f"Prediction on sample data: {preds}")
```


***

### 6. Optional: `config.yaml`

```yaml
experiment_name: heart_disease_classification
models:
  - name: LogisticRegression
    params:
      max_iter: 1000
      solver: liblinear
  - name: RandomForest
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
```


***

## How to run the lab on local system

### Setup

1. Create folder structure as above.
2. Place `heart.csv` file in `data/`.
3. Create and activate Python virtual environment (optional but recommended).
4. Install dependencies:
```bash
pip install -r requirements.txt
```


### Run training and logging

```bash
python src/train_models.py
```


### Access MLflow UI locally

Run MLflow tracking server locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open browser at `http://localhost:5000` to monitor experiments and model registry.

### Register models to MLflow Model Registry via CLI or Python API

Example with Python API (in notebook or script):

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "heart_disease_classification"

experiment = client.get_experiment_by_name(model_name)
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.val_f1 DESC"])

for run in runs:
    mv = client.create_model_version(model_name, f"runs:/{run.info.run_id}/model", run.info.run_id)
    print(f"Registered version {mv.version} from run {run.info.run_id}")

# Promote best version to staging
best_version = 1  # update to best version number
client.transition_model_version_stage(model_name, best_version, "Staging")
```


### Run inference

```bash
python src/inference.py
```


***

## Lab Reflection \& Extension Questions

- How does the MLflow registry help manage multiple model versions and safely promote models?
- What strategies ensure reproducibility and traceability when models/experiments evolve?
- How could you implement automated retraining triggered by new data or model drift?
- How would you extend this with model explainability or CI/CD integration?

***

