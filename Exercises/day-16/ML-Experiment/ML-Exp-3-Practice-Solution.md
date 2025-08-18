# Solution: ML Experiment Tracking on Heart Disease Dataset with MLflow


***

## Step 1: Setup

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib mlflow xgboost
```


***

## Step 2: Python Script for Experiment Tracking

Save this as `mlflow_heart_disease.py`:

```python
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

# Load the dataset from UCI repo
DATA_URL = "https://raw.githubusercontent.com/ishaan28may/uci-heart-disease-dataset/master/heart.csv"
df = pd.read_csv(DATA_URL)

# Feature matrix and target vector
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Choose your model and hyperparameters here
model_name = "LogisticRegression"  # Change to "RandomForest" or "XGBoost" for other models

# Hyperparameters
if model_name == "LogisticRegression":
    model = LogisticRegression(C=1.0, max_iter=1000)
    params = {"C": 1.0}
elif model_name == "RandomForest":
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    params = {"n_estimators": 100, "max_depth": 5}
elif model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5)
    params = {"n_estimators": 100, "max_depth": 5}
else:
    raise ValueError(f"Invalid model_name: {model_name}")

# Set MLflow experiment name
mlflow.set_experiment("heart_disease_mlflow_experiments")

with mlflow.start_run() as run:
    # Log model name and hyperparams
    mlflow.log_param("model_name", model_name)
    for k, v in params.items():
        mlflow.log_param(k, v)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Confusion Matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()
    
    # ROC Curve artifact
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    roc_path = "roc_curve.png"
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Tags & Notes
    mlflow.set_tag("experimenter", "your_name")
    mlflow.set_tag("notes", f"Training with {model_name} and params {params}")
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
```


***

## Step 3: Running Experiments

1. Run the script multiple times, changing:
    - `model_name` (try `"RandomForest"`, `"XGBoost"`)
    - Hyperparameters inside the script (e.g., `C` for LogisticRegression, `n_estimators` for tree models)
```bash
python mlflow_heart_disease.py
```

2. Start the MLflow UI to monitor runs:
```bash
mlflow ui
```

Open in browser: http://localhost:5000

***

## Step 4: Register Best Model

- Use the MLflow UI to find the best run (e.g., highest ROC AUC).
- Register the model in the “Model Registry” tab for deployment or future use.

***

## Summary Table

| Step | Notes |
| :-- | :-- |
| Install packages | pandas, scikit-learn, mlflow, xgboost |
| Load and preprocess | Load data, split train/test |
| Choose models \& params | LogisticRegression, RF, XGB |
| Log params, metrics, artifacts | Confusion matrix, ROC curve |
| Run experiments | Vary models \& hyperparameters |
| Use MLflow UI | Compare, tag, annotate, register |


***
