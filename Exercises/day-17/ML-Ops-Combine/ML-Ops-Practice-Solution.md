# Solution: ML Experiment Tracking, Registry \& API with Breast Cancer Dataset


***

## **Step 1: Setup**

Install all requirements:

```bash
pip install scikit-learn pandas mlflow fastapi uvicorn
```


***

## **Step 2: Training, Experiment Tracking, and Registry**

### `mlflow_breast_cancer.py`

```python
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and scale dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlflow.set_experiment("breast_cancer_registry_demo")
model_name = "breast_cancer_diagnosis"
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, solver='lbfgs')),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42))
]
for model_label, model in models:
    with mlflow.start_run(run_name=model_label) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        # Log params and metrics
        mlflow.log_param("model_type", model_label)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        # Confusion Matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap="Blues", alpha=0.7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], va='center', ha='center')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")
        plt.close()
        # Prepare for signature and input_example logging
        import numpy as np
        input_example = X_test[:2]  # X_test is already a numpy array
        signature = infer_signature(X_train, model.predict(X_train))
        # Register model with all options
        mlflow.sklearn.log_model(
            model,
            name=model_name,
            registered_model_name=model_name,
            input_example=input_example,
            signature=signature
        )
        print(f"{model_label} logged & registered, Run ID: {run.info.run_id}")
```


***

### **Step 3: Promote Best Model in the Registry**

1. Run the script several times (above).
2. Launch the MLflow UI:

```bash
mlflow ui
```

Open http://localhost:5000, go to **Models** tab, and promote the version with highest ROC-AUC to **"Production"**.

***

## **Step 4: Prediction API (FastAPI)**

### `api_breast_cancer.py`

```python
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List

# Must match the feature order as in training!
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

class CancerInput(BaseModel):
    features: conlist(float, min_items=30, max_items=30)

app = FastAPI()

@app.on_event("startup")
def load_model():
    # Loads the latest Production model. Change "Production" to "Staging" if needed.
    global model
    model = mlflow.pyfunc.load_model("models:/breast_cancer_diagnosis/Production")

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def predict(inp: CancerInput):
    preds = model.predict([inp.features])
    prob = model.predict_proba([inp.features])[:,1].tolist()[0]
    label = int(preds)
    return {
        "prediction": label,       # 0=benign, 1=malignant
        "probability_malignant": prob
    }
```

**Run the API:**

```bash
uvicorn api_breast_cancer:app --reload
```

- Visit `http://localhost:8000/docs` for Swagger UI and sample requests.

***

## **Step 5: Example Prediction (with curl)**

```bash
curl -X POST "http://localhost:8000/predict" \
-H  "accept: application/json" -H  "Content-Type: application/json" \
-d "{\"features\": [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
```


***

## **Summary Table**

| Step | File/Command | Purpose |
| :-- | :-- | :-- |
| Run experiment \& registry | `python mlflow_breast_cancer.py` | Track and register best model |
| Promote best model (UI) | `mlflow ui` | Set best version to "Production" |
| Serve API | `uvicorn api_breast_cancer:app --reload` | Launch prediction endpoint |
| API docs | `http://localhost:8000/docs` | Swagger interface for frontend devs |
| Predict using API | `curl ...` | Test API from command line |


***
