# Solution: Managing Model Versions in MLflow Model Registry with Wine Quality


***

## **Step 1: Setup**

Install required packages:

```bash
pip install pandas scikit-learn matplotlib mlflow
```


***

## **Step 2: Prepare the Data**

```python
import pandas as pd

# Download the red wine dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(data_url, sep=';')

# Convert quality to binary classification: good (>=6), bad (<6)
df['target'] = (df['quality'] >= 6).astype(int)
X = df.drop(['quality', 'target'], axis=1)
y = df['target']
```


***

## **Step 3: Train, Log, and Register Models**

Use the following script (save as `mlflow_wine_registry.py`). This will train **two models, log metrics and parameters, and register each in the MLflow Model Registry**.

```python
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load and prepare data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(data_url, sep=';')
df['target'] = (df['quality'] >= 6).astype(int)
X = df.drop(['quality', 'target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mlflow.set_experiment("wine_quality_registry_lab")
model_name = "wine_quality_predictor"

models = [
    {
        "label": "LogisticRegression",
        "model": LogisticRegression(max_iter=1000),
        "params": {"max_iter": 1000}
    },
    {
        "label": "RandomForest",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100}
    }
]

run_ids = []
accuracies = []

# Train and register both models
for entry in models:
    with mlflow.start_run(run_name=entry["label"]) as run:
        # Train
        entry["model"].fit(X_train, y_train)
        y_pred = entry["model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Log
        mlflow.log_params(entry["params"])
        mlflow.log_param("algorithm", entry["label"])
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(entry["model"], "model")
        mlflow.set_tag("note", f"Trained with {entry['label']} on wine quality dataset")
        
        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        print(f"Registered {entry['label']} as version {mv.version} (accuracy: {acc:.3f})")
        run_ids.append(run.info.run_id)
        accuracies.append(acc)

# Print best version info
best_idx = int(accuracies.index(max(accuracies)))
print(f"\nBest model: {models[best_idx]['label']} (run_id={run_ids[best_idx]})")
```


***

## **Step 4: Promote and Annotate Models in MLflow UI**

1. Start MLflow UI:

```bash
mlflow ui
```

Open: http://localhost:5000
2. Go to the **Models** tab. You’ll see `wine_quality_predictor` with two versions.
3. Promote the version with the highest accuracy to **"Staging"** or **"Production"**.
4. Add descriptive comments or annotations (algorithm, accuracy, special notes) in the Model Registry UI.

***

## **Step 5: Use the Latest Staging/Production Model for Inference**

You can load the promoted model by stage as follows:

```python
import mlflow.pyfunc

# Change "Staging" to "Production" if you promoted there
model = mlflow.pyfunc.load_model("models:/wine_quality_predictor/Staging")

# Predict on new data or test set
sample = X_test.iloc[[0]]
prediction = model.predict(sample)
print("Prediction:", prediction)
```


***

## **Step 6: Discussion**

- The registry allows you to track versions and safely promote/rollback models.
- Using model stages integrates with CI/CD pipelines and team workflows.
- Model and metadata are stored for full reproducibility.

***

## **Summary Table**

| Step | Action/Command |
| :-- | :-- |
| Train and log models | `python mlflow_wine_registry.py` |
| Register in model registry | Done in script \& confirmed in MLflow UI |
| Promote, annotate, manage versions | MLflow Models UI, promote best to "Staging" |
| Load model by stage | `mlflow.pyfunc.load_model("models:/.../Staging")` |
| Make predictions | Use `model.predict()` on sample/test data |


***
