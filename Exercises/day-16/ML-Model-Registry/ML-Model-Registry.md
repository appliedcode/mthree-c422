# ML Model Registry Exercise Lab

## Objective

Practice registering and managing different versions of trained ML models using the MLflow Model Registry. Learn to promote models to production and observe how registration integrates with experiment tracking and deployment workflows.

***

## Prerequisites

- Python 3.7+
- `mlflow`
- `scikit-learn` and `pandas`
- Completed MLflow ML experiment runs (from previous labs or any scikit-learn experiment)

Install dependencies if needed:

```bash
pip install mlflow scikit-learn pandas
```


***

## Part 1: Train and Log Two Models

1. **Use any classification dataset**. Example: Iris.
2. Run the following script to create two experiment runs:
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("iris_registry_lab")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# First model
with mlflow.start_run(run_name="rf_model_v1") as run1:
    clf1 = RandomForestClassifier(n_estimators=30, random_state=42)
    clf1.fit(X_train, y_train)
    acc1 = clf1.score(X_test, y_test)
    mlflow.log_param("n_estimators", 30)
    mlflow.log_metric("accuracy", acc1)
    mlflow.sklearn.log_model(clf1, artifact_path="model")

# Second model
with mlflow.start_run(run_name="rf_model_v2") as run2:
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2.fit(X_train, y_train)
    acc2 = clf2.score(X_test, y_test)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc2)
    mlflow.sklearn.log_model(clf2, artifact_path="model")

print("Two runs logged. Now open MLflow UI to register them.")
```


***

## Part 2: Register Models in MLflow Model Registry

1. **Start the MLflow UI**:

```bash
mlflow ui
```

Visit: http://localhost:5000
2. **Navigate to the “Experiments” tab:**
    - Find your two recent runs.
    - For each run, click on the run name to see details.
    - Click on the **“Register Model”** button above the logged model artifact.
3. **Register them with the same model name**, e.g., `iris_random_forest`.
4. **Go to the “Models” tab** (left sidebar).
    - You will see your registered model, with two versions.
    - Optionally, add descriptions or comments for each model version.

***

## Part 3: Promote and Manage Model Versions

1. **Change Model Stage**
    - In the Model Registry, promote the best version to “Staging” and/or “Production.”
    - Try archiving an older version.
2. **Add Annotations and Descriptions**
    - Edit model version descriptions, e.g., “Trained with n_estimators=100, best test accuracy.”
3. **Load Model by Name and Stage from Registry**

Example code:

```python
import mlflow.pyfunc

# Load the latest production model
model_uri = "models:/iris_random_forest/Production"
model = mlflow.pyfunc.load_model(model_uri)
print(model.predict([[5.8, 2.7, 5.1, 1.9]]))
```


***

## Part 4: Reflection \& Discussion

- How does the Model Registry support deployment and rollback workflows?
- What’s the difference between “None”, “Staging”, and “Production” stages?
- How would you automate model promotion in a CI/CD pipeline?

***

## Summary Table

| Task | MLflow UI section |
| :-- | :-- |
| Log/trained model | Experiments |
| Register in the registry | Model artifact, Register |
| Promote/annotate versions | Models |
| Load by name/stage | MLflow Registry \& pyfunc |


***

**End of lab:**
