# ML Experiment Tracking Lab with MLflow

### **Objective**

Practice tracking, comparing, and managing machine learning experiments using MLflow with a simple scikit-learn model on the classic Iris dataset.

***

## **Lab Setup**

**Requirements:**

- Python 3.7+
- pip
- MLflow (`pip install mlflow`)
- scikit-learn (`pip install scikit-learn`)
- pandas (`pip install pandas`)

***

## **Part 1: Initial Setup**

1. **Create a new directory:**

```bash
mkdir mlflow_lab && cd mlflow_lab
```

2. **Create a new script `mlflow_iris.py`:**

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42
)

mlflow.set_experiment("iris_rf_exp_tracking")

with mlflow.start_run():
    n_estimators = 200
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Log hyperparameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    # Add input_example and infer model signature
    import pandas as pd
    input_example = pd.DataFrame(X_train[:2], columns=iris.feature_names) # two sample rows from training
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, clf.predict(X_train))
    
    # Log the model with input example and signature
    mlflow.sklearn.log_model(
        clf,
        name="model",
        input_example=input_example,
        signature=signature
    )
    
    print(f"Logged run with accuracy: {acc:.4f}")
```


***

## **Part 2: Run and Track Experiments**

**A. Run an Experiment**

```bash
python mlflow_iris.py
```

Repeat several times, tweaking `n_estimators` value each time in the script (e.g., try 10, 20, 100, 200).

***

**B. Launch the MLflow Tracking UI:**

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

***

## **Part 3: Lab Tasks**

### **Task 1: Log Multiple Experiments**

- Change `n_estimators` in the script (10, 50, 100, 200).
- Run the script each time.
- Open MLflow UI and compare the runs for best accuracy.


### **Task 2: Track Additional Parameters and Metrics**

- Add logging for another parameter, e.g., `max_depth`
- Log a new metric: e.g., F1-score

```python
from sklearn.metrics import f1_score
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("f1_score", f1_score(y_test, preds, average="micro"))
```


### **Task 3: Register the Best Model**

- From the MLflow UI, find your best run.
- Register the model in the “Model Registry” tab.


### **Task 4: Experiment with Notes \& Tags**

- Add tags to your runs using:

```python
mlflow.set_tag("experimenter", "your_name")
mlflow.set_tag("notes", "testing tree depth impacts")
```

- View and compare tagged runs in the UI.


### **Task 5: Bonus—Tracking Directory/Artifact**

- Log the confusion matrix as an artifact.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
plt.savefig("confusion.png")
mlflow.log_artifact("confusion.png")
```


***

## **Reflection Questions**

1. What hyperparameters had the biggest effect on accuracy?
2. How does tracking models and metrics help in real-world ML projects?
3. How does artifact logging (like confusion matrices) help understanding results?
4. Describe a scenario where you would use the Model Registry.

***

## **Cleanup**

To stop the MLflow UI, press `Ctrl+C` in the terminal window.

***

## **Summary Table**

| Task | Goal |
| :-- | :-- |
| Log and compare multiple runs | Identify best hyperparameters |
| Track additional params/metrics | Practice logging and analysis |
| Register best model | Understand registry concepts |
| Tag/note runs | Organize experiments |
| Log additional artifacts | Visualize/track non-scalar results |


***
