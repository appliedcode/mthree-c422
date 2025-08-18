# Wine Quality MLflow Experiment Tracking — Solution \& Code


***

## **Step 1: Setup**

**Install required packages:**

```bash
pip install pandas scikit-learn mlflow matplotlib requests
```


***

## **Step 2: Download the Dataset**

```python
import pandas as pd

# Download red wine dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(data_url, sep=';')

# Convert quality to binary: good (>=6), bad (<6)
df['target'] = (df['quality'] >= 6).astype(int)
```


***

## **Step 3: MLflow Experiment Tracking Script**

Save this as `mlflow_wine_lab.py`:

```python
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(data_url, sep=";")
df['target'] = (df['quality'] >= 6).astype(int)
X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Experiment parameters
n_estimators = 100          # Try changing this value
max_depth = 5               # Try changing this value

mlflow.set_experiment("wine_quality_rf_exp")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model training
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    plt.title("Confusion Matrix")
    im = ax.imshow(cm, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar(im)
    for i in range(cm.shape[0]):
        for j in range(cm.shape):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Feature importance plot artifact
    feat_imp = clf.feature_importances_
    features = X.columns
    plt.figure(figsize=(8,5))
    plt.barh(features, feat_imp)
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    # Tags
    mlflow.set_tag("note", "Baseline RF run")
    mlflow.set_tag("model_type", "RandomForest")

    print(f"Run {run.info.run_id}: accuracy={acc:.3f}, f1={f1:.3f}")
```


***

## **Step 4: Running the Experiments**

1. Run the script several times, changing `n_estimators` and/or `max_depth` at the top.

```bash
python mlflow_wine_lab.py
```

2. Start the MLflow tracking UI:

```bash
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000) in your browser.
3. Inspect experiment results: compare accuracy, F1, artifacts (plots), tags, and parameters.

***

## **Step 5: Register the Best Model**

- In MLflow UI, select the run with the best F1-score.
- In Run Details, use “Register Model” to add it to the MLflow Model Registry.

***

## **Advanced Tasks**

- Try a different classifier by swapping `RandomForestClassifier` with `LogisticRegression` or `XGBClassifier` and re-run.
- Log additional metrics, such as ROC-AUC.
- Try multiclass classification (predict exact quality score).

***

## **Summary Table**

| Step | Command / Script Section |
| :-- | :-- |
| Install packages | `pip install ...` |
| Download data | `pd.read_csv(...)` |
| Edit/run experiments | Change hyperparams \& rerun |
| Launch MLflow UI | `mlflow ui` |
| Register best model | Use UI Registry tab |


***
