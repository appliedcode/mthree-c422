# Exercise: Assess and Enhance ML Process Maturity

## Objective

Follow a series of tasks to move your ML project towards best practices in reproducibility, validation, versioning, testing, automation, monitoring, and governance—all runnable locally on your machine.

***

## Prerequisites (Setup Once)

1. **Create and activate a project folder and virtual environment:**

```bash
mkdir ml-maturity-exercise
cd ml-maturity-exercise
python -m venv venv
source venv/bin/activate
```

2. **Install packages:**

```bash
pip install pandas scikit-learn mlflow pytest great_expectations prometheus_client
```

3. **Initialize Git repo:**

```bash
git init
```


***

## Step 1: Project Documentation \& Structure Audit

**Create the following files and folders:**

- `README.md`: with setup, dependencies, and run instructions.
- `requirements.txt`: list installed packages.
- `data/`: for datasets.
- `src/`: for code.
- `models/`: for saved models.

**Check completeness:**

```python
# structure_check.py
import os

paths = ['README.md', 'requirements.txt', 'data/', 'src/', 'models/']
for path in paths:
    assert os.path.exists(path), f"{path} not found! Please add."
print("Project folder structure is complete.")
```

_Run this from your repo root:_

```bash
python structure_check.py
```


***

## Step 2: Automate Data Validation

**Download sample dataset (Iris):**

```python
# src/download_data.py
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
df.to_csv("data/iris.csv", index=False)
```

_Run:_

```bash
python src/download_data.py
```

**Automated check using Great Expectations:**

```python
# src/validate_data.py
import great_expectations as ge
import pandas as pd

df = pd.read_csv("data/iris.csv")
df_ge = ge.from_pandas(df)
result = df_ge.expect_column_values_to_not_be_null("sepal_length")
assert result.success, "Nulls found in sepal_length!"
print("Data validation passed.")
```

_Run:_

```bash
python src/validate_data.py
```


***

## Step 3: Model Training, Logging and Versioning

**Train a model and log it to MLflow:**

```python
# src/train_and_log.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(clf, "rf_model")
    print(f"Model logged. Test accuracy: {score:.2f}")
```

_Run:_

```bash
python src/train_and_log.py
mlflow ui  # then open http://localhost:5000 and view the run!
```


***

## Step 4: Unit Testing

**Create and run a unit test for data integrity:**

```python
# src/test_data.py
import pandas as pd
def test_no_nulls():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0
```

_Run:_

```bash
pytest src/test_data.py
```


***

## Step 5: Simple CI/CD Pipeline (Local)

**Add a Makefile to automate steps:**

```
# Makefile
data:
	python src/download_data.py
validate:
	python src/validate_data.py
train:
	python src/train_and_log.py
test:
	pytest src/test_data.py
```

_Run (from project root):_

```bash
make data
make validate
make test
make train
```


***

## Step 6: Model Monitoring (Simulated Locally)

**Serve and monitor test accuracy with Prometheus:**

```python
# src/monitor.py
from prometheus_client import Gauge, start_http_server
import time

accuracy_gauge = Gauge('model_test_accuracy', 'Test accuracy of RF model')
accuracy_gauge.set(0.97)  # Replace with actual metric

start_http_server(8000)
print("Prometheus metrics server running on port 8000... Ctrl+C to exit.")
while True:
    time.sleep(10)
```

_Run:_

```bash
python src/monitor.py
```

Access metrics at http://localhost:8000 in your browser.

***

## Step 7: Simple Governance (User Check)

**Enforce deployment authorization:**

```python
# src/deploy.py
import getpass
approved_users = ['ml_lead', 'cto']
user = getpass.getuser()
assert user in approved_users, "You are not authorized to deploy!"
print(f"User {user} authorized. Deploying model.")
```

_Edit approved_users as needed, run:_

```bash
python src/deploy.py
```


***

## Step 8: Self-Assessment Script

**Automate maturity audit:**

```python
# src/maturity_audit.py
import os
criteria = {
    "data_versioned": os.path.exists('data/iris.csv'),
    "model_logged": os.path.exists('mlruns/'),
    "unit_tests": os.path.exists('src/test_data.py'),
    "cicd": os.path.exists('Makefile')
}
for key, value in criteria.items():
    print(f"{key}: {'OK' if value else 'Missing'}")
```

_Run:_

```bash
python src/maturity_audit.py
```


***

## Wrap-Up

- All code/scripts run locally; each script improves a key ML maturity process area.
- Mix and match steps for your own projects.
- Refine audit, monitoring, and governance for your organization’s needs.

***

**This complete workflow ensures practice and progress in ML process maturity, directly in your local environment.**

