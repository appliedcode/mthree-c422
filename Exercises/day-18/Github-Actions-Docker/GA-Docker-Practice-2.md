# Lab Exercise: Automating Titanic Survival Prediction with Docker and GitHub Actions


***

## Objective

Build and automate a machine learning pipeline to predict Titanic passenger survival, containerize the environment with Docker, and automate training, evaluation, and artifact management using GitHub Actions CI/CD.

***

## Use Case Scenario

Predict whether a passenger survived the Titanic disaster based on demographic and travel information. Automate the entire workflow from data preprocessing to model packaging with Docker and CI/CD pipelines.

***

## Prerequisites

- Python, Docker, Git, and GitHub knowledge
- Basic understanding of classification models
- A GitHub account and Docker installed locally

***

## Step 1: Setup GitHub Repository and Files

### 1. Create GitHub Repo

- Create repo named `titanic-survival-docker-ci`
- Clone locally:

```bash
git clone https://github.com/your-username/titanic-survival-docker-ci.git
cd titanic-survival-docker-ci
```


### 2. Add Python scripts, Dockerfile, and requirements


***

### train_titanic.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Basic preprocessing
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Define categorical and numerical features
categorical_features = ['Sex']
numerical_features = ['Pclass', 'Age', 'Fare']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'titanic_model.joblib')
print("Model trained and saved as titanic_model.joblib")
```


***

### evaluate_titanic.py

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load test dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Load model
model = joblib.load('titanic_model.joblib')

# Predict
y_pred = model.predict(X)

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```


***

### requirements.txt

```
pandas
scikit-learn
joblib
```


***

### Dockerfile

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "train_titanic.py"]
```


***

### 3. Commit and push initial code

```bash
git add .
git commit -m "Add Titanic training, evaluation scripts, Dockerfile, and requirements"
git push origin main
```


***

## Step 2: GitHub Actions Workflow

### 1. Create workflow directory and file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/titanic-docker-ci.yml`:

```yaml
name: Titanic Survival Docker CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build_and_test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build Docker image
      run: docker build -t titanic-survival:latest .

    - name: Run training container
      run: docker run --rm titanic-survival:latest

    - name: Run evaluation container
      run: docker run --rm titanic-survival:latest python evaluate_titanic.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v3
      with:
        name: titanic_model
        path: titanic_model.joblib
```


***

### 2. Commit and push workflow

```bash
git add .github/workflows/titanic-docker-ci.yml
git commit -m "Add GitHub Actions workflow for Docker build, training, and evaluation"
git push origin main
```


***

## Step 3: Verify

- Monitor workflow runs under GitHub Actions tab.
- Ensure all steps pass.
- Confirm `titanic_model.joblib` artifact uploaded.

***

## Optional Extensions

- Push Docker image to container registry.
- Add notifications on workflow status.
- Extend to serve model as API in a container.
- Add tests and validation steps to workflow.

***

## Deliverables

- GitHub repo with training, evaluation scripts, Dockerfile, workflow
- Evidence of successful automated runs with artifacts

***
