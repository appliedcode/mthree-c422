# Lab Exercise: Automating Titanic Survival Prediction Pipeline with Multi-Stage GitHub Actions and Docker


***

## Objective

Build a containerized ML pipeline to predict Titanic passenger survival and automate training, evaluation, model artifact management, and optional Docker image publication using a **multi-stage GitHub Actions workflow**.

***

## Use Case Scenario

Predict whether a passenger survived the Titanic disaster from demographics and travel info. The training and evaluation are automated in stages inside a Docker container through continuous integration/deployment pipelines.

***

## Prerequisites

- Python, Docker, Git, GitHub basics
- Understanding of binary classification and ML pipelines
- GitHub account, Docker installed locally

***

## Step 1: Setup GitHub Repository and Files

### 1. Create GitHub Repo

- Name it `titanic-survival-docker-ci`
- Clone locally:

```bash
git clone https://github.com/your-username/titanic-survival-docker-ci.git
cd titanic-survival-docker-ci
```


***

### 2. Add Python Scripts, Dockerfile, Requirements


***

#### train_titanic.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load Titanic data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Preprocess: select relevant columns and drop missing data
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Define categorical and numerical features
categorical_features = ['Sex']
numerical_features = ['Pclass', 'Age', 'Fare']

# Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Train/test split (not training-test split here, all data used to train model artifact)
pipeline.fit(X, y)

# Save trained model pipeline
joblib.dump(pipeline, 'titanic_model.joblib')
print("Titanic model trained and saved as titanic_model.joblib")
```


***

#### evaluate_titanic.py

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load Titanic data for evaluation
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Load trained model pipeline
model = joblib.load('titanic_model.joblib')

# Predict on eval data
y_pred = model.predict(X)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```


***

#### requirements.txt

```
pandas
scikit-learn
joblib
```


***

#### Dockerfile (Multi-stage build NOT included here as multi-stage handled in workflow)

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
git commit -m "Add Titanic train, evaluate scripts, requirements, and Dockerfile"
git push origin main
```


***

## Step 2: Multi-Stage GitHub Actions Workflow

Create `.github/workflows/titanic-docker-ci.yml`:

```yaml
name: Titanic Survival Multi-Stage CI/CD

on:
  push:
    branches: [ main ]

jobs:

  build:
    runs-on: ubuntu-latest
    outputs:
      image: ${{ steps.build-image.outputs.image }}
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build Docker image
      id: build-image
      run: |
        docker build -t titanic-survival:latest .
        echo "image=titanic-survival:latest" >> $GITHUB_OUTPUT

  train:
    runs-on: ubuntu-latest
    needs: build
    outputs:
      model-path: ${{ steps.train.outputs.model-path }}
    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Run training container
      id: train
      run: |
        docker run --rm titanic-survival:latest python train_titanic.py
        echo "model-path=titanic_model.joblib" >> $GITHUB_OUTPUT

    - name: Upload model artifact
      uses: actions/upload-artifact@v3
      with:
        name: titanic-model
        path: titanic_model.joblib

  evaluate:
    runs-on: ubuntu-latest
    needs: train
    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Download model artifact
      uses: actions/download-artifact@v3
      with:
        name: titanic-model
        path: .

    - name: Build Docker image
      run: docker build -t titanic-survival:latest .

    - name: Run evaluation
      run: docker run --rm -v ${{ github.workspace }}:/app -w /app titanic-survival:latest python evaluate_titanic.py

  publish:
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Push Docker image
      run: |
        docker tag titanic-survival:latest your-dockerhub-username/titanic-survival:latest
        docker push your-dockerhub-username/titanic-survival:latest
```


***

## Step 3: Verify Pipeline

- Push changes to trigger GitHub Actions workflow.
- Verify the **build**, **train**, **evaluate**, and optionally **publish** stages succeed.
- Confirm the model artifact is uploaded.
- Check logs for metrics output.

***

## Optional Extensions:

- Add unit tests and integrate into workflow.
- Deploy model container as API.
- Implement notifications on workflow events.
- Cache Docker layers to speed up builds.

***

## Deliverables

- GitHub repository with:
    - `train_titanic.py`
    - `evaluate_titanic.py`
    - `requirements.txt`
    - `Dockerfile`
    - `.github/workflows/titanic-docker-ci.yml`
- Evidence of successful multi-stage GitHub Actions runs and artifact uploads.

***