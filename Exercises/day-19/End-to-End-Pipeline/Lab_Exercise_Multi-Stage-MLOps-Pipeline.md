# 🚀 Lab Exercise: Multi-Stage MLOps Pipeline with Data Processing, Model Training, Dockerization, and FastAPI Serving, all Automated via GitHub Actions


***

## Objective

Implement a multi-stage MLOps pipeline that runs the following stages in sequence:

1. **Data Processing \& Cleaning**
2. **Model Training**
3. **Docker Image Build \& Push (Training \& Serving Images)**
4. **Model Serving with FastAPI**
5. **API Health Check \& Prediction Testing**

All stages are executed and orchestrated as dependent jobs in a **single GitHub Actions workflow**, ensuring smooth and orderly execution.

***

## Project File Structure

```
.
├── app
│   ├── data_processing.py
│   ├── model_training.py
│   ├── serve_model.py
│   ├── requirements.txt
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── raw_titanic.csv
└── .github
    └── workflows
        └── mlops-pipeline.yml
```


***

## Complete Multi-Stage GitHub Actions Workflow

### File: `.github/workflows/mlops-pipeline.yml`

```yaml
name: MLOps Multi-Stage Pipeline

on:
  push:
    branches:
      - main

jobs:

  data-processing:
    name: Data Processing & Cleaning
    runs-on: ubuntu-latest
    outputs:
      processed_data_path: processed_titanic.csv

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10

    - name: Install dependencies
      run: pip install pandas

    - name: Run data processing
      run: |
        python app/data_processing.py
      # The data_processing.py writes processed_titanic.csv
    - name: Upload processed data as artifact
      uses: actions/upload-artifact@v3
      with:
        name: processed-data
        path: app/processed_titanic.csv

  model-training:
    name: Model Training
    runs-on: ubuntu-latest
    needs: data-processing
    outputs:
      model_path: titanic_model.joblib

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Download processed data
      uses: actions/download-artifact@v3
      with:
        name: processed-data
        path: app/

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10

    - name: Install dependencies
      run: pip install pandas scikit-learn joblib

    - name: Run model training
      run: python app/model_training.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v3
      with:
        name: model-artifact
        path: app/titanic_model.joblib

  docker-build-push:
    name: Build and Push Docker Images
    runs-on: ubuntu-latest
    needs: model-training

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Download model artifact
      uses: actions/download-artifact@v3
      with:
        name: model-artifact
        path: app/

    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build training Docker image
      run: |
        docker build -t ${{ secrets.DOCKER_USERNAME }}/titanic-train:latest -f app/Dockerfile.train .

    - name: Push training image
      run: docker push ${{ secrets.DOCKER_USERNAME }}/titanic-train:latest

    - name: Build serving Docker image
      run: |
        docker build -t ${{ secrets.DOCKER_USERNAME }}/titanic-serve:latest -f app/Dockerfile.serve .

    - name: Push serving image
      run: docker push ${{ secrets.DOCKER_USERNAME }}/titanic-serve:latest

  serve-and-test:
    name: Serve and Test Model API
    runs-on: ubuntu-latest
    needs: docker-build-push

    services:
      model-service:
        image: ${{ secrets.DOCKER_USERNAME }}/titanic-serve:latest
        ports:
          - 8000:8000
        options: >-
          --health-cmd "curl --fail http://localhost:8000/health || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - name: Wait for service health
      run: |
        timeout 60 bash -c 'until curl --silent --fail http://localhost:8000/health; do sleep 5; done'

    - name: Test prediction API
      run: |
        curl -X POST http://localhost:8000/predict \
          -H "Content-Type: application/json" \
          -d '{"Pclass":3,"Sex":0,"Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked_Q":0,"Embarked_S":1}'
```


***

## Supporting Code Files (in `app/` folder)


***

### 1. `data_processing.py`

```python
import pandas as pd

def process_data():
    df = pd.read_csv('../raw_titanic.csv')  # Adjust path as needed
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[^0], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    df.to_csv('processed_titanic.csv', index=False)
    print("Processed data saved.")

if __name__ == '__main__':
    process_data()
```


***

### 2. `model_training.py`

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv('processed_titanic.csv')
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.4f}")
    
    joblib.dump(model, 'titanic_model.joblib')
    print("Model saved.")

if __name__ == '__main__':
    train()
```


***

### 3. `serve_model.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('titanic_model.joblib')

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int = 0
    Embarked_S: int = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Passenger):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[^0]
    proba = model.predict_proba(df)[^1]
    
    return {"survived": bool(pred), "survival_probability": proba}
```


***

### 4. `requirements.txt`

```
pandas
scikit-learn
joblib
fastapi
uvicorn
pydantic
```


***

## Dockerfiles


***

### `Dockerfile.train`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy raw data and scripts
COPY raw_titanic.csv ../raw_titanic.csv
COPY data_processing.py .
COPY model_training.py .

RUN python data_processing.py
RUN python model_training.py

CMD ["echo", "Training finished"]
```


***

### `Dockerfile.serve`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve_model.py .
COPY titanic_model.joblib .

EXPOSE 8000

CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
```


***

## How to Run and Test

1. Push the repo to GitHub with the above structure.
2. Add your Docker Hub credentials as GitHub secrets: `DOCKER_USERNAME`, `DOCKER_PASSWORD`.
3. The workflow `.github/workflows/mlops-pipeline.yml` runs on every push to `main`.
4. It will process data, train model, build and push both Docker images, then spin up FastAPI service and test the inference API.

***

