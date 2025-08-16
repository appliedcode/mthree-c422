# Practice: Loan Default Risk Prediction with DVC \& Git LFS


***

## 1. Project Setup

Create folder structure:

```
loan-dvc-lfs-ml/
  ├── data/
  ├── models/
  ├── src/
  │    ├── train.py
  │    ├── api.py
  ├── requirements.txt
  ├── Dockerfile
  ├── params.yaml
  ├── .gitattributes
  ├── README.md
```


***

## 2. `requirements.txt`

```txt
flask
pandas
scikit-learn
joblib
dvc
gitpython
```


***

## 3. Dataset

- Download the Kaggle Loan Prediction dataset CSV (`train.csv`) manually from:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- Place inside `data/train.csv`.

***

## 4. Initialize Git, DVC, and Git LFS

```bash
cd loan-dvc-lfs-ml

git init
dvc init
git lfs install
```

Track dataset with DVC and Git LFS:

```bash
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add raw dataset tracked with DVC"

git lfs track "*.csv"
git add .gitattributes
git commit -m "Track CSV files with Git LFS"
```


***

## 5. `params.yaml` (for experiment parameters)

```yaml
train:
  test_size: 0.2
  random_state: 42
  n_estimators: 100
```


***

## 6. `src/train.py` — Data prep, training, and saving model with DVC tracking

```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import yaml
import os

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load data
df = pd.read_csv("data/train.csv")

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode(), inplace=True)
df['Dependents'].fillna(df['Dependents'].mode(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode(), inplace=True)

# Encode categorical columns
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Encode target
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
)

# Train model
model = RandomForestClassifier(n_estimators=params['train']['n_estimators'], random_state=params['train']['random_state'])
model.fit(X_train, y_train)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("Model training complete and saved to models/model.joblib")
```

Track the model artifact with DVC:

```bash
dvc add models/model.joblib
git add models/model.joblib.dvc .gitignore
git commit -m "Add trained model tracked with DVC"
```


***

## 7. `src/api.py` — Flask REST API to serve the model

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/model.joblib")

# Model input columns expected during training
MODEL_COLUMNS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

@app.route("/predict", methods=["POST"])
def predict():
    input_json = request.get_json()
    input_df = pd.DataFrame([input_json])

    # Add missing columns with default 0
    for col in MODEL_COLUMNS:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[MODEL_COLUMNS]

    prediction = model.predict(input_df)[0]
    return jsonify({"loan_status_prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```


***

## 8. `Dockerfile` — Containerize training and serving application

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "src/api.py"]
```


***

## 9. How to Build and Run

- Build Docker image:

```bash
docker build -t loan-dvc-lfs-app .
```

- Run container:

```bash
docker run -p 5000:5000 loan-dvc-lfs-app
```


***

## 10. Test API

```python
import requests

sample_input = {
    "Gender": 1,
    "Married": 0,
    "Dependents": 0,
    "Education": 1,
    "Self_Employed": 0,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 200,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2
}

resp = requests.post("http://localhost:5000/predict", json=sample_input)
print(resp.json())
```


***

## Summary

- You have versioned your dataset and model artifact using **DVC** and **Git LFS**.
- Local API serving with Flask inside a Docker container.
- Full lifecycle from raw data to deployment with reproducibility and collaboration support.

***

This practice setup mirrors industry best practices for managing large ML projects with growing datasets and evolving models.

