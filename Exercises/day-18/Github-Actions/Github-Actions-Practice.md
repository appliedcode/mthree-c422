# Lab Exercise: Automating House Price Prediction Model with GitHub Actions


***

## Objective

Build and automate a regression ML pipeline that predicts house prices using the Boston Housing dataset, leveraging GitHub Actions for CI/CD of model training and evaluation.

***

## Use Case Scenario

You are working for a real estate company that wants to predict housing prices based on various features like number of rooms, property age, and location factors to assist buyers and sellers.

***

## Prerequisites

- Basic knowledge of Python, Git, and GitHub.
- Familiarity with ML regression concepts.
- GitHub account.

***

## Step 1: Setup GitHub Repository and Files

### 1. Create GitHub Repo

- Create a GitHub repository named `house-price-prediction-ci`.
- Clone it locally:

```bash
git clone https://github.com/your-username/house-price-prediction-ci.git
cd house-price-prediction-ci
```


### 2. Add Python Scripts and Requirements

Create the following files in your repo folder.

***

### train_house_price.py

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Boston housing dataset
data = load_boston()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'house_price_model.joblib')
```


***

### evaluate_house_price.py

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = load_boston()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load('house_price_model.joblib')

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
```


***

### requirements.txt

```
scikit-learn
joblib
```


***

### 3. Commit and push initial code

```bash
git add .
git commit -m "Initial commit with house price training and evaluation"
git push origin main
```


***

## Step 2: Write GitHub Actions Workflow

### 1. Create workflow directory and file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/house-price-ci.yml` with the following content:

```yaml
name: House Price Model CI/CD

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repo
      uses: actions/checkout@v3

    - name: Setup Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Train house price model
      run: python train_house_price.py

    - name: Evaluate house price model
      run: python evaluate_house_price.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v3
      with:
        name: house-price-model
        path: house_price_model.joblib
```


***

### 2. Commit and push workflow file

```bash
git add .github/workflows/house-price-ci.yml
git commit -m "Add GitHub Actions workflow for house price prediction"
git push origin main
```


***

## Step 3: Verify Workflow Execution

1. Navigate to the GitHub repo online.
2. Click the **Actions** tab.
3. Confirm the workflow triggered on your push completes successfully.
4. Download the model artifact from the workflow run.

***

## Optional Enhancements

- Add unit tests for data preprocessing and model prediction.
- Include Slack/email notifications on build status.
- Extend to deploy the model to a REST API or cloud platform.
- Log metrics to file and upload alongside artifacts.

***

## Deliverables Checklist

- GitHub repo with:
    - `train_house_price.py`
    - `evaluate_house_price.py`
    - `requirements.txt`
    - `.github/workflows/house-price-ci.yml`
- Proof of GitHub Actions workflow run with successful model training artifact.

***
