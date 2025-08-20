# Lab Exercise: Automated Testing and Code Quality Checks for Wine Quality ML Project with GitHub Actions


***

## Objective

Implement automated unit tests and enforce code quality standards for a wine quality prediction ML project. Automate these checks in a CI pipeline using GitHub Actions.

***

## Step 1: Setup GitHub Repository and Code

### 1. Create GitHub repository

- Create a repository named `wine-quality-ci`.
- Clone it locally:

```bash
git clone https://github.com/your-username/wine-quality-ci.git
cd wine-quality-ci
```


***

### 2. Add ML Code: `wine_quality_ml.py`

```python
import pandas as pd

def preprocess_data(df):
    """
    Preprocess the Wine Quality dataset by normalizing chemical features.
    """
    df = df.copy()
    features = df.columns.difference(['quality'])
    
    for col in features:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df

def predict_quality(df):
    """
    Dummy wine quality prediction:
    Predict 'High' if alcohol >= 0.6 and pH between 0.4 and 0.6 after normalization, else 'Low'.
    Assumes preprocessing has been done.
    """
    required_cols = {'alcohol', 'pH'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required_cols}")

    predictions = []
    for _, row in df.iterrows():
        if row['alcohol'] >= 0.6 and 0.4 <= row['pH'] <= 0.6:
            predictions.append('High')
        else:
            predictions.append('Low')
    return predictions
```


***

### 3. Add Unit Tests: `test_wine_quality_ml.py`

```python
import pytest
import pandas as pd
from wine_quality_ml import preprocess_data, predict_quality

def test_preprocess_data_normalizes():
    data = {
        'alcohol': [10, 14],
        'pH': [3.0, 3.8],
        'quality': [5, 7]
    }
    df = pd.DataFrame(data)
    processed = preprocess_data(df)
    
    assert all(col in processed.columns for col in ['alcohol', 'pH', 'quality'])
    for col in ['alcohol', 'pH']:
        assert processed[col].min() == 0
        assert processed[col].max() == 1

def test_predict_quality_returns_expected():
    data = {
        'alcohol': [0.7, 0.5],
        'pH': [0.5, 0.7]
    }
    df = pd.DataFrame(data)
    predictions = predict_quality(df)
    assert predictions == ['High', 'Low']

def test_predict_quality_missing_columns():
    df = pd.DataFrame({'alcohol': [0.7]})
    with pytest.raises(ValueError):
        predict_quality(df)
```


***

### 4. Create `requirements.txt`

```
pandas
pytest
flake8
black
```


***

### 5. Commit and push all files

```bash
git add .
git commit -m "Add wine quality ML code, tests and requirements"
git push origin main
```


***

## Step 2: Configure GitHub Actions Workflow

### 1. Create workflow directory and workflow file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/ci.yml`:

```yaml
name: Wine Quality ML CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint_test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Setup Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run linting with flake8
      run: flake8 --max-line-length=88 .

    - name: Check code formatting with black
      run: black --check .

    - name: Run pytest
      run: pytest --maxfail=1 --disable-warnings -q
```


***

### 2. Commit and push workflow

```bash
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions workflow for testing and code quality"
git push origin main
```


***

## Step 3: Validate the CI Pipeline

- Go to your GitHub repo’s **Actions** tab.
- Observe workflow runs triggered by pushes or pull requests.
- Confirm all steps (lint, formatting check, tests) pass.
- Optionally test failure by introducing lint or formatting errors, or by failing a test case.

***

## Step 4: Run Locally (Optional)

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run tests:

```bash
pytest
```

- Run lint:

```bash
flake8 .
```

- Run code format check:

```bash
black --check .
```


***

## Summary

- `wine_quality_ml.py` contains data preprocessing and dummy prediction logic.
- `test_wine_quality_ml.py` holds unit tests validating preprocessing and prediction.
- `requirements.txt` lists necessary libraries including testing and code quality tools.
- GitHub Actions workflow runs linting, formatting checks, and tests automatically.
- This setup helps maintain high-quality, reliable ML code integrated via CI/CD.

***
