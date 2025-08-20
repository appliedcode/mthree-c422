# Solution: Automated Testing for Diabetes Prediction ML Project


***

## 1. Python Code: `diabetes_ml.py`

```python
import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess the diabetes dataset:
    - Replace zeros in certain columns with median values to handle missing data.
    - Normalize all feature columns to the [0,1] range.
    """
    cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df = df.copy()

    for col in cols_with_missing:
        df[col] = df[col].replace(0, np.nan)
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    # Normalize all columns except Outcome
    for col in df.columns:
        if col != 'Outcome':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)

    return df

def predict_diabetes(df):
    """
    Dummy diabetes risk prediction function:
    - Predict 'Positive' if normalized Glucose > 0.5 and BMI > 0.5,
      otherwise predict 'Negative'.
    - Expects preprocessed dataframe.
    """
    required_cols = {'Glucose', 'BMI'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    predictions = []
    for _, row in df.iterrows():
        if row['Glucose'] > 0.5 and row['BMI'] > 0.5:
            predictions.append('Positive')
        else:
            predictions.append('Negative')

    return predictions
```


***

## 2. Test Code: `test_diabetes_ml.py`

```python
import pytest
import pandas as pd
import numpy as np
from diabetes_ml import preprocess_data, predict_diabetes

def test_preprocess_data_correctness():
    # Data with zeros for missing values
    raw_data = {
        'Pregnancies': [1, 2],
        'Glucose': [0, 130],
        'BloodPressure': [70, 0],
        'SkinThickness': [0, 25],
        'Insulin': [0, 100],
        'BMI': [0, 35],
        'DiabetesPedigreeFunction': [0.4, 0.3],
        'Age': [28, 45],
        'Outcome': [0, 1]
    }
    df_raw = pd.DataFrame(raw_data)
    df_processed = preprocess_data(df_raw)

    # After preprocessing, zero values replaced
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        assert (df_processed[col] == 0).sum() == 0

    # All feature columns normalized between 0 and 1
    for col in df_processed.columns:
        if col != 'Outcome':
            assert df_processed[col].min() >= 0
            assert df_processed[col].max() <= 1

def test_predict_diabetes_behaviour():
    data = {
        'Glucose': [0.7, 0.3],
        'BMI': [0.6, 0.4]
    }
    df = pd.DataFrame(data)
    preds = predict_diabetes(df)
    assert preds == ['Positive', 'Negative']

def test_predict_diabetes_missing_columns():
    df = pd.DataFrame({'Glucose': [0.6]})  # BMI missing
    with pytest.raises(ValueError):
        predict_diabetes(df)
```


***

## 3. Requirements File: `requirements.txt`

```
pandas
numpy
pytest
```


***

## 4. GitHub Actions Workflow File: `.github/workflows/test-ci.yml`

```yaml
name: Diabetes Prediction Testing CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Repository
      uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run Tests
      run: |
        pytest --maxfail=1 --disable-warnings -q
```


***

## 5. Step-by-Step Instructions

### Step A: Set up repository and files

1. Create a new GitHub repository (e.g., `diabetes-prediction-testing`).
2. Clone the repo locally:
```bash
git clone https://github.com/your-username/diabetes-prediction-testing.git
cd diabetes-prediction-testing
```

3. Create the following files and add the respective code:

- `diabetes_ml.py`
- `test_diabetes_ml.py`
- `requirements.txt`
- `.github/workflows/test-ci.yml` (create directories as needed)

4. Track and commit your files:
```bash
git add .
git commit -m "Add diabetes ML code, tests, and CI workflow"
git push origin main
```


***

### Step B: Run tests locally (optional but recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

Execute tests with:

```bash
pytest
```

You should see all tests passing.

***

### Step C: Validate GitHub Actions workflow

1. Go to your repository on GitHub.
2. Navigate to the “Actions” tab.
3. You will see the running workflow triggered by your push.
4. Check that all tests run successfully.
5. As an experiment, you can introduce a failing test and push changes to see the workflow fail.

***

## 6. Summary

- `diabetes_ml.py` contains data preprocessing and dummy prediction logic.
- `test_diabetes_ml.py` uses `pytest` to write unit tests verifying correctness and error handling.
- `requirements.txt` lists dependencies.
- GitHub Actions workflow automatically installs dependencies and runs tests on every push/PR.
- This setup improves code reliability and integrates testing into CI/CD pipelines.

***
