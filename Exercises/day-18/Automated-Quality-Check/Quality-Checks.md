# Lab Exercise: Automated Testing and Code Quality Checks for ML Project using GitHub Actions


***

## Objective

Learn to implement automated unit tests and code quality checks (linting and formatting) for an ML project using GitHub Actions for continuous integration.

***

## Use Case Scenario

You are developing a machine learning sentiment analysis project. To ensure your code is robust and maintainable, you want to automatically run unit tests, check code formatting with `black`, and perform linting with `flake8` on every code push or pull request.

***

## Prerequisites

- Basic Python familiarity, including testing with `pytest`.
- Understanding of linting and code formatting tools like `flake8` and `black`.
- GitHub and GitHub Actions basics.

***

## Step 1: Setup GitHub Repository and Code

### 1. Create GitHub repository

- Create a repo called `ml-sentiment-quality-ci`.
- Clone it locally:

```bash
git clone https://github.com/your-username/ml-sentiment-quality-ci.git
cd ml-sentiment-quality-ci
```


***

### 2. Add ML code: `sentiment_analysis.py`

```python
def preprocess(text):
    """Lowercase and strip whitespace from text."""
    return text.lower().strip()

def predict_sentiment(text):
    """Dummy prediction: positive if 'good' in text, else negative."""
    text = preprocess(text)
    if 'good' in text:
        return 'positive'
    return 'negative'
```


***

### 3. Add test cases: `test_sentiment_analysis.py`

```python
from sentiment_analysis import preprocess, predict_sentiment

def test_preprocess():
    assert preprocess("  Hello World! ") == "hello world!"

def test_predict_sentiment_positive():
    assert predict_sentiment("This is good") == 'positive'

def test_predict_sentiment_negative():
    assert predict_sentiment("This is bad") == 'negative'
```


***

### 4. Create requirements file: `requirements.txt`

```
pytest
flake8
black
```


***

### 5. Commit and push

```bash
git add .
git commit -m "Add sentiment analysis code, tests, and requirements"
git push origin main
```


***

## Step 2: Create GitHub Actions Workflow

### 1. Create workflow directory and file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/ci.yml` with the following content:

```yaml
name: ML Project CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint_test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run flake8 linting
      run: flake8 --max-line-length=88 .

    - name: Run black check
      run: black --check .

    - name: Run pytest tests
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

## Step 3: Verify CI Pipeline

1. Go to GitHub repository’s **Actions** tab.
2. Look for the triggered workflow that runs on your push.
3. Confirm that linting, formatting checks, and tests pass.
4. Try introducing a style violation or failing test and push to observe the workflow failure.

***

## Optional Enhancements

- Enable auto-formatting with `black` on push using a GitHub Action.
- Add coverage reporting (`pytest-cov`) and upload to services like Codecov.
- Enforce branch protection rules requiring successful CI checks before merging.

***

## Deliverables Checklist

- GitHub repo including:
    - `sentiment_analysis.py`
    - `test_sentiment_analysis.py`
    - `requirements.txt`
    - `.github/workflows/ci.yml`
- Evidence of passing GitHub Actions runs with testing and code quality steps.
- Demonstration of failure scenarios (optional).

***

## Summary

This exercise teaches how to integrate automated ML unit testing and code quality checks into a GitHub Actions CI/CD pipeline, boosting development rigor and code maintainability.

***
