# Lab Exercise: Implementing Automated Testing Strategies for ML Project with GitHub Actions


***

## Objective

Learn to implement automated unit and integration testing for machine learning code and automate these tests in a GitHub Actions workflow to ensure code reliability and quality.

***

## Use Case Scenario

You are developing a machine learning text classification project. To maintain quality and prevent regressions, you want to write automated tests for your core functions and run these tests automatically on every push or pull request using GitHub Actions.

***

## Prerequisites

- Basic Python programming with familiarity in writing functions.
- Knowledge of testing frameworks like `pytest`.
- Basic understanding of Git, GitHub, and GitHub Actions.

***

## Step 1: Setup GitHub Repository and Project

### 1. Create GitHub repo

- Create a repo named `ml-testing-ci`.
- Clone locally:

```bash
git clone https://github.com/your-username/ml-testing-ci.git
cd ml-testing-ci
```


### 2. Add ML code with functions to test

Create `ml_model.py` with core ML functions:

```python
def preprocess_text(text):
    # Simple lowercase text cleaning
    return text.lower().strip()

def predict_label(text):
    # Dummy predict: spam if contains 'buy', else ham
    if 'buy' in text.lower():
        return 'spam'
    else:
        return 'ham'
```


***

### 3. Add tests using pytest

Create `test_ml_model.py`:

```python
from ml_model import preprocess_text, predict_label

def test_preprocess_text():
    raw_text = "  Hello World!  "
    clean_text = preprocess_text(raw_text)
    assert clean_text == "hello world!"

def test_predict_label_spam():
    spam_text = "Buyers are waiting"
    assert predict_label(spam_text) == 'spam'

def test_predict_label_ham():
    ham_text = "How are you?"
    assert predict_label(ham_text) == 'ham'
```


***

### 4. Create requirements.txt

```
pytest
```


***

### 5. Commit and push code

```bash
git add ml_model.py test_ml_model.py requirements.txt
git commit -m "Add ML functions and pytest tests"
git push origin main
```


***

## Step 2: Setup GitHub Actions Workflow for Automated Testing

### 1. Create workflow directory and file

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/ml-testing.yml` with:

```yaml
name: ML Code Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests with pytest
      run: |
        pytest --maxfail=1 --disable-warnings -q
```


***

### 2. Commit and push workflow

```bash
git add .github/workflows/ml-testing.yml
git commit -m "Add GitHub Actions workflow for automated testing"
git push origin main
```


***

## Step 3: Verify Automated Testing

- Go to your GitHub repo Actions tab.
- See the workflow triggered by your push or PR.
- Confirm all tests pass.
- Try breaking a test and pushing to watch the workflow fail as expected.

***

## Optional Enhancements

- Add code coverage reporting with `pytest-cov`.
- Set up tests to run in multiple Python versions.
- Add unit tests for data loading, model training, and inference logic.
- Configure required status checks on pull requests to enforce passing tests before merging.

***

## Deliverables Checklist

- GitHub repo with:
    - `ml_model.py` containing ML functions.
    - `test_ml_model.py` with pytest test cases.
    - `requirements.txt` including pytest.
    - `.github/workflows/ml-testing.yml` automating tests on push.
- Evidence of GitHub Actions test runs passing/failing.

***
