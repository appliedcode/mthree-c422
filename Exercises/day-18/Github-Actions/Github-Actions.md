# Lab Exercise: Automating Spam Email Classifier Training with GitHub Actions

***

## Objective

Build and automate a spam email classification ML pipeline using GitHub Actions to train, evaluate, and manage the model lifecycle with CI/CD.

***

## Use Case Scenario

Automatically classify incoming emails as "spam" or "not spam" using a machine learning model trained on a labeled email dataset.

***

## Prerequisites

- Experience with Python, Git, and GitHub basics.
- Basic knowledge of machine learning and text classification.
- A GitHub account.

***

## Step 1: Setup GitHub Repository and Files

### 1. Create GitHub Repo

- Go to GitHub and create a new repository called `spam-classifier-ci-cd`.
- Clone it locally:

```bash
git clone https://github.com/your-username/spam-classifier-ci-cd.git
cd spam-classifier-ci-cd
```


### 2. Add Python Scripts and Requirements

Create the following files in the repo folder.

***

### train_spam_classifier.py

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset loaded with {len(data)} rows")
print(f"Columns: {list(data.columns)}")

# Select only the first two columns (v1 and v2) and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

print(f"After cleaning - Dataset shape: {data.shape}")
print(f"Label distribution:\n{data['label'].value_counts()}")
print(f"First few rows:\n{data.head()}")

# Ensure we have enough data
if len(data) < 10:
    raise ValueError(f"Dataset too small: only {len(data)} rows")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'spam_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model trained and saved successfully!")
```


***

### evaluate_spam_classifier.py

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Load dataset (same preprocessing as training)
data = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset loaded with {len(data)} rows")

# Select only the first two columns (v1 and v2) and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

print(f"After cleaning - Dataset shape: {data.shape}")
print(f"Label distribution:\n{data['label'].value_counts()}")

# Train/test split (same as training for consistent evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

print(f"Test set size: {len(X_test)}")

# Load vectorizer and model
try:
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('spam_model.joblib')
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f"\n=== MODEL EVALUATION RESULTS ===")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")

# Detailed classification report
print(f"\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Show some example predictions
print(f"\n=== SAMPLE PREDICTIONS ===")
sample_indices = [0, 1, 2, 3, 4]
for i in sample_indices:
    if i < len(X_test):
        text_sample = X_test.iloc[i][:50] + "..." if len(X_test.iloc[i]) > 50 else X_test.iloc[i]
        print(f"Text: {text_sample}")
        print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
        print("-" * 50)
```


***

### requirements.txt

```
pandas
scikit-learn
joblib
```


***

### 3. Commit and push initial code

```bash
git add .
git commit -m "Initial commit with training and evaluation scripts"
git push origin main
```


***

## Step 2: Create GitHub Actions Workflow

### 1. Create workflow directory and file

In your repo directory:

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/spam-classifier-ci.yml` with the following content:

```yaml
name: Spam Classifier CI/CD

on:
  push:
    branches: [ mlop-pipeline ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: 3.10.18

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r app/requirements.txt

    - name: Download dataset from alternative source
      run: |
        curl -L -o spam.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

    - name: Verify dataset
      run: |
        head -n 10 spam.csv
        echo "Total lines:"
        wc -l spam.csv

    - name: Train Spam Classifier
      run: python app/train_spam_classifier.py

    - name: Evaluate Spam Classifier
      run: python app/evaluate_spam_classifier.py

    - name: Upload model artifact
      uses: actions/upload-artifact@v4
      with:
        name: spam-model
        path: |
          spam_model.joblib
          vectorizer.joblib
```


***

### 2. Commit and push workflow

```bash
git add .github/workflows/spam-classifier-ci.yml
git commit -m "Add GitHub Actions workflow for training and evaluation"
git push origin main
```


***

## Step 3: Verify Workflow Execution

1. Go to your GitHub repository page.
2. Click on the **Actions** tab.
3. Check the workflow run triggered by your push.
4. Ensure all steps complete successfully.
5. Check uploaded artifacts for the model files.

***

## Optional Extensions

- Add unit tests for your train and evaluate scripts, and integrate testing into the workflow.
- Configure notifications on workflow success/failure (Slack, email).
- Automate deployment of the model to a cloud service or API.
- Log metrics to a file and upload as artifacts for tracking.

***

## Deliverables Checklist

- GitHub repository with:
    - `train_spam_classifier.py`
    - `evaluate_spam_classifier.py`
    - `requirements.txt`
    - `.github/workflows/spam-classifier-ci.yml`
- Evidence of successful GitHub Actions workflow runs with model artifacts.

***
