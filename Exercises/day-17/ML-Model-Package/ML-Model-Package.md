# Lab Exercise: Building and Packaging a Machine Learning Model for Local Deployment

## Objective

In this lab, you will learn how to build a simple machine learning model, package it as a Python package with all dependencies, and run it locally to make predictions. You will gain hands-on experience creating reusable ML model packages for local deployment and usage.

## Lab Overview

1. Prepare a dataset and train a machine learning model.
2. Build a Python package to wrap your ML model.
3. Create a CLI or API to interact with the packaged model.
4. Install your package locally and run prediction commands.
5. (Optional) Package your model with MLflow and serve locally.

***

## Step 1: Train and Save a Simple Model

a) Create a Python script `train_model.py` that:

- Loads the built-in iris dataset from scikit-learn.
- Trains a logistic regression classifier.
- Saves the trained model to disk using `joblib`.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, 'iris_model.joblib')
print("Model trained and saved as iris_model.joblib")
```

Run this script in terminal:

```bash
python train_model.py
```


***

## Step 2: Create a Python Package Structure

Organize your project directory as:

```
ml_model_package/
├── ml_model/
│   ├── __init__.py
│   ├── model.py
│   └── predict.py
├── setup.py
└── README.md
```


***

## Step 3: Build Your ML Model Package Code

### ml_model/model.py

This module will load the saved model and provide a prediction function.

```python
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'iris_model.joblib')
model = joblib.load(MODEL_PATH)

def predict(input_features):
    """
    input_features: list or numpy array of 4 float features
    Returns predicted class label.
    """
    x = np.array(input_features).reshape(1, -1)
    pred = model.predict(x)
    return int(pred[0])
```


***

### ml_model/predict.py

A command-line interface script to take user input and perform prediction.

```python
import sys
from .model import predict

def main():
    if len(sys.argv) != 5:
        print("Usage: python -m ml_model.predict feature1 feature2 feature3 feature4")
        sys.exit(1)

    features = list(map(float, sys.argv[1:5]))
    prediction = predict(features)
    print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
```


***

## Step 4: Setup Packaging

Create `setup.py` in your root directory:

```python
from setuptools import setup, find_packages

setup(
    name='ml_model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'joblib'
    ],
    entry_points={
        'console_scripts': [
            'ml-predict=ml_model.predict:main'
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.joblib']
    }
)
```


***

## Step 5: Install and Test Your Package Locally

In terminal, from your package root directory:

```bash
pip install -e .
```

This installs your model package in editable mode.

Now you can run:

```bash
ml-predict 5.1 3.5 1.4 0.2
```

You should see the predicted class label printed.

***

## Optional Step 6: Serve Model via Local Flask API

You may extend the package by creating a simple FastAPI or Flask app to serve predictions locally.

***

## Summary

In this lab, you:

- Trained and saved an ML model.
- Wrapped model loading and prediction in a reusable Python package.
- Created CLI for local usage.
- Installed and used your package locally.

This process simulates the foundation for ML model deployment allowing easy integration into larger systems or pipelines.

***

## Extras

- Experiment with packaging your model with MLflow and serve it locally.
- Add input validation and error handling to your prediction scripts.
- Explore unit testing your model API functions.

***