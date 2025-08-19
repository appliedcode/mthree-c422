# Lab Exercise: Model Lineage Tracking, Packaging, and Serialization


***

## Objective

- Track model lineage across dataset and model versions.
- Package ML model artifacts for reproducibility and deployment.
- Serialize and deserialize ML models in different formats.

***

## Setup

- Python 3.x
- Install packages:

```bash
pip install scikit-learn pandas joblib onnxruntime onnx matplotlib
```


***

# Part 1: Model Lineage Tracking

## Step 1: Simulate Dataset and Model Versions

Generate multiple dataset versions, train models, and record lineage metadata.

```python
import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# Directories
os.makedirs('datasets', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('lineage', exist_ok=True)

# Function to generate synthetic dataset version
def generate_dataset(version):
    data = {
        'feature1': range(100),
        'feature2': [x*version for x in range(100)],
        'label': [i%2 for i in range(100)]
    }
    df = pd.DataFrame(data)
    file_path = f'datasets/dataset_v{version}.csv'
    df.to_csv(file_path, index=False)
    return file_path

# Train model and record lineage metadata
def train_model(version, dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['feature1', 'feature2']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=version)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f'models/model_v{version}.joblib'
    import joblib
    joblib.dump(model, model_path)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Lineage metadata
    lineage = {
        'model_version': f'v{version}',
        'dataset_version': f'v{version}',
        'dataset_path': dataset_path,
        'model_path': model_path,
        'train_time': datetime.now().isoformat(),
        'accuracy': acc,
        'hyperparameters': {'C': 1.0, 'max_iter': 100}
    }
    
    # Save lineage metadata JSON
    lineage_path = f'lineage/lineage_v{version}.json'
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)
    
    print(f"Trained model v{version} with accuracy {acc:.3f} and saved lineage.")
    return model_path, lineage_path

if __name__ == "__main__":
    # Generate and train 3 versions
    for v in range(1, 4):
        dataset = generate_dataset(v)
        train_model(v, dataset)
```


***

## Part 2: Model Packaging

## Step 2: Package Model with Metadata and Dependencies Info

Create a package folder with serialized model, metadata, and requirements.

```python
import shutil

def package_model(version):
    package_dir = f'package/model_v{version}'
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy model file
    model_src = f'models/model_v{version}.joblib'
    model_dst = os.path.join(package_dir, 'model.joblib')
    shutil.copyfile(model_src, model_dst)
    
    # Copy lineage metadata
    lineage_src = f'lineage/lineage_v{version}.json'
    lineage_dst = os.path.join(package_dir, 'metadata.json')
    shutil.copyfile(lineage_src, lineage_dst)
    
    # Write requirements.txt (example)
    with open(os.path.join(package_dir, 'requirements.txt'), 'w') as f:
        f.write("scikit-learn\npandas\njoblib\n")
    
    print(f"Packaged model v{version} into folder '{package_dir}'")

if __name__ == "__main__":
    for v in range(1, 4):
        package_model(v)
```


***

## Part 3: Model Serialization and Deserialization

## Step 3: Serialize using joblib and Validate

```python
import joblib
import numpy as np

def load_and_test_model(version):
    model_path = f'package/model_v{version}/model.joblib'
    model = joblib.load(model_path)
    
    # Sample input (matching training features)
    sample = np.array([[5, 5*version]])
    pred = model.predict(sample)[0]
    print(f"Model v{version} prediction for input {sample.tolist()} is {pred}")

if __name__ == "__main__":
    for v in range(1, 4):
        load_and_test_model(v)
```


***

## Optional: Serialize to ONNX and Inference

```python
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def serialize_to_onnx(version):
    model_path = f'models/model_v{version}.joblib'
    model = joblib.load(model_path)
    
    initial_type = [('input', FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_path = f'package/model_v{version}/model.onnx'
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"Serialized model v{version} to ONNX format.")

def onnx_inference(version):
    onnx_path = f'package/model_v{version}/model.onnx'
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    sample = [[5, 5*version]]
    pred_onx = sess.run(None, {input_name: sample})
    print(f"ONNX model v{version} prediction: {pred_onx}")

if __name__ == "__main__":
    for v in range(1, 4):
        serialize_to_onnx(v)
        onnx_inference(v)
```


***

# Summary

| Part | Description | Key Output |
| :-- | :-- | :-- |
| 1. Model Lineage Tracking | Train models on versions; record lineage JSON | `lineage_v1.json`, `lineage_v2.json` |
| 2. Model Packaging | Bundle model, metadata, requirements | Packaged folders `package/model_vX` |
| 3. Model Serialization | Serialize with joblib (and optional ONNX) | Serialized models and ONNX files |


***

# How to run

1. Save the code blocks as separate Python scripts or combine logically.
2. Run Part 1 script first to generate datasets and train models with lineage.
3. Run Part 2 script to package models.
4. Run Part 3 scripts to test deserialization and optional ONNX integration.

***
