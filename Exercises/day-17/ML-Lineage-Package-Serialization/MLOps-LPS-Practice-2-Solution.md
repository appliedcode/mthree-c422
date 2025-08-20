# Solution 
Here is a complete Python solution for the **Employee Attrition Prediction** problem with model lineage tracking, packaging, and serialization. It covers dataset generation, training, lineage metadata capture, model packaging, and serialization testing.

***

# Directory structure to create before running:

```
employee_attrition_metadata/
├── data/
├── models/
├── lineage/
├── package/
```

Create these folders inside your workspace.

***

# Script: run_employee_attrition_metadata.py

```python
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime
import joblib
import shutil

# Directories setup
DATA_DIR = 'data'
MODEL_DIR = 'models'
LINEAGE_DIR = 'lineage'
PACKAGE_DIR = 'package'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LINEAGE_DIR, exist_ok=True)
os.makedirs(PACKAGE_DIR, exist_ok=True)

# Dataset generation function
def generate_attrition_dataset(version, save_dir=DATA_DIR):
    np.random.seed(1234 + version)
    n_samples = 800
    
    age = np.random.randint(22, 60, size=n_samples)
    job_roles = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']
    job_role = np.random.choice(job_roles, size=n_samples, p=[0.3, 0.35, 0.1, 0.15, 0.1])
    monthly_income = np.random.normal(5000 + version*500, 1500, n_samples).clip(2000, 15000)
    years_at_company = np.random.randint(0, 20, n_samples)
    
    attrition_prob = (
        (monthly_income < (4000 + version * 200)).astype(float) * 0.35 +
        (years_at_company < 3).astype(float) * 0.3 +
        ((job_role == 'Sales') | (job_role == 'Marketing')).astype(float) * 0.2
    )
    attrition_prob = np.clip(attrition_prob, 0, 1)
    attrition = np.random.binomial(1, attrition_prob)
    
    df = pd.DataFrame({
        'age': age,
        'job_role': job_role,
        'monthly_income': monthly_income.round(2),
        'years_at_company': years_at_company,
        'attrition': attrition
    })
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = f'{save_dir}/employee_attrition_v{version}.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated dataset version {version} at {file_path}")
    return file_path

# Train model and save lineage metadata
def train_and_track_lineage(version, dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['age', 'job_role', 'monthly_income', 'years_at_company']]
    y = df['attrition']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), ['job_role']),
    ], remainder='passthrough')
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression(max_iter=200))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=version)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    model_path = f'{MODEL_DIR}/attrition_model_v{version}.joblib'
    joblib.dump(pipeline, model_path)
    
    lineage = {
        'model_version': f'v{version}',
        'dataset_version': f'v{version}',
        'dataset_path': dataset_path,
        'model_path': model_path,
        'train_time': datetime.now().isoformat(),
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec
        },
        'model_info': {
            'type': 'LogisticRegression',
            'max_iter': 200,
            'preprocessing': 'OneHotEncoding for job_role'
        }
    }
    
    lineage_path = f'{LINEAGE_DIR}/lineage_v{version}.json'
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)
    
    print(f"Trained model v{version} -- Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
    print(f"Saved lineage metadata to {lineage_path}")
    return model_path, lineage_path

# Package model with metadata and dependencies
def package_model(version):
    package_path = f'{PACKAGE_DIR}/model_v{version}'
    os.makedirs(package_path, exist_ok=True)
    
    model_file = f'{MODEL_DIR}/attrition_model_v{version}.joblib'
    metadata_file = f'{LINEAGE_DIR}/lineage_v{version}.json'
    
    shutil.copyfile(model_file, os.path.join(package_path, 'model.joblib'))
    shutil.copyfile(metadata_file, os.path.join(package_path, 'metadata.json'))
    
    with open(os.path.join(package_path, 'requirements.txt'), 'w') as f:
        f.write("scikit-learn\npandas\nnumpy\njoblib\n")
    
    print(f"Packaged model v{version} at {package_path}")

# Load packaged model and test prediction
def test_packaged_model(version):
    model_path = f'{PACKAGE_DIR}/model_v{version}/model.joblib'
    model = joblib.load(model_path)
    
    sample = pd.DataFrame({
        'age': [^30],
        'job_role': ['Engineering'],
        'monthly_income': ,
        'years_at_company': [^1]
    })
    
    pred = model.predict(sample)
    print(f"Predicted attrition for sample employee using model v{version}: {pred} (1=Will leave, 0=Will stay)")

if __name__ == "__main__":
    # Generate datasets and train models versions 1 to 3
    for v in range(1, 4):
        dataset_file = generate_attrition_dataset(v)
        train_and_track_lineage(v, dataset_file)
        package_model(v)
    
    # Test loading and prediction using version 2 packaged model
    test_packaged_model(2)
```


***

# How to run

1. Inside your working directory, create folders:
```
data, models, lineage, package
```

2. Save the above script as `run_employee_attrition_metadata.py`.
3. Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

4. Run the script:
```bash
python run_employee_attrition_metadata.py
```


***

# What this script accomplishes

- Generates synthetic employee attrition datasets (3 versions).
- Trains logistic regression classifiers while capturing metrics and lineage metadata.
- Saves lineage info as JSON files.
- Packages models with metadata and requirements.
- Deserializes packaged model and tests sample prediction.

***
