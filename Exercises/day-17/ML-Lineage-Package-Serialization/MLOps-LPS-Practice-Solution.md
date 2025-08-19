# Solution 
Here is a complete end-to-end Python solution for the **Credit Card Fraud Detection** problem with model lineage tracking, packaging, and serialization. It includes dataset generation, model training with lineage metadata, model packaging, and serialization validation.

***

# Directory structure to create before running:

```
fraud_detection_metadata/
├── data/
├── models/
├── lineage/
├── package/
```

Create these folders inside your workspace.

***

# Script: run_fraud_detection_metadata.py

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
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
import joblib
import shutil

# Setup directories
DATA_DIR = 'data'
MODEL_DIR = 'models'
LINEAGE_DIR = 'lineage'
PACKAGE_DIR = 'package'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LINEAGE_DIR, exist_ok=True)
os.makedirs(PACKAGE_DIR, exist_ok=True)

# Dataset generation function
def generate_transaction_dataset(version, save_dir=DATA_DIR):
    np.random.seed(1000 + version)
    n_samples = 1000
    
    transaction_amount = np.random.exponential(scale=50 + version*10, size=n_samples).round(2)
    transaction_time = np.random.randint(0, 86400, n_samples)  # seconds in a day
    merchant_categories = ['groceries', 'electronics', 'restaurants', 'travel', 'entertainment']
    merchant_category = np.random.choice(merchant_categories, n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    cardholder_age = np.random.normal(40, 12, n_samples).astype(int).clip(18, 90)
    
    fraud_prob = (
        (transaction_amount > (70 + version * 5)).astype(float) * 0.3 +
        (merchant_category == 'electronics').astype(float) * 0.25 +
        (merchant_category == 'travel').astype(float) * 0.2 +
        (cardholder_age < 25).astype(float) * 0.15
    )
    fraud_prob = np.clip(fraud_prob, 0, 1)
    is_fraud = np.random.binomial(1, fraud_prob)
    
    df = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'transaction_time': transaction_time,
        'merchant_category': merchant_category,
        'cardholder_age': cardholder_age,
        'is_fraud': is_fraud
    })
    
    os.makedirs(save_dir, exist_ok=True)
    file_path = f'{save_dir}/credit_card_transactions_v{version}.csv'
    df.to_csv(file_path, index=False)
    print(f"Generated dataset version {version} saved to {file_path}")
    return file_path

# Train model and track lineage
def train_and_track_lineage(version, dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['transaction_amount', 'transaction_time', 'merchant_category', 'cardholder_age']]
    y = df['is_fraud']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), ['merchant_category'])
    ], remainder='passthrough')
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression(max_iter=200))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=version)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    model_path = f'{MODEL_DIR}/fraud_model_v{version}.joblib'
    joblib.dump(pipeline, model_path)
    
    lineage = {
        'model_version': f'v{version}',
        'dataset_version': f'v{version}',
        'dataset_path': dataset_path,
        'model_path': model_path,
        'train_time': datetime.now().isoformat(),
        'precision': precision,
        'recall': recall,
        'hyperparameters': {
            'model_type': "LogisticRegression",
            'max_iter': 200,
            'preprocessing': 'OneHotEncoding for merchant_category'
        }
    }
    
    lineage_path = f'{LINEAGE_DIR}/lineage_v{version}.json'
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)
    
    print(f"Trained model v{version} - Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(f"Lineage metadata saved to {lineage_path}")
    return model_path, lineage_path

# Package model with metadata and requirements
def package_model(version):
    package_dir = f'{PACKAGE_DIR}/model_v{version}'
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy model file
    model_src = f'{MODEL_DIR}/fraud_model_v{version}.joblib'
    model_dst = os.path.join(package_dir, 'model.joblib')
    shutil.copyfile(model_src, model_dst)
    
    # Copy lineage metadata
    lineage_src = f'{LINEAGE_DIR}/lineage_v{version}.json'
    lineage_dst = os.path.join(package_dir, 'metadata.json')
    shutil.copyfile(lineage_src, lineage_dst)
    
    # Write requirements.txt
    with open(os.path.join(package_dir, 'requirements.txt'), 'w') as f:
        f.write("scikit-learn\npandas\nnumpy\njoblib\n")
    
    print(f"Packaged model v{version} into {package_dir}")

# Deserialize and test model prediction
def test_model(version):
    model_path = f'{PACKAGE_DIR}/model_v{version}/model.joblib'
    model = joblib.load(model_path)
    
    sample = pd.DataFrame({
        'transaction_amount': [120.0],
        'transaction_time': [45000],
        'merchant_category': ['electronics'],
        'cardholder_age': 
    })
    pred = model.predict(sample)
    print(f"Deserialized model v{version} prediction for sample input: {pred} (1=Fraud, 0=Legit)")

if __name__ == "__main__":
    # Generate datasets and train models for versions 1 to 3
    for v in range(1, 4):
        dataset_file = generate_transaction_dataset(v)
        train_and_track_lineage(v, dataset_file)
        package_model(v)
    
    # Test deserialization and prediction for model version 2
    test_model(2)
```


***

# How to Run

1. Create the folders: `data`, `models`, `lineage`, and `package` inside your working directory.
2. Save the script above as `run_fraud_detection_metadata.py`.
3. Install required packages:
```bash
pip install pandas numpy scikit-learn joblib
```

4. Run the script:
```bash
python run_fraud_detection_metadata.py
```


***

# What This Script Does

- Generates three synthetic credit card transaction datasets with different distributions.
- Trains logistic regression models for fraud detection on each dataset version.
- Records lineage metadata including dataset and model paths, training time, and evaluation metrics.
- Packages each model with metadata and requirements files.
- Demonstrates deserialization and prediction on a sample input with a packaged model.

***