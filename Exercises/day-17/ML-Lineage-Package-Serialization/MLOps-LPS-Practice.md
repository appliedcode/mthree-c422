# Problem Statement: Model Lineage Tracking, Packaging, and Serialization for Credit Card Fraud Detection

## Overview

In this exercise, you will implement an ML workflow focused on credit card fraud detection. Using a synthetic transactional dataset with labeled fraud/non-fraud transactions, you will train and track multiple versions of classification models. You will capture lineage metadata detailing dataset versions, model parameters, training time, and evaluation metrics. The exercise also involves packaging models with metadata and performing model serialization and deserialization for reproducibility and deployment readiness.

***

## Dataset Description and Collection Code

You will generate synthetic credit card transaction data with the following features:


| Feature | Description |
| :-- | :-- |
| transaction_amount | Numeric amount of transaction |
| transaction_time | Time of transaction (in seconds) |
| merchant_category | Categorical category of merchant |
| cardholder_age | Age of the cardholder |
| is_fraud | Target label: 1 = fraud, 0 = legit |

Below is the Python code snippet to generate versions of this dataset as CSV files:

```python
import pandas as pd
import numpy as np
import os

def generate_transaction_dataset(version, save_dir='data'):
    np.random.seed(1000 + version)
    n_samples = 1000
    
    transaction_amount = np.random.exponential(scale=50 + version*10, size=n_samples).round(2)
    transaction_time = np.random.randint(0, 86400, n_samples)  # seconds in a day
    merchant_categories = ['groceries', 'electronics', 'restaurants', 'travel', 'entertainment']
    merchant_category = np.random.choice(merchant_categories, n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    cardholder_age = np.random.normal(40, 12, n_samples).astype(int).clip(18, 90)
    
    # Fraud probability influenced by high amounts and risky categories
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
    print(f"Generated dataset version {version} saved at {file_path}")
    return file_path
```


***

## Task Requirements

1. **Data Collection:**
    - Run the above function to generate multiple dataset versions with differing transaction and fraud patterns.
2. **Model Training and Lineage Tracking:**
    - Train classification models (e.g., logistic regression) for fraud detection on each dataset version.
    - For each model, record lineage metadata including dataset version, model version, training parameters, evaluation metrics (e.g., precision, recall), and training timestamps.
3. **Model Packaging:**
    - Package each model version along with related metadata and a list of dependencies.
4. **Model Serialization:**
    - Serialize models using joblib or pickle.
    - Deserialize models and validate predictions on test samples.
5. **Deliverables:**
    - Code for dataset generation, model training, lineage tracking, packaging, serialization, and evaluation.
    - Saved datasets, model files, lineage JSON files.
    - Packaged model directories containing model, metadata, and requirements.

***

## Objective

- Build strong practices for tracing lineage in fraud detection ML pipelines.
- Package models for easy reproducibility and deployment.
- Demonstrate serialization techniques for model persistence and loading.

***