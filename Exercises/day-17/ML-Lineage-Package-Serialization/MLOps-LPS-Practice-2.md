# Problem Statement: Model Lineage Tracking, Packaging, and Serialization for Employee Attrition Prediction

## Overview

In this exercise, you will build an ML workflow for predicting employee attrition using a synthetic HR dataset. You are tasked with training multiple versions of classification models to predict whether an employee will leave the company. You will track model lineage by recording dataset versions, training parameters, model versions, and evaluation metrics. Further, you will package models alongside metadata and dependencies and ensure model serialization/deserialization for reproducibility and deployment.

***

## Dataset Description and Collection Code

The dataset includes the following features:


| Feature | Description |
| :-- | :-- |
| age | Employee age |
| job_role | Categorical job role |
| monthly_income | Employee’s monthly income |
| years_at_company | Number of years with the company |
| attrition | Target label: 1 = left, 0 = stayed |

Below is Python code to generate synthetic versions of this dataset saved as CSV files:

```python
import pandas as pd
import numpy as np
import os

def generate_attrition_dataset(version, save_dir='data'):
    np.random.seed(1234 + version)
    n_samples = 800
    
    age = np.random.randint(22, 60, size=n_samples)
    job_roles = ['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']
    job_role = np.random.choice(job_roles, size=n_samples, p=[0.3, 0.35, 0.1, 0.15, 0.1])
    monthly_income = np.random.normal(5000 + version*500, 1500, n_samples).clip(2000, 15000)
    years_at_company = np.random.randint(0, 20, n_samples)
    
    # Attrition probability influenced by low income, short tenure, and certain roles
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
    print(f"Generated employee attrition dataset version {version} saved to {file_path}")
    return file_path
```


***

## Task Requirements

1. **Data Collection:**
    - Use the provided function to generate multiple versions of the employee attrition dataset.
2. **Model Training and Lineage Tracking:**
    - Train classification models (e.g., logistic regression) for attrition prediction on each dataset version.
    - Capture lineage metadata: dataset/model versions, model parameters, training time, and evaluation metrics such as accuracy, precision, and recall.
3. **Model Packaging:**
    - Package trained models with related metadata and dependency information.
4. **Model Serialization:**
    - Serialize models using joblib or pickle.
    - Deserialize models to confirm prediction correctness on sample employee profiles.
5. **Deliverables:**
    - Scripts for dataset generation, model training, lineage tracking, packaging, serialization, and evaluation.
    - Saved datasets, model files, lineage metadata JSON files.
    - Packaged directories containing models, metadata, and requirements.

***

## Objective

- Develop skills in managing ML model lifecycle for HR attrition prediction.
- Emphasize reproducibility using lineage tracking.
- Learn to package and serialize models for production readiness.

***
