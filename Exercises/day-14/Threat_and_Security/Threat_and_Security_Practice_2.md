## **Problem Statement** – AI Security, Threat Simulation, and Privacy-Preserving AI in Healthcare Diagnosis

You are part of the **AI ethics and compliance team** in a healthcare analytics company.
Your current project involves building a **machine learning model** to predict whether a patient is likely to be diagnosed with diabetes based on clinical measurements (e.g., glucose level, BMI, age).

**Since medical data is highly sensitive**, you must ensure the model follows **AI governance principles** — with a focus on **security, privacy, fairness, and transparency**.

Your responsibilities for this model are to:

1. **Dataset Handling \& Preprocessing**
    - Load and explore the **Pima Indians Diabetes dataset**.
    - Prepare the dataset for binary classification: `1 = diabetes`, `0 = no diabetes`.
2. **Threat Simulation**
    - Simulate a **data poisoning attack** by flipping a small fraction of outcome labels to test model robustness against adversarial input.
3. **Baseline Model Training**
    - Train and evaluate a classification model (e.g., Random Forest or Logistic Regression) on the poisoned data.
4. **Privacy Preservation**
    - Implement a **privacy-preserving step**, such as removing high-risk attributes (`Age`, `BMI`) or adding noise to sensitive features before model training.
    - Compare performance before/after privacy transformation.
5. **Auditing \& Transparency**
    - Use **SHAP explainability** to interpret model behavior and verify whether sensitive attributes have undue influence on predictions.
    - Maintain an **audit log** recording performance metrics, parameters, and threats simulated.
6. **Governance Reporting**
    - Produce a governance-friendly compliance report summarizing threat simulation results, privacy measures, model accuracy trade-offs, and interpretability insights.
    - Ensure the report would satisfy healthcare AI regulatory expectations (e.g., HIPAA, WHO guidance, ISO health AI standards).

***

### **Dataset Collection Code**

```python
import pandas as pd
from sklearn.datasets import fetch_openml

# Load Pima Indians Diabetes dataset from OpenML
pima = fetch_openml(name='diabetes', version=1, as_frame=True)

# Features (X) and Target (y)
X = pima.data
y = pima.target.map({'tested_positive': 1, 'tested_negative': 0})

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())
X.head()
```


***