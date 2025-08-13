## **Problem Statement** – AI Governance, Fairness, and Auditing in Heart Disease Prediction

You are part of a **health technology company’s AI compliance team**.
Your task is to evaluate and audit a heart disease prediction model to ensure it adheres to **AI governance principles**: **fairness, transparency, and accountability**.

The predictive model aims to determine whether a patient is at risk of heart disease based on medical and lifestyle features. Since healthcare AI systems must comply with **strict ethics and regulatory frameworks** (such as HIPAA, FDA AI/ML Guidance, IEEE and WHO AI principles), you must:

- **Identify and measure bias** in model predictions against a sensitive attribute (in this case, `sex`: Male vs Female patients).
- **Ensure transparency** by using explainability tools (e.g., **SHAP**) to show which features most influence model predictions.
- **Implement AI auditing workflows** by logging performance metrics, bias statistics, and interpretability findings.
- **Produce a compliance and governance report** documenting the model’s evaluation, ready for auditors and oversight boards.

**Business Context:**
Medical diagnosis models that show bias could lead to harmful disparities in care. Auditing the model is essential for meeting ethical obligations, maintaining trust, and ensuring compliance with regulatory standards.

***

### **Dataset Collection Code**

```python
from sklearn.datasets import fetch_openml

# Load the Heart Disease dataset from OpenML
heart_data = fetch_openml(name='heart-disease', version=1, as_frame=True)

# Features (X) and Target (y)
X = heart_data.data
y = heart_data.target.map({'present': 1, 'absent': 0})  # 1 = Disease present, 0 = Absent

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())
```


***

