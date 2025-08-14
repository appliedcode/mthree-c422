## **Problem Statement**

You are tasked with evaluating and auditing a machine learning–based **credit scoring model** to ensure it complies with **AI governance principles**, including fairness, transparency, and accountability.

The goal is to:

- Detect and measure potential **bias** in model predictions across demographic groups (in this case, `age_group` — younger vs older applicants).
- Provide **model transparency** through explainability tools such as **SHAP** to understand feature influences on decisions.
- Implement **AI auditing steps** by logging performance metrics, bias analysis, and interpretability findings, with proper documentation that could be used for **regulatory compliance** and internal governance processes.

You must train, evaluate, audit, and document the AI model in a way that aligns with **ethical AI adoption** in financial services.

**In this exercise, we will use the publicly available "Credit-G" dataset from OpenML.**

### **Dataset Collection Code**

```python
from sklearn.datasets import fetch_openml

# Load dataset from OpenML
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)

# Features (X) and Target (y)
X = credit_data.data
y = credit_data.target

# Convert target labels to binary (1 = good credit, 0 = bad credit)
y = y.map({'good': 1, 'bad': 0})

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())
```


***
