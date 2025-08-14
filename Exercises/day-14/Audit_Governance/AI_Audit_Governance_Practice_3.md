## **Problem Statement** – AI Governance, Bias Detection \& Auditing in Income Prediction Model

You are part of an **AI governance team** at a tech company developing a predictive analytics system to identify individuals likely to earn more than **\$50K/year**.

The model must comply with **AI governance principles**, focusing on **fairness, transparency, and accountability**.

Your tasks in this exercise are to:

- **Detect and measure bias** in predictions across demographic groups (here we will use `"sex"` as a sensitive attribute: Male vs Female).
- **Provide model transparency** using **SHAP explainability** to understand the most influential features driving predictions.
- **Implement AI auditing** by logging performance metrics, bias statistics, and interpretability results.
- **Generate a compliance report** that can be reviewed by internal governance boards or external regulators.

**Business Context:**
Such a model must not unfairly disadvantage certain protected groups. This auditing ensures the system adheres to ethical AI frameworks (like IEEE, OECD, NIST AI RMF) and privacy regulations.

***

### **Dataset Collection Code**

```python
from sklearn.datasets import fetch_openml

# Load Adult Census dataset from OpenML
adult_data = fetch_openml(name='adult', version=2, as_frame=True)

# Extract features and target
X = adult_data.data
y = adult_data.target

# Convert target to binary: >50K = 1, <=50K = 0
y = y.map({'>50K': 1, '<=50K': 0})

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())
```


***