## **Problem Statement** – AI Security, Threat Simulation, and Privacy-Preserving AI in Credit Card Fraud Detection

You are working in the **security analytics division** of a global financial services company.
Your team’s goal is to build, evaluate, and audit a **credit card fraud detection model** that not only detects fraudulent transactions but also complies with **AI security and privacy governance best practices**.

Your responsibilities include:

1. **Dataset Handling \& Preprocessing**
    - Load and analyze the **Credit Card Fraud dataset** containing anonymized transaction features.
    - Prepare the data for binary classification (`fraud` vs `non-fraud`).
2. **Threat Simulation**
    - Simulate a **data poisoning attack** where a small percentage of fraud labels are intentionally flipped, testing the system’s resilience to adversarial manipulation.
3. **Baseline Model Training**
    - Train a machine learning classifier to detect fraud and evaluate its performance on the poisoned dataset.
4. **Privacy Preservation**
    - Apply a **privacy-preserving transformation** (e.g., partial feature masking or removal of certain sensitive transaction indicators) and retrain the model.
    - Compare performance trade-offs between the original and privacy-preserved versions.
5. **Auditing \& Transparency**
    - Use **SHAP** to explain the model’s decisions and uncover key features influencing fraud classification.
    - Create an **audit log** recording performance metrics, poisoning parameters, and privacy measures taken.
6. **Governance Report**
    - Generate a report summarizing the threat simulation, model results, privacy steps, and interpretability findings, suitable for an internal security and compliance review.

**Business Context:**
Financial fraud models are prime targets for **adversarial attacks** and can inadvertently leak sensitive transaction patterns. Regulatory frameworks like **GDPR**, **PCI DSS**, and emerging **AI Act** mandate proper bias control, privacy safeguards, and detailed audit trails for AI systems used in financial services.

***

### **Dataset Collection Code**

```python
import pandas as pd

# Load the Credit Card Fraud Detection dataset from Kaggle public source
# If running in Colab, you must upload 'creditcard.csv' or fetch from Kaggle using API
# For example:
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv('creditcard.csv')

# Features (X) and Target (y)
X = df.drop(columns=['Class'])
y = df['Class']  # 1 = Fraud, 0 = Non-Fraud

print("Dataset shape:", X.shape)
print("Fraud distribution:\n", y.value_counts())

# Preview the first few rows
df.head()
```


***