## **Solution – AI Governance, Compliance, and Auditing in Adult Census Income Prediction**

### **Colab Code**

```python
# -*- coding: utf-8 -*-
"""Adult Census AI Governance, Compliance, and Auditing"""

# Install necessary packages
!pip install shap scikit-learn pandas -q

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import datetime

# -------------------
# Dataset Collection
# -------------------
adult_data = fetch_openml(name='adult', version=2, as_frame=True)

# Features (X) and Target (y)
X = adult_data.data
y = adult_data.target.map({'>50K': 1, '<=50K': 0})

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Encode categorical variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Sensitive attribute: We reconstruct 'sex' column for bias detection
# Original dataset contains 'sex', we need it separately for bias audit
adult_original = adult_data.data
sensitive_attr = adult_original['sex'].map({'Male': 1, 'Female': 0})  # 1=Male, 0=Female

X['sex_attr'] = sensitive_attr  # Keep a separate column

# -------------------
# Train-Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------
# Model Training
# -------------------
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# -------------------
# Model Evaluation
# -------------------
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------
# AI Auditing Step 1: Log metrics
# -------------------
audit_log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}
print("\nAudit Log Initial Entry:\n", audit_log)

# -------------------
# Bias Detection by Sex
# -------------------
# Retrieve sensitive attribute from test set
test_results = X_test.copy()
test_results['actual'] = y_test.values
test_results['predicted'] = y_pred

grouped = test_results.groupby('sex_attr').agg(
    total=('actual', 'count'),
    positive_rate_actual=('actual', 'mean'),
    positive_rate_predicted=('predicted', 'mean')
)

grouped.index = grouped.index.map({0: 'Female', 1: 'Male'})
print("\nOutcome rates by Sex:\n", grouped)

# Add bias report to audit log
bias_report = grouped.to_string()
audit_log['bias_report'] = bias_report

# -------------------
# Model Transparency with SHAP
# -------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot summary for class '1' (>50K income)
shap.summary_plot(shap_values[1], X_test, show=True)

# -------------------
# Compliance & Audit Documentation
# -------------------
compliance_notes = f"""
Model Compliance and Audit Report:
-----------------------------------
Timestamp: {audit_log['timestamp']}

Bias Detection:
Outcome disparities between sex groups:
{bias_report}

Model Performance:
- Detailed metrics in classification report
- Confusion matrix: {audit_log["confusion_matrix"]}

Transparency:
- SHAP used to interpret feature importance and influence
- Explainability provided for decision accountability

Governance & Auditing:
- Metrics, bias analysis, and interpretability documented
- Ready for review under AI governance frameworks and regulatory requirements
"""

print(compliance_notes)
```


***

### **How This Solution Works**

1. **Data Loading** – Fetches the **Adult Census Income dataset** from OpenML.
2. **Encoding** – Uses **One-Hot Encoding** for categorical variables while preserving `sex` for bias analysis.
3. **Model Training** – Trains a **RandomForestClassifier** to predict income level.
4. **Bias Detection** – Compares positive prediction rates between **Male** and **Female** groups.
5. **Transparency** – Uses **SHAP** to identify most important features influencing income predictions.
6. **Auditing \& Compliance** – Logs metrics, bias analysis, confusion matrix, and generates a textual governance report.

***
