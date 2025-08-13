## **Solution – AI Governance, Compliance, and Auditing in Credit Scoring Model**

### **Colab Code**

```python
# -*- coding: utf-8 -*-
"""Credit Scoring AI Governance, Compliance, and Auditing"""

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
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)

# Features (X) and Target (y)
X = credit_data.data
y = credit_data.target.map({'good': 1, 'bad': 0})

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Add synthetic sensitive attribute - Age Group (0: Younger, 1: Older)
np.random.seed(42)
X['age_group'] = np.random.choice([0, 1], size=len(y), p=[0.6, 0.4])

# -------------------
# Train-test split
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
# Bias Detection by Age Group
# -------------------
test_results = X_test.copy()
test_results['actual'] = y_test
test_results['predicted'] = y_pred

grouped = test_results.groupby('age_group').agg(
    total=('actual', 'count'),
    positive_rate_actual=('actual', 'mean'),
    positive_rate_predicted=('predicted', 'mean')
)
print("\nOutcome rates by Age Group:\n", grouped)

# Add bias report to audit log
bias_report = grouped.to_string()
audit_log['bias_report'] = bias_report

# -------------------
# Model Interpretability with SHAP
# -------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot summary for positive class (good credit)
shap.summary_plot(shap_values[1], X_test)

# -------------------
# Compliance & Audit Documentation
# -------------------
compliance_notes = f"""
Model Compliance and Audit Report:
-----------------------------------
Timestamp: {audit_log['timestamp']}

Bias Detection:
Outcome disparities between age groups:
{bias_report}

Model Performance:
- Detailed metrics in classification report
- Confusion matrix: {audit_log["confusion_matrix"]}

Transparency:
- SHAP used to interpret feature importance and influence
- Explainability provided for decision accountability

Governance & Auditing:
- Metrics, bias analysis, and interpretability documented
- Ready for review under AI governance and financial regulatory frameworks
"""

print(compliance_notes)
```


***

### **What This Solution Does**

1. **Loads the Credit Scoring dataset** from OpenML.
2. Creates a **synthetic sensitive attribute** `age_group` for bias analysis.
3. **Trains** a RandomForest model to predict creditworthiness.
4. **Evaluates** the model using classification metrics and confusion matrix.
5. Logs **AI audit data** including bias detection results.
6. Uses **SHAP explainability** to reveal important features.
7. Produces a **compliance report** ready for governance and regulatory review.

***
