## **Solution – AI Governance, Bias Detection \& Auditing in Heart Disease Prediction**

```python
# -*- coding: utf-8 -*-
"""Heart Disease AI Governance, Fairness, and Auditing"""

# Install required packages
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
heart_data = fetch_openml(name='heart-disease', version=1, as_frame=True)

# Features (X) and Target (y)
X = heart_data.data
y = heart_data.target.map({'present': 1, 'absent': 0})  # 1 = Disease present, 0 = No disease

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# Ensure 'sex' column exists as sensitive attribute (already part of dataset)
# Map to binary: 1=Male, 0=Female
X['sex'] = X['sex'].astype(int)  # Already encoded in dataset

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
test_results = X_test.copy()
test_results['actual'] = y_test.values
test_results['predicted'] = y_pred

grouped = test_results.groupby('sex').agg(
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

# Plot summary for positive class (heart disease present = 1)
shap.summary_plot(shap_values[1], X_test, show=True)

# -------------------
# Compliance & Audit Documentation
# -------------------
compliance_notes = f"""
Model Compliance and Audit Report:
-----------------------------------
Timestamp: {audit_log['timestamp']}

Bias Detection:
Outcome disparities between male and female patients:
{bias_report}

Model Performance:
- Detailed metrics in classification report
- Confusion matrix: {audit_log["confusion_matrix"]}

Transparency:
- SHAP used to interpret feature importance and influence
- Explainability ensures accountability in medical decisions

Governance & Auditing:
- Metrics, bias analysis, and interpretability documented
- Report aligns with AI in healthcare governance frameworks (HIPAA, WHO guidelines)
"""

print(compliance_notes)
```


***

### ✅ **What This Does**

1. **Loads the Heart Disease dataset** from OpenML (a medical dataset labeled "present"/"absent").
2. **Keeps `"sex"` as a sensitive feature** for bias detection.
3. Trains a **RandomForestClassifier** to predict heart disease risk.
4. **Evaluates performance** with accuracy, precision, recall, F1-score, and confusion matrix.
5. **Detects bias** in predictions across male and female groups.
6. **Explains the model** with SHAP to see which features drive predictions.
7. Generates a **Compliance \& Audit Governance Report** suitable for healthcare AI review.

***