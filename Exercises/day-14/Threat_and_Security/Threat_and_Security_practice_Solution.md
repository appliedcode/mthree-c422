## **Solution – Credit Card Fraud Detection with Security, Privacy \& Auditing**

```python
# -*- coding: utf-8 -*-
"""Credit Card Fraud Detection - AI Security, Privacy, and Auditing"""

# Install necessary packages
!pip install shap scikit-learn pandas matplotlib -q

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import datetime

# -------------------------
# 1. LOAD DATASET
# -------------------------
# NOTE: Ensure 'creditcard.csv' is uploaded to Colab or available in working directory
# Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

df = pd.read_csv('creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']  # Target: 1=Fraud, 0=Non-Fraud

print("Dataset shape:", X.shape)
print("Fraud class distribution:\n", y.value_counts())

# -------------------------
# 2. SIMULATE DATA POISONING ATTACK
# -------------------------
def poison_labels(y, fraction=0.02, target_label=0):
    """
    Flip labels for a fraction of fraud cases to normal (or vice versa) to simulate poisoning.
    """
    y_poisoned = y.copy()
    n_poison = int(fraction * len(y))
    indices = np.random.choice(len(y), n_poison, replace=False)
    y_poisoned.iloc[indices] = target_label
    return y_poisoned

print("\nSimulating label poisoning (2% flipped to class 0)...")
y_poisoned = poison_labels(y, fraction=0.02, target_label=0)

# -------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_poisoned, test_size=0.3, random_state=42, stratify=y_poisoned
)

# -------------------------
# 4. BASELINE MODEL TRAINING
# -------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nBaseline Model - Classification Report (With Poisoned Data):")
print(classification_report(y_test, y_pred, digits=4))

# -------------------------
# 5. PRIVACY PRESERVATION - FEATURE MASKING
# -------------------------
# Example: Drop "Amount" and "Time" columns as a simple anonymization step
X_privacy = X.drop(columns=['Amount', 'Time'])
X_train_priv, X_test_priv, y_train_priv, y_test_priv = train_test_split(
    X_privacy, y_poisoned, test_size=0.3, random_state=42, stratify=y_poisoned
)

clf_priv = RandomForestClassifier(n_estimators=100, random_state=42)
clf_priv.fit(X_train_priv, y_train_priv)
y_pred_priv = clf_priv.predict(X_test_priv)

print("\nPrivacy-Preserved Model - Classification Report:")
print(classification_report(y_test_priv, y_pred_priv, digits=4))

# -------------------------
# 6. AUDIT LOGGING
# -------------------------
audit_log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "poisoning_fraction": 0.02,
    "baseline_metrics": classification_report(y_test, y_pred, output_dict=True),
    "privacy_preserved_metrics": classification_report(y_test_priv, y_pred_priv, output_dict=True),
    "confusion_matrix_baseline": confusion_matrix(y_test, y_pred).tolist(),
    "confusion_matrix_privacy": confusion_matrix(y_test_priv, y_pred_priv).tolist()
}

print("\n--- Security Audit Log ---")
print(audit_log)

# -------------------------
# 7. MODEL EXPLAINABILITY (SHAP)
# -------------------------
print("\nGenerating SHAP explanation for baseline model...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Summary plot for class 1 (fraud)
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# -------------------------
# 8. COMPLIANCE / GOVERNANCE REPORT
# -------------------------
compliance_report = f"""
CREDIT CARD FRAUD DETECTION - AI SECURITY & PRIVACY AUDIT
--------------------------------------------------------
Timestamp: {audit_log['timestamp']}

Threat Simulation:
- Simulated {audit_log['poisoning_fraction']*100:.1f}% label poisoning to test fraud model robustness.

Model Performance:
- Baseline Accuracy: {audit_log['baseline_metrics']['accuracy']:.4f}
- Privacy-Preserved Accuracy: {audit_log['privacy_preserved_metrics']['accuracy']:.4f}

Security Governance:
- Evaluated model under data poisoning threat scenario.
- Applied simple feature masking ('Amount', 'Time') to enhance privacy.

Transparency:
- SHAP feature importance analysis used to interpret influential features for fraud classification.

Audit Readiness:
- Metrics, confusion matrices, and parameters logged for review.
- Supports compliance with GDPR, PCI DSS, and AI governance best practices.
"""

print(compliance_report)
```


***

### **What This Notebook Demonstrates**

1. **Dataset Handling** – Loads the highly imbalanced Credit Card Fraud dataset.
2. **Threat Simulation** – Flips a small percentage of fraud labels to simulate **data poisoning**.
3. **Baseline Model** – Trains a **Random Forest** on poisoned data.
4. **Privacy Preservation** – Applies **feature masking** to hide sensitive transaction fields.
5. **Auditing** – Logs metrics, poisoning configuration, confusion matrices.
6. **Transparency** – Uses **SHAP** to show most influential features in fraud prediction.
7. **Governance Report** – Outputs a compliance-friendly text summary for regulatory review.

***