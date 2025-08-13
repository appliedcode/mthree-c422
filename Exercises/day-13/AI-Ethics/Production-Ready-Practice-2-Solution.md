# **Solution: Operationalizing Ethics in AI Development \& Deployment — Ride‑Hailing Driver Evaluation**


***

## **1. Dataset Creation**

```python
import pandas as pd
import numpy as np

np.random.seed(77)

# Synthetic dataset
n = 1500
genders = ["Male", "Female", "Other"]
cities = ["Metro", "Urban", "Town"]

data = pd.DataFrame({
    "Driver_ID": range(1, n+1),
    "Age": np.random.randint(21, 65, n),
    "Gender": np.random.choice(genders, n, p=[0.6, 0.35, 0.05]),
    "City_Type": np.random.choice(cities, n, p=[0.4, 0.45, 0.15]),
    "Trips_Completed": np.random.randint(50, 2000, n),
    "Avg_Rating": np.round(np.random.uniform(3.0, 5.0, n), 2),
    "Complaints": np.random.randint(0, 20, n),
    "Accidents": np.random.randint(0, 5, n)
})

# Introduce base label
data["Flagged_For_Review"] = np.where(
    (data["Complaints"] > 5) | (data["Accidents"] > 0),
    np.random.choice([1, 0], n, p=[0.7, 0.3]),
    np.random.choice([1, 0], n, p=[0.15, 0.85])
)

# Add bias: higher flag rates for Metro drivers
mask_metro = data["City_Type"] == "Metro"
data.loc[mask_metro, "Flagged_For_Review"] = np.where(
    mask_metro,
    np.random.choice([1, 0], mask_metro.sum(), p=[0.4, 0.6]),
    data["Flagged_For_Review"]
)

data.head()
```


***

## **Part A — Operationalizing Ethics in Development**


***

### **Task 1 — Bias Assessment Before Deployment**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare data (one-hot encoding for categorical vars)
X = pd.get_dummies(data.drop(columns=["Driver_ID", "Flagged_For_Review"]), drop_first=True)
y = data["Flagged_For_Review"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Baseline Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# Merge sensitive attrs for bias check
df_test = X_test.copy()
df_test["y_pred"] = y_pred
df_test["Gender"] = data.loc[y_test.index, "Gender"]
df_test["City_Type"] = data.loc[y_test.index, "City_Type"]

# Bias check
print("\nPositive prediction rate by Gender:")
print(df_test.groupby("Gender")["y_pred"].mean())

print("\nPositive prediction rate by City_Type:")
print(df_test.groupby("City_Type")["y_pred"].mean())
```


***

### **Task 2 — Ethics‑Aware Fairness Gate**

```python
def fairness_check(df, sensitive_attr, threshold=0.08):
    rates = df.groupby(sensitive_attr)["y_pred"].mean()
    disparity = rates.max() - rates.min()
    print(f"{sensitive_attr} disparity: {round(disparity, 3)}")
    return disparity <= threshold

print("\nFairness Gate Results:")
gender_fair = fairness_check(df_test, "Gender")
city_fair = fairness_check(df_test, "City_Type")

if not (gender_fair and city_fair):
    print("❌ Fairness gate FAILED — mitigation needed before deployment.")
else:
    print("✅ Fairness gate PASSED")
```


***

### **Task 3 — Feature Importance Ethics Audit**

```python
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features by Importance:")
print(importances.head(10))
```

💡 **Ethics note:** If `Gender_` or `City_Type_` rank high, flag for removal or mitigation.

***

### **Task 4 — Ethics Development User Stories**

Two sample **user stories**:

- *US1:* *As an ethics officer, I want the driver evaluation model to maintain bias disparity below 8% so that reviews are fair across all demographics.*
- *US2:* *As an operations manager, I want all model predictions logged with context so any disputed driver flagging can be audited.*

***

## **Part B — Operationalizing Ethics in Deployment**


***

### **Task 5 — Accountability via Prediction Logging**

```python
import datetime

prediction_log = []
MODEL_VERSION = "v1.0"

def predict_and_log(driver_features):
    pred = model.predict(driver_features)[0]
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "inputs": driver_features.to_dict(orient='records')[0],
        "prediction": int(pred)
    }
    prediction_log.append(log_entry)
    return pred

# Example log
sample_driver = X_test.iloc[[0]]
predict_and_log(sample_driver)
prediction_log[:1]
```


***

### **Task 6 — Post‑Deployment Monitoring**

```python
# Demographic parity after deployment
dp_gender = df_test.groupby("Gender")["y_pred"].mean().max() - df_test.groupby("Gender")["y_pred"].mean().min()
dp_city = df_test.groupby("City_Type")["y_pred"].mean().max() - df_test.groupby("City_Type")["y_pred"].mean().min()

print(f"Gender disparity: {dp_gender:.3f}")
print(f"City disparity: {dp_city:.3f}")

if dp_gender > 0.08 or dp_city > 0.08:
    print("⚠️ ALERT: Bias detected — review required")
else:
    print("✅ Fairness levels acceptable")
```


***

### **Task 7 — Privacy Protection**

```python
import hashlib

data["Driver_ID"] = data["Driver_ID"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
data.head()
```


***

### **Task 8 — Inclusivity Feedback Loop**

```python
# Sample feedback from underrepresented groups
feedback = pd.DataFrame({
    "Age": [30, 45],
    "Gender": ["Other", "Other"],
    "City_Type": ["Town", "Town"],
    "Trips_Completed": [500, 1200],
    "Avg_Rating": [4.6, 4.3],
    "Complaints": [1, 0],
    "Accidents": [0, 0],
    "Flagged_For_Review": [0, 0]
})

# Append for retraining
data_updated = pd.concat([data, feedback], ignore_index=True)
print("Updated dataset size:", data_updated.shape)
```


***

### **Task 9 — Ethical Incident Simulation**

**Scenario:** Post‑deployment, monitoring detects 20% disparity for `City_Type`.

**Incident Response Plan:**

1. **Detection** — Monitoring system alerts due to high disparity.
2. **Investigation** — Audit model logs, check feature importance, review training data.
3. **Communication** — Notify compliance and ops teams, share summary with stakeholders.
4. **Remediation** — Retrain with balanced city representation; remove/adjust `City_Type` weight; tighten fairness gate.

***

✅ **This notebook covers:**

- Bias checking in dev
- Automated fairness gates
- Feature importance ethics audits
- Deployment logging \& accountability
- Ongoing fairness monitoring
- Privacy protections
- Inclusivity feedback loops
- Ethical incident planning

***