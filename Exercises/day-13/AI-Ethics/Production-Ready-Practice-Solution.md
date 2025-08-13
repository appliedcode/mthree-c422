# **Solution: Operationalizing Ethics in AI Development \& Deployment — Recruitment Context**


***

## **1. Dataset Creation**

```python
import pandas as pd
import numpy as np

np.random.seed(123)

# Synthetic Recruitment Dataset
n = 1200
genders = ["Male", "Female", "Other"]
ethnicities = ["Group_A", "Group_B", "Group_C"]
education_levels = ["High School", "Bachelors", "Masters", "PhD"]

data = pd.DataFrame({
    "Applicant_ID": range(1, n+1),
    "Age": np.random.randint(20, 60, n),
    "Gender": np.random.choice(genders, n, p=[0.5, 0.45, 0.05]),
    "Ethnicity": np.random.choice(ethnicities, n, p=[0.4, 0.35, 0.25]),
    "Education": np.random.choice(education_levels, n, p=[0.3, 0.4, 0.2, 0.1]),
    "Years_Experience": np.random.randint(0, 20, n),
    "Skill_Score": np.random.randint(40, 100, n)  # Out of 100
})

# Introduce a bias pattern
data["Shortlisted"] = np.where(
    (data["Education"].isin(["Masters", "PhD"])) & (data["Skill_Score"] > 70),
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.35, 0.65])
)

# Slight advantage for Ethnicity = Group_A
mask_group_a = data["Ethnicity"] == "Group_A"
data.loc[mask_group_a, "Shortlisted"] = np.where(
    mask_group_a,
    np.random.choice([1, 0], mask_group_a.sum(), p=[0.55, 0.45]),
    data["Shortlisted"]
)

data.head()
```


***

## **Part A — Operationalizing Ethics in Development**


***

### **Task 1 — Bias Detection in Development**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare data
X = pd.get_dummies(data.drop(columns=["Applicant_ID", "Shortlisted"]), drop_first=True)
y = data["Shortlisted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train baseline model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Baseline Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# Merge back with sensitive attributes for bias check
df_test = X_test.copy()
df_test["y_true"] = y_test
df_test["y_pred"] = y_pred
df_test["Gender"] = data.loc[y_test.index, "Gender"]
df_test["Ethnicity"] = data.loc[y_test.index, "Ethnicity"]

# Bias by Gender
print("\nPositive prediction rate by Gender:")
print(df_test.groupby("Gender")["y_pred"].mean())

# Bias by Ethnicity
print("\nPositive prediction rate by Ethnicity:")
print(df_test.groupby("Ethnicity")["y_pred"].mean())
```

💡 **Interpretation:**
If there’s a large gap (>0.1 or 10%) between the highest and lowest group, it indicates potential bias.

***

### **Task 2 — Fairness Testing in Development**

```python
def fairness_check(df, sensitive_attr, threshold=0.1):
    rates = df.groupby(sensitive_attr)["y_pred"].mean()
    disparity = rates.max() - rates.min()
    print(f"{sensitive_attr} Disparity: {round(disparity, 3)}")
    return disparity <= threshold

print("\nFairness Checks:")
gender_fair = fairness_check(df_test, "Gender")
ethnic_fair = fairness_check(df_test, "Ethnicity")

if not (gender_fair and ethnic_fair):
    print("❌ Fairness check FAILED — Mitigation required.")
else:
    print("✅ Fairness check PASSED.")
```


***

### **Task 3 — Feature Role \& Ethics Review**

```python
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features (Importance):")
print(importances.head(10))
```

💡 If `Gender_` or `Ethnicity_` features appear very high, you should consider **removing or reducing their weight**.

***

### **Task 4 — Ethics-Aware Agile User Stories**

Example user stories:

- **US1:** *As a recruiter, I want the AI model to evaluate candidates without bias based on gender so that all applicants are treated fairly.*
- **US2:** *As a compliance officer, I want automated fairness checks in the model training pipeline so that biased models are not deployed.*

***

## **Part B — Operationalizing Ethics in Deployment**


***

### **Task 5 — Logging \& Accountability**

```python
import datetime

prediction_log = []
MODEL_VERSION = "1.0.0"

def predict_and_log(applicant_features):
    pred = model.predict(applicant_features)[0]
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "input_data": applicant_features.to_dict(orient='records')[0],
        "prediction": int(pred)
    }
    prediction_log.append(log_entry)
    return pred

# Example prediction logging
sample_applicant = X_test.iloc[[0]]
predict_and_log(sample_applicant)
prediction_log[:2]
```


***

### **Task 6 — Real-Time Monitoring**

```python
# Simulated post-deployment monitoring
dp_gender = df_test.groupby("Gender")["y_pred"].mean().max() - df_test.groupby("Gender")["y_pred"].mean().min()
dp_ethnicity = df_test.groupby("Ethnicity")["y_pred"].mean().max() - df_test.groupby("Ethnicity")["y_pred"].mean().min()

print(f"Gender Disparity: {dp_gender:.3f}")
print(f"Ethnicity Disparity: {dp_ethnicity:.3f}")

if dp_gender > 0.1 or dp_ethnicity > 0.1:
    print("⚠️ ALERT: Fairness threshold exceeded — Investigation required")
else:
    print("✅ Fairness levels within acceptable range")
```


***

### **Task 7 — Privacy Controls**

```python
import hashlib

data["Applicant_ID"] = data["Applicant_ID"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
data.head()
```


***

### **Task 8 — Inclusivity Feedback Loop**

```python
# Simulating underrepresented feedback cases
feedback = pd.DataFrame({
    "Age": [29, 32],
    "Gender": ["Other", "Other"],
    "Ethnicity": ["Group_C", "Group_C"],
    "Education": ["Bachelors", "Masters"],
    "Years_Experience": [5, 7],
    "Skill_Score": [82, 76],
    "Shortlisted": [1, 1]
})

# Append for future retraining
data_updated = pd.concat([data, feedback], ignore_index=True)
print("Updated dataset size:", data_updated.shape)
```


***

### **Task 9 — Ethical Incident Simulation**

**Bias Complaint Example Plan**

1. **Detection** — Monitoring dashboard flags Ethnicity bias > 0.15.
2. **Investigation** — Review model logs \& fairness metrics, audit training data composition.
3. **Communication** — Report findings to compliance team \& stakeholders.
4. **Remediation** — Retrain with balanced data + remove sensitive feature weights.
5. **Prevention** — Add stricter CI/CD bias thresholds.

***

✅ This code gives you a **complete working Colab notebook** to:

- Build the biased dataset
- Train \& check bias in dev
- Implement fairness gates
- Simulate deployment, logging, monitoring, privacy protection
- Add feedback loops \& incident response

***

