# 🎓 **Solution + Code: AI Ethics \& Core Principles with University Admissions Dataset**


***

## 0️⃣ Setup \& Dataset Creation

```python
# Install required packages for modeling and visualization
!pip install --quiet scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Dataset creation ---
np.random.seed(123)

genders = ["Male", "Female", "Non-binary"]
majors = ["Computer Science", "Engineering", "Business", "Arts", "Science"]
regions = ["Urban", "Suburban", "Rural"]

n = 1500
data = pd.DataFrame({
    "Applicant_ID": range(1, n+1),
    "Age": np.random.randint(17, 40, n),
    "Gender": np.random.choice(genders, n, p=[0.47, 0.48, 0.05]),
    "High_School_GPA": np.round(np.random.uniform(2.0, 4.0, n), 2),
    "SAT_Score": np.random.randint(800, 1601, n),
    "Intended_Major": np.random.choice(majors, n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
    "Region": np.random.choice(regions, n, p=[0.5, 0.3, 0.2])
})

# Introduce pattern: high GPA & SAT => higher admission chance
data["Admitted"] = np.where(
    (data["High_School_GPA"] >= 3.5) & (data["SAT_Score"] >= 1300),
    np.random.choice([1, 0], n, p=[0.8, 0.2]),
    np.random.choice([1, 0], n, p=[0.25, 0.75])
)

data.head()
```


***

## 1️⃣ Fairness — Admission Rate By Groups

```python
# Calculate admission rate by Gender
gender_rates = data.groupby("Gender")["Admitted"].mean() * 100
print("Admission Rates by Gender (%)")
print(gender_rates)

# Calculate admission rate by Region
region_rates = data.groupby("Region")["Admitted"].mean() * 100
print("\nAdmission Rates by Region (%)")
print(region_rates)

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.barplot(x=gender_rates.index, y=gender_rates.values, ax=ax[0])
ax[0].set_title("Admission Rate by Gender")
sns.barplot(x=region_rates.index, y=region_rates.values, ax=ax[1])
ax[1].set_title("Admission Rate by Region")
plt.show()
```

**Interpretation Example:**
If one gender or region has a significantly lower admission rate without clear academic cause, it may signal possible **bias or imbalance in data**.

***

## 2️⃣ Transparency — Logistic Regression \& Feature Importance

```python
# Prepare data for modeling
df_model = pd.get_dummies(data.drop(columns=["Applicant_ID"]), drop_first=True)
X = df_model.drop("Admitted", axis=1)
y = df_model["Admitted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Show feature coefficients
feat_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
print(feat_importance)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Example Insight:**
Positive coefficient for `SAT_Score` = higher SAT increases admission likelihood.
Negative coefficient for certain `Region_…` → lower admission odds for that group.

***

## 3️⃣ Accountability — Logging Predictions

```python
import csv

def log_prediction(model, input_df, log_file="admissions_log.csv"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    record = {
        "timestamp": datetime.now().isoformat(),
        **{col: input_df[col].values[0] for col in input_df.columns},
        "prediction": int(pred),
        "probability": float(proba)
    }
    # Append log
    file_exists = False
    try:
        open(log_file).close()
        file_exists = True
    except:
        pass
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    return record

# Example logging with 1 sample from test set
sample_input = X_test.iloc[[0]]
log_prediction(model, sample_input)
```

**Why logs help:** Auditors can review inputs \& outputs to detect bias, explain decisions, and trace issues.

***

## 4️⃣ Privacy — Anonymize IDs

```python
import hashlib

data["Applicant_ID_Hash"] = data["Applicant_ID"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
print(data[["Applicant_ID", "Applicant_ID_Hash"]].head())
```

**Trade-off:** Anonymization keeps identities hidden but makes it harder to link with original records unless a secure mapping is kept.

***

## 5️⃣ Safety \& Reliability — Data Validation

```python
# Inject invalid entries
data.loc[0, "SAT_Score"] = 2000
data.loc[1, "High_School_GPA"] = -1

# Validation function
def validate_data(df):
    errors = []
    if (df["SAT_Score"] > 1600).any() or (df["SAT_Score"] < 400).any():
        errors.append("Invalid SAT score found.")
    if (df["High_School_GPA"] < 0).any() or (df["High_School_GPA"] > 4.0).any():
        errors.append("Invalid GPA found.")
    return errors

print("Validation errors:", validate_data(data))
```

**Action:** Such checks prevent the model from learning from nonsensical data.

***

## 6️⃣ Inclusivity — Representation Analysis

```python
# Gender representation %
gender_dist = data["Gender"].value_counts(normalize=True) * 100
print("Gender Representation (%)")
print(gender_dist)

# Region representation %
region_dist = data["Region"].value_counts(normalize=True) * 100
print("\nRegion Representation (%)")
print(region_dist)

# Intended Major representation %
major_dist = data["Intended_Major"].value_counts(normalize=True) * 100
print("\nIntended Major Representation (%)")
print(major_dist)
```

**Reflection:** If a major group (e.g., Non-binary applicants from Rural region) is <5%, outreach programs or targeted recruitment could improve balance.

***

## 📌 Summary Learnings:

- **Fairness:** Identified group disparities in admissions.
- **Transparency:** Understood how Logistic Regression weights features.
- **Accountability:** Created logs for predictions.
- **Privacy:** Anonymized applicant IDs.
- **Safety/Reliability:** Detected and flagged faulty records.
- **Inclusivity:** Quantified representation of demographic/academic groups.

***