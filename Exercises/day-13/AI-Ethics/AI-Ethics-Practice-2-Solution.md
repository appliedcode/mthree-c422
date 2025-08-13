## **Solution for AI Ethics and Core Principles — Employee Promotion Dataset**

### **1. Dataset Creation**

```python
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(100)

# Possible values
genders = ["Male", "Female", "Non-binary"]
departments = ["Sales", "Engineering", "HR", "Marketing", "Finance"]

# Create dataset
n = 1200
data = pd.DataFrame({
    "Employee_ID": range(1, n+1),
    "Age": np.random.randint(22, 65, n),
    "Gender": np.random.choice(genders, n, p=[0.50, 0.45, 0.05]),
    "Department": np.random.choice(departments, n, p=[0.25, 0.35, 0.15, 0.15, 0.10]),
    "Years_at_Company": np.random.randint(0, 30, n),
    "Performance_Score": np.random.randint(1, 6, n),  # 1 to 5
    "Current_Salary": np.random.randint(30, 200, n)  # in thousands
})

# Introduce a promotion pattern
data["Promoted"] = np.where(
    (data["Performance_Score"] >= 4) & (data["Years_at_Company"] >= 5), 
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.2, 0.8])
)

data.head()
```


***

## **Task 1 – Fairness**

**Calculate promotion rates by Gender and Department:**

```python
# Promotion rate by gender
gender_promotion = data.groupby("Gender")["Promoted"].mean() * 100
print("Promotion Rate by Gender (%):\n", gender_promotion)

# Promotion rate by department
dept_promotion = data.groupby("Department")["Promoted"].mean() * 100
print("\nPromotion Rate by Department (%):\n", dept_promotion)
```

**Discussion:**

- Differences in rates could indicate bias in past promotions or structural disparities.
- It’s important to check if performance/tenure distributions differ by demographic groups.

**Bias mitigation examples:**

1. **Re-sampling or re-weighting** the underrepresented or disadvantaged groups.
2. **Fairness-aware algorithms** that penalize biased decisions.
3. **Policy interventions** (e.g., structured performance reviews).

***

## **Task 2 – Transparency**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# Prepare data
features = ["Age", "Gender", "Department", "Years_at_Company", "Performance_Score", "Current_Salary"]
X = pd.get_dummies(data[features], drop_first=True)
y = data["Promoted"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# Extract one tree for transparency
one_tree = rf.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(one_tree, feature_names=X.columns, filled=True, max_depth=3)
plt.show()

# Show text rules from one tree
print(export_text(one_tree, feature_names=list(X.columns)))
```

**Example Decision Path:**
> If `Performance_Score >= 4.5` and `Years_at_Company >= 6`, likely promoted.
> This tells management that tenure + strong performance is key in the model’s view.

***

## **Task 3 – Accountability**

```python
import datetime

log_records = []

def log_prediction(model, input_df):
    pred = model.predict(input_df)[0]
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "input_data": input_df.to_dict(orient='records')[0],
        "prediction": int(pred)
    }
    log_records.append(log_entry)
    return pred

# Example logging of a single employee
sample_input = X_test.iloc[[0]]
log_prediction(rf, sample_input)
log_records[:2]  # Show first 2 logs
```

**Auditing Benefit:**

- Maintains a record of why and when decisions were made.
- Allows review in case of disputes or bias investigations.

***

## **Task 4 – Privacy**

```python
# Anonymize Employee_ID
import hashlib

def hash_id(x):
    return hashlib.sha256(str(x).encode()).hexdigest()

data["Employee_ID"] = data["Employee_ID"].apply(hash_id)

data.head()
```

**Trade-off Discussion:**

- Hashing protects identity but still allows record linking if needed.
- Full anonymization may reduce ability to track multiple applications by the same individual.

***

## **Task 5 – Safety \& Reliability**

```python
# Introduce bad records
data.loc[0, "Years_at_Company"] = -3
data.loc[1, "Performance_Score"] = 7  # invalid

# Validation
def validate_row(row):
    if row["Years_at_Company"] < 0 or not (1 <= row["Performance_Score"] <= 5):
        return False
    return True

invalid_rows = data[~data.apply(validate_row, axis=1)]
print("Invalid Rows Found:\n", invalid_rows)
```


***

## **Task 6 – Inclusivity**

```python
# Representation by Gender
gender_counts = data["Gender"].value_counts(normalize=True) * 100
print("Gender Representation (%):\n", gender_counts)

# Representation by Department
dept_counts = data["Department"].value_counts(normalize=True) * 100
print("\nDepartment Representation (%):\n", dept_counts)
```

**Reflection:**

- If any group has significantly lower representation (<5%), targeted recruitment or outreach can make data more balanced.
- A balanced dataset reduces bias risk.

***

## **Bonus – Fairness Metrics \& Comparison**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("RF Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("LR Accuracy:", accuracy_score(y_test, lr_preds))

# Example fairness metric — Demographic Parity Difference by Gender
def demographic_parity(y_true, y_pred, sensitive_values):
    groups = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive_values})
    rates = groups.groupby("group")["y_pred"].mean()
    return rates.max() - rates.min()

dp_diff = demographic_parity(y_test, rf.predict(X_test), 
                              data.iloc[y_test.index]["Gender"])
print(f"Demographic Parity Difference (RF): {dp_diff:.3f}")
```


***

✅ This full solution gives:

- **Code for data generation**
- **Analysis for fairness, transparency, accountability, privacy, safety, and inclusivity**
- **Optional fairness metrics**

***


