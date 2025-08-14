# Solution: Detecting and Mitigating Bias in Hiring Recommendation Algorithms


***

## 1. Dataset Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample

np.random.seed(42)
n = 1200

genders = ["Male", "Female", "Non-binary"]
ethnicities = ["Group_A", "Group_B", "Group_C"]
degrees = ["High School", "Bachelors", "Masters", "PhD"]

data = pd.DataFrame({
    "Applicant_ID": range(1, n+1),
    "Gender": np.random.choice(genders, n, p=[0.5, 0.45, 0.05]),
    "Ethnicity": np.random.choice(ethnicities, n, p=[0.4, 0.35, 0.25]),
    "Age": np.random.randint(20, 60, n),
    "Degree": np.random.choice(degrees, n, p=[0.2, 0.5, 0.25, 0.05]),
    "Years_Experience": np.random.randint(0, 20, n),
    "Skill_Test_Score": np.random.randint(40, 100, n)
})

# Introduce bias: Group_A and Males slightly higher shortlist rate
data["Shortlisted"] = np.where(
    ((data["Ethnicity"] == "Group_A") & (data["Skill_Test_Score"] > 70)) |
    ((data["Gender"] == "Male") & (data["Years_Experience"] > 5)),
    np.random.choice([1, 0], n, p=[0.7, 0.3]),
    np.random.choice([1, 0], n, p=[0.4, 0.6])
)

data.head()
```


***

## 2. Understanding Types of Bias (Summary for Reference)

- **Data Bias:** Unequal representation of groups or biased historical data; e.g., fewer non-binary applicants or fewer minority ethnicities, leading to less data for those groups.
- **Algorithmic Bias:** Model may favor features correlated with sensitive attributes or sensitive attributes explicitly included, resulting in unfair outcomes.
- **Societal Bias:** Existing social inequalities reflected in economic opportunities or hiring, encoded in the data.

***

## 3. Detecting Bias in Data

### 3.1 Distribution of Groups

```python
# Distribution by Gender
gender_counts = data["Gender"].value_counts(normalize=True)
print("Gender Distribution:")
print(gender_counts)

# Distribution by Ethnicity
ethnicity_counts = data["Ethnicity"].value_counts(normalize=True)
print("\nEthnicity Distribution:")
print(ethnicity_counts)

# Bar charts
gender_counts.plot(kind='bar', title='Gender Distribution')
plt.show()

ethnicity_counts.plot(kind='bar', title='Ethnicity Distribution')
plt.show()
```


### 3.2 Shortlist Rates Across Groups and Statistical Test

```python
# Shortlist rates by Gender
gender_shortlist_rate = data.groupby("Gender")["Shortlisted"].mean()
print("Shortlist Rates by Gender:")
print(gender_shortlist_rate)

# Shortlist rates by Ethnicity
ethnicity_shortlist_rate = data.groupby("Ethnicity")["Shortlisted"].mean()
print("\nShortlist Rates by Ethnicity:")
print(ethnicity_shortlist_rate)

# Chi-square test for Gender and Shortlisted
contingency_gender = pd.crosstab(data["Gender"], data["Shortlisted"])
chi2_g, p_g, dof_g, expected_g = chi2_contingency(contingency_gender)
print(f"\nChi-square test Gender-Shortlisted p-value: {p_g}")

# Chi-square test for Ethnicity and Shortlisted
contingency_ethnicity = pd.crosstab(data["Ethnicity"], data["Shortlisted"])
chi2_e, p_e, dof_e, expected_e = chi2_contingency(contingency_ethnicity)
print(f"Chi-square test Ethnicity-Shortlisted p-value: {p_e}")

# Plot shortlist rates
gender_shortlist_rate.plot(kind='bar', color='c', title='Shortlist Rate by Gender', ylim=(0,1))
plt.ylabel('Rate')
plt.show()

ethnicity_shortlist_rate.plot(kind='bar', color='m', title='Shortlist Rate by Ethnicity', ylim=(0,1))
plt.ylabel('Rate')
plt.show()
```


***

## 4. Detecting Algorithmic Bias

### 4.1 Prepare Data and Train Model

```python
# Encode categorical features
X = pd.get_dummies(data[["Gender", "Ethnicity", "Degree", "Years_Experience", "Skill_Test_Score", "Age"]], drop_first=True)
y = data["Shortlisted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression
model = LogisticRegression(solver='liblinear', max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```


### 4.2 Compute Group-wise Metrics

```python
def calculate_metrics(df, true_labels, preds, group_col):
    results = {}
    for group in df[group_col].unique():
        idx = df[group_col] == group
        tn, fp, fn, tp = confusion_matrix(true_labels[idx], preds[idx]).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        tpr = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        results[group] = {"Accuracy": accuracy, "TPR": tpr, "FPR": fpr}
    return pd.DataFrame(results)

# Prepare test data with group columns
df_test = X_test.copy()
df_test["Gender"] = data.loc[y_test.index, "Gender"]
df_test["Ethnicity"] = data.loc[y_test.index, "Ethnicity"]

metrics_gender = calculate_metrics(df_test, y_test, y_pred, "Gender")
metrics_ethnicity = calculate_metrics(df_test, y_test, y_pred, "Ethnicity")

print("Metrics by Gender:")
print(metrics_gender)

print("\nMetrics by Ethnicity:")
print(metrics_ethnicity)
```


### 4.3 Feature Importance

```python
coef = pd.Series(model.coef_[0], index=X.columns)
print("Feature Importances:")
print(coef.sort_values(ascending=False))
```


***

## 5. Mitigating Bias

### 5.1 Pre-processing: Resampling Minority Group (Example: Ethnicity Group_C)

```python
# Separate groups
df_minority = data[data['Ethnicity'] == 'Group_C']
df_majority = data[data['Ethnicity'] != 'Group_C']

# Upsample minority
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

# Combine to get balanced dataset
data_balanced = pd.concat([df_majority, df_minority_upsampled])

# Check new proportions
print("New Ethnicity Distribution After Balancing:")
print(data_balanced['Ethnicity'].value_counts(normalize=True))

# Check shortlist rates post balancing
print("\nShortlist rates by Ethnicity after balancing:")
print(data_balanced.groupby("Ethnicity")["Shortlisted"].mean())
```


### 5.2 Train Model on Balanced Data

```python
# Prepare new dataset
X_balanced = pd.get_dummies(data_balanced[["Gender", "Ethnicity", "Degree", "Years_Experience", "Skill_Test_Score", "Age"]], drop_first=True)
y_balanced = data_balanced["Shortlisted"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

model_bal = LogisticRegression(solver='liblinear', max_iter=200)
model_bal.fit(X_train_b, y_train_b)

y_pred_b = model_bal.predict(X_test_b)
print(f"Balanced Model Accuracy: {accuracy_score(y_test_b, y_pred_b):.3f}")
```


### 5.3 Evaluate Fairness Metrics on Balanced Model

```python
df_test_b = X_test_b.copy()
df_test_b["Gender"] = data_balanced.loc[y_test_b.index, "Gender"]
df_test_b["Ethnicity"] = data_balanced.loc[y_test_b.index, "Ethnicity"]

metrics_gender_b = calculate_metrics(df_test_b, y_test_b, y_pred_b, "Gender")
metrics_ethnicity_b = calculate_metrics(df_test_b, y_test_b, y_pred_b, "Ethnicity")

print("Balanced Model Metrics by Gender:")
print(metrics_gender_b)

print("\nBalanced Model Metrics by Ethnicity:")
print(metrics_ethnicity_b)
```


***

## 6. Fairness Metrics and Evaluation

### 6.1 Define Fairness Metrics

```python
def demographic_parity(y_true, y_pred, groups):
    rates = pd.DataFrame({'y_pred': y_pred, 'group': groups}).groupby('group')['y_pred'].mean()
    return rates.max() - rates.min()

def equal_opportunity(y_true, y_pred, groups):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
    group_tprs = {}
    for g in df['group'].unique():
        subset = df[df['group'] == g]
        tp = ((subset['y_pred'] == 1) & (subset['y_true'] == 1)).sum()
        fn = ((subset['y_pred'] == 0) & (subset['y_true'] == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0
        group_tprs[g] = tpr
    return max(group_tprs.values()) - min(group_tprs.values())

def disparate_impact(y_true, y_pred, groups, privileged_group):
    rates = pd.DataFrame({'y_pred': y_pred, 'group': groups}).groupby('group')['y_pred'].mean()
    if privileged_group not in rates:
        return None
    priv_rate = rates[privileged_group]
    min_rate = rates.min()
    return min_rate / priv_rate if priv_rate > 0 else None
```


### 6.2 Compute Metrics for Baseline and Balanced Model (Ethnicity)

```python
# Baseline Model
dp_base = demographic_parity(y_test, y_pred, df_test["Ethnicity"])
eo_base = equal_opportunity(y_test, y_pred, df_test["Ethnicity"])
di_base = disparate_impact(y_test, y_pred, df_test["Ethnicity"], privileged_group="Group_A")

print(f"Baseline Model:\nDemographic Parity Difference: {dp_base:.3f}\nEqual Opportunity Difference: {eo_base:.3f}\nDisparate Impact Ratio: {di_base:.3f}")

# Balanced Model
dp_bal = demographic_parity(y_test_b, y_pred_b, df_test_b["Ethnicity"])
eo_bal = equal_opportunity(y_test_b, y_pred_b, df_test_b["Ethnicity"])
di_bal = disparate_impact(y_test_b, y_pred_b, df_test_b["Ethnicity"], privileged_group="Group_A")

print(f"\nBalanced Model:\nDemographic Parity Difference: {dp_bal:.3f}\nEqual Opportunity Difference: {eo_bal:.3f}\nDisparate Impact Ratio: {di_bal:.3f}")
```


***

## 7. Reflection and Summary

- The baseline dataset shows bias through lower shortlist rates among Ethnicity Group_B and Group_C and uneven algorithm performance.
- Balancing the dataset helps mitigate data bias and achieves better fairness metrics, though sometimes accuracy might slightly fluctuate.
- Mitigation via re-sampling is a simple but effective approach; in-processing and post-processing methods could further improve results.
- Ethical deployment requires trading off fairness and accuracy with stakeholder priorities and transparency.
- Continuous monitoring post-deployment is essential to detect and address emerging bias.

***
