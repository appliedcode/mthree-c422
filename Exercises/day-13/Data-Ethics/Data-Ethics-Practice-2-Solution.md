# **Solution: Data Ethics in Healthcare Wearable Device Data Management**


***

## **1. Dataset Creation**

```python
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt

np.random.seed(123)
n = 1000

consent_types = ["explicit", "implicit", "none"]
countries = ["USA", "UK", "Germany", "India", "Japan"]

data = pd.DataFrame({
    "Device_ID": range(1, n+1),
    "User_Age": np.random.randint(18, 80, n),
    "Country": np.random.choice(countries, n),
    "Consent_Type": np.random.choice(consent_types, n, p=[0.65, 0.25, 0.10]),
    "Average_HeartRate": np.random.randint(50, 160, n),
    "Daily_Steps": np.random.randint(1000, 20000, n),
    "Sleep_Hours": np.round(np.random.uniform(3, 10, n), 1),
    "GPS_Location_Share": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "Health_Alert": np.random.choice([0, 1], n, p=[0.85, 0.15])
})

data.head()
```


***

## **1. Data Collection, Privacy \& Consent**

```python
def filter_valid_consent(df):
    return df[df["Consent_Type"] != "none"]

valid_data = filter_valid_consent(data)

print(f"Before filtering: {len(data)} records")
print(f"After filtering: {len(valid_data)} records")
```

💡 **Note:** Removing "none" consent records ensures legal compliance, but reduces dataset size.

***

## **2. Implicit vs Explicit Consent**

```python
consent_counts = data["Consent_Type"].value_counts(normalize=True)
print(consent_counts)

consent_counts.plot(kind="bar", title="Consent Type Distribution", color=['green','orange','red'])
plt.ylabel("Proportion")
plt.show()

print("Implicit consent is risky for wearables because users may be unaware of background data collection.")
```


***

## **3. Bias in Data**

```python
# Health alert rates by country
alert_rates = data.groupby("Country")["Health_Alert"].mean()
print(alert_rates)

alert_rates.plot(kind="bar", title="Health Alert Rates by Country", color='skyblue')
plt.ylabel("Proportion with Health Alert")
plt.show()

print("Any large differences might be due to actual health factors or different adoption/user bases by country.")
```


***

## **4. Data Minimization**

```python
steps_dataset = data[["Country", "Daily_Steps"]]
steps_dataset.head()

print("This dataset excludes heart rate, GPS and other sensitive info, helping reduce risk exposure.")
```


***

## **5. Anonymization \& Pseudonymization**

```python
data["Device_PseudoID"] = data["Device_ID"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
data_pseudo = data.drop(columns=["Device_ID"])
data_pseudo.head()
```


***

## **6. De‑Anonymization Risks**

```python
# Auxiliary dataset from external source
aux_data = data[["User_Age", "Country"]].sample(150, random_state=1)

# Attempt naive re-identification
matches = pd.merge(data_pseudo, aux_data, on=["User_Age", "Country"])
match_rate = len(matches) / len(aux_data) * 100
print(f"Potential re-identification match rate: {match_rate:.2f}%")
```

💡 Even pseudonymized data can be re‑identified with demographic linkage — a privacy risk.

***

## **7. Data Governance \& Stewardship**

```python
access_matrix = pd.DataFrame({
    "Field": ["Device_ID", "Country", "Average_HeartRate", "Daily_Steps", "Sleep_Hours", "GPS_Location_Share", "Health_Alert"],
    "Admin": ["Read/Write"]*7,
    "Data_Scientist": ["No Access", "Read", "Read", "Read", "Read", "Read", "Read"],
    "Research_Partner": ["No Access", "Read", "Anonymized", "Read", "Read", "No Access", "Aggregated Only"]
})

print(access_matrix)
print("\nHealthcare Data Steward role: Ensure compliance with privacy regulations, approve access, keep lineage logs, respond to incidents.")
```


***

## **8. Data Quality \& Lineage**

```python
# Quality checks
print("Missing Values:\n", data.isnull().sum())
print("Invalid Sleep_Hours (>24):", (data["Sleep_Hours"] > 24).sum())

# Lineage log simulation
lineage_log = []
lineage_log.append({"step": "raw_ingest", "records": len(data)})
lineage_log.append({"step": "consent_filter", "records": len(valid_data)})
print("\nLineage Log:", lineage_log)
```


***

## **9. Ethical Data Sharing**

```python
# Aggregate by country
agg_health = data.groupby("Country").agg({
    "Daily_Steps": "mean",
    "Average_HeartRate": "mean",
    "Health_Alert": "mean"
}).reset_index()

print("Aggregated Public Health Report:\n", agg_health)

# Approval logic
def approve_request(req_type):
    if req_type in ["aggregated", "anonymized"]:
        return "Approved"
    return "Denied"

for req in ["aggregated", "anonymized", "full"]:
    print(f"{req} data request: {approve_request(req)}")
```


***

## **10. Monitoring \& Compliance**

```python
# Consent compliance ratio
consent_ratio = len(valid_data) / len(data)

# Disparity in Health Alert rates
alert_disparity = alert_rates.max() - alert_rates.min()

print(f"Consent Compliance: {consent_ratio:.2f}")
print(f"Health Alert Disparity: {alert_disparity:.2f}")

if consent_ratio < 0.95:
    print("⚠ ALERT: Consent compliance below threshold.")
if alert_disparity > 0.12:
    print("⚠ ALERT: Large disparity in health alerts detected.")

print("\nEscalation Steps:")
print("1. Notify Data Steward")
print("2. Investigate cause (sampling bias, quality issue)")
print("3. Apply corrective action (rebalancing, additional consent requests)")
print("4. Document and report resolution")
```


***
