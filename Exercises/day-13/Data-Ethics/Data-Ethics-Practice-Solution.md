# **Solution: Data Ethics in Smart City IoT Sensor Data Management**


***

## **1. Dataset Creation**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib

np.random.seed(42)
n = 800

consent_types = ["explicit", "implicit", "none"]
districts = ["Downtown", "Suburban", "Industrial", "Green Zone"]

data = pd.DataFrame({
    "Device_ID": range(1, n+1),
    "Owner_Age": np.random.randint(18, 80, n),
    "District": np.random.choice(districts, n),
    "Consent_Type": np.random.choice(consent_types, n, p=[0.6, 0.3, 0.1]),
    "Vehicle_Count": np.random.randint(0, 200, n),
    "Avg_Speed": np.round(np.random.uniform(10, 80, n), 1),
    "Air_Quality_Index": np.random.randint(10, 300, n),
    "Reported_Incident": np.random.choice([0, 1], n, p=[0.85, 0.15])
})

data.head()
```


***

## **1. Data Collection, Privacy \& Consent**

**Task 1.1 \& 1.2**

```python
def filter_valid_consent(df):
    return df[df["Consent_Type"].isin(["explicit", "implicit"])]

valid_data = filter_valid_consent(data)

print(f"Before filtering: {len(data)} records")
print(f"After filtering: {len(valid_data)} records")
```

**Explanation:** Filtering ensures legal and ethical use of data, but may result in loss of sample size.

***

## **2. Implicit vs Explicit Consent**

**Task 2.1 \& 2.2**

```python
consent_counts = data["Consent_Type"].value_counts()
print(consent_counts)

consent_counts.plot(kind="bar", title="Consent Type Distribution", color=['green','orange','red'])
plt.ylabel("Count")
plt.show()

print("Discussion: Implicit consent in IoT is risky since users might not be aware of data collection, leading to potential legal issues.")
```


***

## **3. Bias in Data**

```python
incident_rates = data.groupby("District")["Reported_Incident"].mean()
print(incident_rates)

incident_rates.plot(kind="bar", title="Reported Incident Rates by District")
plt.show()

print("Bias observation: If certain districts have much higher/lower rates, this may be due to uneven sensor placement or socio-economic factors.")
```


***

## **4. Data Minimization**

```python
traffic_data = data[["District", "Vehicle_Count", "Avg_Speed"]]
traffic_data.head()

print("We removed personal and unrelated attributes. This reduces privacy risk and unnecessary exposure of sensitive info.")
```


***

## **5. Anonymization \& Pseudonymization**

```python
data["Device_PseudoID"] = data["Device_ID"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
data_anonymized = data.drop(columns=["Device_ID"])
data_anonymized.head()
```

**Explanation:** Device_ID is replaced with irreversible hashed pseudonym to protect identity.

***

## **6. De-Anonymization Risks**

```python
# Auxiliary dataset simulating publicly available demographic info
aux_data = data[["Owner_Age", "District"]].sample(100, random_state=1)

# Attempt to re-identify
matches = pd.merge(data_anonymized, aux_data, on=["Owner_Age", "District"])
match_rate = len(matches) / len(aux_data) * 100

print(f"Potential re-identifications via linkage: {match_rate:.2f}%")
```

**Observation:** Even without direct identifiers, linking with demographic info can compromise privacy.

***

## **7. Data Governance \& Stewardship**

```python
access_control = pd.DataFrame({
    "Field": ["Device_ID", "District", "Air_Quality_Index", "Owner_Age", "Reported_Incident"],
    "Admin": ["Read/Write"]*5,
    "Analyst": ["No Access", "Read", "Read", "Read", "Read"],
    "Public": ["No Access", "Read", "Read", "No Access", "Aggregated Only"]
})

print(access_control)
print("\nResponsibilities of Data Steward: Ensuring policy compliance, approving data access, monitoring privacy violations, and managing data lifecycle.")
```


***

## **8. Data Quality \& Lineage**

```python
# Quality check
print("Missing values:\n", data.isnull().sum())
print("Out of range AQI count:", ((data["Air_Quality_Index"] < 0) | (data["Air_Quality_Index"] > 500)).sum())

# Transformation log simulation
lineage_log = []
lineage_log.append({"step": "initial_load", "records": len(data)})
lineage_log.append({"step": "filter_valid_consent", "records": len(valid_data)})
print("\nLineage Log:", lineage_log)
```


***

## **9. Ethical Data Sharing**

```python
# Aggregated dataset
agg_data = data.groupby("District").agg({
    "Vehicle_Count": "mean",
    "Avg_Speed": "mean",
    "Air_Quality_Index": "mean",
    "Reported_Incident": "mean"
}).reset_index()

print("Aggregated dataset ready for sharing:\n", agg_data)

def approve_request(request_type):
    if request_type in ["aggregated", "anonymized"]:
        return "Approved"
    return "Denied"

for r in ["aggregated", "anonymized", "full"]:
    print(f"Request {r}: {approve_request(r)}")
```


***

## **10. Monitoring \& Compliance**

```python
# Consent compliance
consent_ratio = len(valid_data) / len(data)
# Incident disparity
inc_disparity = incident_rates.max() - incident_rates.min()

print(f"Consent compliance: {consent_ratio:.2f}")
print(f"Incident disparity: {inc_disparity:.2f}")

if consent_ratio < 0.9:
    print("⚠ ALERT: Consent compliance below threshold.")
if inc_disparity > 0.15:
    print("⚠ ALERT: High disparity in incident reporting detected by district.")

print("\nEscalation Steps: 1) Flag issue to Data Steward, 2) Investigate causes, 3) Apply corrective measures, 4) Report to oversight body.")
```


***
