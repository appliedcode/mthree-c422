## **Problem Set: Data Ethics in Smart City IoT Sensor Data Management**

### **Scenario**

You are part of a **Smart City analytics team** responsible for collecting and analyzing IoT sensor data to improve urban safety, traffic flow, and environmental monitoring. The city council wants to ensure **ethical handling of data** — from **collection and consent** to **governance, privacy, and compliance**.

You will work with a **synthetic dataset** simulating **vehicle tracking sensors**, **air quality monitors**, and some **citizen-reported incidents**.

***

### **Dataset Setup (Colab-Ready)**

```python
import pandas as pd
import numpy as np

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

## **Exercises**

### **1. Data Collection, Privacy \& Consent**

- **Task 1.1:** Write a function to filter dataset to only keep records with valid consent for analytics.
- **Task 1.2:** Compare size before and after consent filtering and discuss trade‑offs.

***

### **2. Implicit vs Explicit Consent**

- **Task 2.1:** Count and plot proportion of explicit, implicit, and none.
- **Task 2.2:** Discuss how **implicit consent** might be risky in IoT sensor deployments.

***

### **3. Bias in Data**

- **Task 3.1:** Check if `Reported_Incident` rates differ significantly by `District`.
- **Task 3.2:** Identify possible causes (sensor density, demographic differences) and how to correct bias.

***

### **4. Data Minimization**

- **Task 4.1:** Select only the necessary columns for traffic congestion analysis, removing unrelated personal or sensitive attributes.
- **Task 4.2:** Explain benefits \& risks of minimization.

***

### **5. Anonymization \& Pseudonymization**

- **Task 5.1:** Remove `Device_ID` or replace with hashed pseudonym.
- **Task 5.2:** Test whether you can still link records across datasets.

***

### **6. De-Anonymization Risks**

- **Task 6.1:** Create an auxiliary dataset with `Owner_Age` + `District`.
- **Task 6.2:** Attempt a simple join to re-identify pseudonymized records and measure percentage matched.

***

### **7. Data Governance \& Stewardship**

- **Task 7.1:** Create a data access control table defining roles (Admin, Analyst, Public).
- **Task 7.2:** Outline responsibilities of a Data Steward for this smart city project.

***

### **8. Data Quality \& Lineage**

- **Task 8.1:** Implement a basic data quality report: missing values, out-of-range `Air_Quality_Index`.
- **Task 8.2:** Create a transformation log tracking each processing step.

***

### **9. Ethical Data Sharing**

- **Task 9.1:** Build a sharable, aggregated version of the dataset by district (no personal info).
- **Task 9.2:** Define rules for approving or denying sharing requests.

***

### **10. Monitoring \& Compliance**

- **Task 10.1:** Write a compliance check that:
    - Alerts if consent compliance ratio <90%
    - Alerts if `Reported_Incident` rate disparity by district >0.15
- **Task 10.2:** Document escalation steps if an alert is triggered.

***

### **Bonus**

- Apply **k-anonymity checks** to District + Age combination.
- Simulate **real-time consent withdrawal** and show how to handle it in downstream analytics.

***