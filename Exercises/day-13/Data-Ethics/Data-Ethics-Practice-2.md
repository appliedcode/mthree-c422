## **Problem Set: Data Ethics in Healthcare Wearable Device Data Management**

### **Scenario**

You are part of a team managing **health and fitness wearable devices** that track heart rate, steps, sleep patterns, and GPS location for users across different countries. The company aims to use this data for public health studies **while ensuring ethical data handling** throughout its lifecycle.

You will work with a **synthetic dataset** representing anonymized wearable device data along with user consent status.

***

### **Dataset Setup (Colab‑Ready)**

```python
import pandas as pd
import numpy as np

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

## **Exercises**

### **1. Data Collection, Privacy \& Consent**

- **Task 1.1**: Filter out any records where consent is `"none"`.
- **Task 1.2**: Discuss how different consent methods affect the dataset size and representativeness.

***

### **2. Implicit vs Explicit Consent**

- **Task 2.1**: Count the proportion of explicit, implicit, and no consent.
- **Task 2.2**: Explain risks of using implicitly collected wearable device data.

***

### **3. Bias in Data**

- **Task 3.1**: Check if `Health_Alert` rates differ significantly by `Country`.
- **Task 3.2**: Discuss whether the difference is due to actual health conditions or uneven device adoption rates.

***

### **4. Data Minimization**

- **Task 4.1**: Create a dataset for *step count analysis* only, excluding health-sensitive info.
- **Task 4.2**: Discuss why minimization is important for wearable data.

***

### **5. Anonymization \& Pseudonymization**

- **Task 5.1**: Replace `Device_ID` with a hashed pseudonym.
- **Task 5.2**: Evaluate whether linking is still possible if another dataset leaks partial attributes.

***

### **6. De‑Anonymization Risks**

- **Task 6.1**: Create an auxiliary dataset with `User_Age` and `Country`.
- **Task 6.2**: Attempt to re‑identify pseudonymized records by joining on these fields.

***

### **7. Data Governance \& Stewardship**

- **Task 7.1**: Draft a governance table showing dataset fields and access levels for roles (Admin, Data Scientist, Research Partner).
- **Task 7.2**: List responsibilities of the Healthcare Data Steward.

***

### **8. Data Quality \& Lineage**

- **Task 8.1**: Identify missing/invalid values (e.g., `Sleep_Hours > 24`).
- **Task 8.2**: Create a mock lineage log showing transformations from raw to processed data.

***

### **9. Ethical Data Sharing**

- **Task 9.1**: Produce a public health report aggregated at the country level (no individual data).
- **Task 9.2**: Define criteria for approving external research data requests.

***

### **10. Monitoring \& Compliance**

- **Task 10.1**: Write a script to flag:
    - Consent compliance < 95%
    - `Health_Alert` rate disparity by country > 0.12
- **Task 10.2**: Define actions for compliance breaches including notification, review, and remediation.

***

### **Bonus Ideas**

- Assess **k‑anonymity** on (`Country`, `User_Age`) combinations.
- Simulate **real‑time user consent withdrawal** and its effect on active analytics pipelines.

***
