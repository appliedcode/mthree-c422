## **Problem Set: Operationalizing Ethics in AI Development \& Deployment — Ride‑Hailing Driver Evaluation**

We’ll simulate an AI system that rates and flags ride‑hailing drivers for review based on their trip history and customer feedback.
The goal: embed and operationalize ethics from design through to post‑deployment monitoring.

***

### **1. Dataset Creation (Run in Colab)**

```python
import pandas as pd
import numpy as np

np.random.seed(77)

# Synthetic Dataset for Ride-Hailing Drivers
n = 1500
genders = ["Male", "Female", "Other"]
cities = ["Metro", "Urban", "Town"]

data = pd.DataFrame({
    "Driver_ID": range(1, n+1),
    "Age": np.random.randint(21, 65, n),
    "Gender": np.random.choice(genders, n, p=[0.6, 0.35, 0.05]),
    "City_Type": np.random.choice(cities, n, p=[0.4, 0.45, 0.15]),
    "Trips_Completed": np.random.randint(50, 2000, n),
    "Avg_Rating": np.round(np.random.uniform(3.0, 5.0, n), 2),  # rating out of 5
    "Complaints": np.random.randint(0, 20, n),
    "Accidents": np.random.randint(0, 5, n)
})

# Introduce bias: drivers from Metro more likely to be flagged for review
data["Flagged_For_Review"] = np.where(
    (data["Complaints"] > 5) | (data["Accidents"] > 0),
    np.random.choice([1, 0], n, p=[0.7, 0.3]),
    np.random.choice([1, 0], n, p=[0.15, 0.85])
)

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

- Train a **baseline model** to predict `Flagged_For_Review`.
- Measure positive prediction rates across **Gender** and **City_Type**.
- Identify if certain groups are unfairly targeted.

***

### **Task 2 — Ethics‑Aware Fairness Gate**

- Create a fairness testing function that:
    - Calculates **demographic disparity**.
    - Fails the build if disparity > `0.08`.

***

### **Task 3 — Feature Importance Ethics Audit**

- Output top feature importances.
- Flag socio‑demographic features (like `Gender`, `City_Type`) if they rank highly.
- Suggest mitigations such as **masking**, **regularization**, or **balanced training data**.

***

### **Task 4 — Ethical Development Documentation**

- Draft two **ethics user stories** for the development team that explicitly incorporate bias mitigation and transparency.

***

## **Part B — Operationalizing Ethics in Deployment**


***

### **Task 5 — Accountability via Prediction Logging**

- Build a prediction API function that logs:
    - Timestamp
    - Model version
    - Inputs
    - Prediction outcome

***

### **Task 6 — Post‑Deployment Monitoring**

- Simulate continuous fairness metric tracking by group.
- Trigger an alert if disparity > `0.08`.

***

### **Task 7 — Privacy Protection**

- Apply **hashing** to `Driver_ID` before storing logs to protect identity.

***

### **Task 8 — Inclusivity Feedback Loop**

- Simulate feedback submission from underrepresented driver groups.
- Show how these examples are added to the dataset for retraining.

***

### **Task 9 — Incident Response Simulation**

- Create a mock ethical incident scenario (e.g., high flag rates for one city).
- Write a **4‑step incident plan**:

1. Detection
2. Investigation
3. Communication
4. Remediation

***

### **Bonus Ideas**

- Use **SHAP** values to explain why specific drivers are being flagged.
- Implement **real‑time dashboard simulation** for bias monitoring in deployment.

***