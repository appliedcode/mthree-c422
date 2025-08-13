## **Problem Set: Operationalizing Ethics in AI Development \& Deployment — Recruitment Screening Context**

We’ll simulate an AI model that screens job applicants for shortlisting.
Your tasks will focus on embedding and operationalizing ethics both **during development** and **after deployment**.

***

### **1. Dataset Creation (Run in Colab)**

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


# Introduce a bias pattern:
# Applicants with Master's or PhD and Skill_Score > 70 more likely to be shortlisted
data["Shortlisted"] = np.where(
    (data["Education"].isin(["Masters", "PhD"])) & (data["Skill_Score"] > 70),
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.35, 0.65])
)


# Extra: Slight advantage for Ethnicity = Group_A
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

- Train a **baseline model** to predict `Shortlisted`.
- Measure positive prediction rates by `Gender` and `Ethnicity`.
- Identify potential bias sources.

***

### **Task 2 — Fairness Testing in Dev**

- Implement a **pre‑deployment fairness test** that fails if demographic disparity > `0.1`.

***

### **Task 3 — Feature Role \& Ethics Review**

- Generate **feature importance rankings**.
- Flag socio‑demographic features (Gender, Ethnicity) if they rank high.
- Suggest how to de‑bias them (e.g., removal, reweighting, representation balancing).

***

### **Task 4 — Ethics-Aware Agile User Stories**

- Document **two user stories** for your model development that explicitly incorporate ethics (e.g., *"As a recruiter, I want the model to avoid bias based on gender"*).

***

## **Part B — Operationalizing Ethics in Deployment**


***

### **Task 5 — Logging \& Accountability**

- Create a prediction function that **logs**:
    - Timestamp
    - Input features
    - Model output
    - Model version

***

### **Task 6 — Real-Time Monitoring**

- Simulate post‑deployment monitoring:
    - Track fairness disparity by Gender \& Ethnicity.
    - Trigger alerts if bounds are exceeded.

***

### **Task 7 — Privacy Controls**

- Apply **pseudo‑ID hashing** for `Applicant_ID` before storing predictions/logs.

***

### **Task 8 — Inclusivity Feedback Loop**

- Simulate adding feedback cases from underrepresented groups to improve future retraining.

***

### **Task 9 — Ethical Incident Simulation**

- Create a mock scenario where a bias complaint is received.
- Write a structured **incident response plan**:
    - Detection
    - Investigation
    - Communication
    - Remediation

***

### **Bonus**

- Integrate fairness metric checks (e.g., demographic parity) into a CI/CD simulation.
- Provide a **transparency report** summarizing model behavior and fairness results for stakeholders.

***

Here's a **fresh problem set** similar to the recruitment screening example, but focused on a **healthcare treatment recommendation context**:

***

## **Problem Set: Operationalizing Ethics in AI Development \& Deployment — Healthcare Treatment Recommendation Context**

We'll simulate an AI model that recommends treatment plans for patients based on their medical history and demographics. Your tasks will focus on embedding and operationalizing ethics both **during development** and **after deployment**.

***

### **1. Dataset Creation (Run in Colab)**

```python
import pandas as pd
import numpy as np

np.random.seed(456)

# Synthetic Healthcare Dataset
n = 1500
genders = ["Male", "Female", "Non-binary"]
ethnicities = ["White", "Black", "Hispanic", "Asian", "Other"]
insurance_types = ["Public", "Private", "Uninsured"]
conditions = ["Diabetes", "Hypertension", "Heart Disease", "Arthritis", "Mental Health"]

data = pd.DataFrame({
    "Patient_ID": range(1, n+1),
    "Age": np.random.randint(18, 85, n),
    "Gender": np.random.choice(genders, n, p=[0.48, 0.48, 0.04]),
    "Ethnicity": np.random.choice(ethnicities, n, p=[0.45, 0.2, 0.15, 0.15, 0.05]),
    "Insurance_Type": np.random.choice(insurance_types, n, p=[0.3, 0.6, 0.1]),
    "Primary_Condition": np.random.choice(conditions, n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
    "Severity_Score": np.random.randint(1, 11, n),  # 1-10 scale
    "Previous_Treatments": np.random.randint(0, 5, n),
    "Comorbidities": np.random.randint(0, 4, n)
})

# Introduce bias patterns:
# Higher severity + private insurance more likely to get advanced treatment
data["Advanced_Treatment_Recommended"] = np.where(
    (data["Severity_Score"] >= 7) & (data["Insurance_Type"] == "Private"),
    np.random.choice([1, 0], n, p=[0.8, 0.2]),
    np.random.choice([1, 0], n, p=[0.4, 0.6])
)

# Extra bias: Slight disadvantage for certain ethnic groups
mask_bias = data["Ethnicity"].isin(["Black", "Hispanic"])
data.loc[mask_bias, "Advanced_Treatment_Recommended"] = np.where(
    mask_bias,
    np.random.choice([1, 0], mask_bias.sum(), p=[0.3, 0.7]),
    data.loc[mask_bias, "Advanced_Treatment_Recommended"]
)

data.head()
```


***

## **Part A — Operationalizing Ethics in Development**


***

### **Task 1 — Bias Detection in Development**

- Train a **baseline model** to predict `Advanced_Treatment_Recommended`.
- Measure positive prediction rates by `Gender`, `Ethnicity`, and `Insurance_Type`.
- Identify potential healthcare disparities and bias sources.

***

### **Task 2 — Fairness Testing in Dev**

- Implement a **pre-deployment fairness test** that fails if treatment recommendation disparity between ethnic groups > `0.15`.

***

### **Task 3 — Feature Role \& Ethics Review**

- Generate **feature importance rankings**.
- Flag socio-demographic features (Gender, Ethnicity, Insurance_Type) if they rank high.
- Suggest how to de-bias them (e.g., clinical feature prioritization, insurance-blind modeling).

***

### **Task 4 — Ethics-Aware Clinical User Stories**

- Document **two user stories** for your model development that explicitly incorporate medical ethics (e.g., *"As a physician, I want the model to recommend treatments based purely on medical need, not insurance status"*).

***

## **Part B — Operationalizing Ethics in Deployment**


***

### **Task 5 — Clinical Decision Logging \& Accountability**

- Create a treatment recommendation function that **logs**:
    - Timestamp
    - Patient features (anonymized)
    - Model recommendation
    - Confidence score
    - Model version
    - Physician override (if any)

***

### **Task 6 — Real-Time Health Equity Monitoring**

- Simulate post-deployment monitoring:
    - Track treatment recommendation disparities by `Ethnicity` \& `Insurance_Type`.
    - Trigger alerts if health equity bounds are exceeded.

***

### **Task 7 — Patient Privacy Controls**

- Apply **medical-grade anonymization** for `Patient_ID` using secure hashing before storing logs.
- Implement data retention policies for clinical logs.

***

### **Task 8 — Health Equity Feedback Loop**

- Simulate collecting feedback from healthcare providers about treatment outcomes across different demographic groups.
- Design a process to incorporate this feedback into model retraining.

***

### **Task 9 — Medical Ethics Incident Simulation**

- Create a mock scenario where a healthcare disparity complaint is received.
- Write a structured **medical ethics incident response plan**:
    - Detection \& clinical review
    - Investigation with medical ethics committee
    - Patient/provider communication
    - Clinical remediation \& policy updates

***

### **Bonus**

- Integrate clinical fairness metric checks (e.g., equalized odds for treatment recommendations) into a medical AI deployment pipeline.
- Provide a **clinical transparency report** summarizing model behavior, health equity metrics, and clinical validation results for hospital stakeholders and regulatory bodies.

***
