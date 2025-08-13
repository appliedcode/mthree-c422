## **Problem Set: AI Ethics and Core Principles with Larger Dataset**

This lab will help you practice **Fairness, Transparency, Accountability, Privacy, Safety/Reliability, and Inclusivity** using a **synthetic dataset** included below.

***

### **Dataset Creation**

Run the code below to generate your dataset in Colab. It simulates 1000 loan applications.

```python
import pandas as pd
import numpy as np


# Set random seed for reproducibility
np.random.seed(42)


# Possible values for categorical features
genders = ["Male", "Female", "Other"]
ethnicities = ["White", "Black", "Asian", "Hispanic", "Other"]


# Generate dataset
n = 1000
data = pd.DataFrame({
    "Applicant_ID": range(1, n+1),
    "Age": np.random.randint(18, 70, n),
    "Gender": np.random.choice(genders, n, p=[0.48, 0.48, 0.04]),
    "Ethnicity": np.random.choice(ethnicities, n, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
    "Income": np.random.randint(20, 150, n),  # Annual income in thousands
    "Credit_Score": np.random.randint(300, 851, n),
    "Loan_Amount": np.random.randint(5, 100, n),  # Loan requested in thousands
})


# Introduce a pattern: approval more likely with high credit score & income
data["Loan_Approved"] = np.where(
    (data["Credit_Score"] > 650) & (data["Income"] > 50), 
    np.random.choice([1, 0], n, p=[0.8, 0.2]),
    np.random.choice([1, 0], n, p=[0.3, 0.7])
)


# Preview dataset
data.head()
```


***

### **Your Tasks**

#### **1. Fairness**

- Calculate loan approval rates by **Gender** and **Ethnicity**.
- Identify disparities and discuss causes.
- Suggest 2–3 techniques to reduce bias.

***

#### **2. Transparency**

- Train a **Decision Tree Classifier** to predict `Loan_Approved`.
- Visualize the tree.
- Explain in plain language at least one decision path from the model.

***

#### **3. Accountability**

- Write a function that takes `model` and `input_data` and logs predictions together with inputs.
- Explain how such logs help in auditing AI decisions.

***

#### **4. Privacy**

- Anonymize or mask `Applicant_ID`.
- Discuss trade-offs between data privacy and utility.

***

#### **5. Safety \& Reliability**

- Add some invalid entries to the dataset (e.g., `Age = -5` or `Credit_Score = 2000`).
- Implement validation/error handling to manage such inputs gracefully.

***

#### **6. Inclusivity**

- Check representation percentages for each **Gender** and **Ethnicity**.
- Reflect on whether certain groups are underrepresented and how to make data more inclusive.

***

### **Bonus (Optional)**

- Compute fairness metrics (e.g., disparate impact).
- Compare decision tree with logistic regression in terms of fairness and transparency.
- Apply **differential privacy** techniques and observe their effect on model performance.

***

Here is a new exercise set similar to the one you provided, focused on AI Ethics and Core Principles with a different synthetic dataset related to employee promotion decisions.

***

## **Problem Set: AI Ethics and Core Principles with Employee Promotion Dataset**

This lab will help you practice **Fairness, Transparency, Accountability, Privacy, Safety/Reliability, and Inclusivity** using a **synthetic dataset** provided below.

***

### **Dataset Creation**

Run the code below to generate your dataset in Colab. It simulates 1200 employee promotion records.

```python
import pandas as pd
import numpy as np


# Set random seed for reproducibility
np.random.seed(100)


# Possible values for categorical features
genders = ["Male", "Female", "Non-binary"]
departments = ["Sales", "Engineering", "HR", "Marketing", "Finance"]


# Generate dataset
n = 1200
data = pd.DataFrame({
    "Employee_ID": range(1, n+1),
    "Age": np.random.randint(22, 65, n),
    "Gender": np.random.choice(genders, n, p=[0.50, 0.45, 0.05]),
    "Department": np.random.choice(departments, n, p=[0.25, 0.35, 0.15, 0.15, 0.10]),
    "Years_at_Company": np.random.randint(0, 30, n),
    "Performance_Score": np.random.randint(1, 6, n),  # Scale 1-5
    "Current_Salary": np.random.randint(30, 200, n),  # Annual salary in thousands
})


# Introduce pattern: promotion more likely with high performance and more years at company
data["Promoted"] = np.where(
    (data["Performance_Score"] >= 4) & (data["Years_at_Company"] >= 5), 
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.2, 0.8])
)


# Preview dataset
data.head()
```


***

### **Your Tasks**

#### **1. Fairness**

- Calculate promotion rates by **Gender** and **Department**.
- Identify any disparities and discuss potential reasons.
- Suggest 2–3 approaches to mitigate bias.

***

#### **2. Transparency**

- Train a **Random Forest Classifier** to predict `Promoted`.
- Extract and visualize one decision tree from the forest.
- Explain in simple terms one decision path leading to promotion.

***

#### **3. Accountability**

- Create a logging function that records each prediction made by your model along with the input features used.
- Explain how such logs support auditing and responsibility in AI decision-making.

***

#### **4. Privacy**

- Apply anonymization to `Employee_ID` (e.g., hashing or pseudonymization).
- Discuss the balance between preserving privacy and maintaining data usefulness for analysis.

***

#### **5. Safety \& Reliability**

- Introduce some flawed records like negative `Years_at_Company` or impossible `Performance_Score` values (e.g., 0 or 6).
- Implement validation checks to catch and handle these anomalies properly.

***

#### **6. Inclusivity**

- Analyze percentage representation for each **Gender** and **Department** group.
- Reflect on whether any groups are underrepresented and how dataset composition can be improved for inclusivity.

***

### **Bonus (Optional)**

- Calculate fairness metrics such as equal opportunity difference or demographic parity.
- Compare Random Forest results with Logistic Regression in terms of fairness, explainability, and performance.
- Experiment with data masking or noise addition to observe effects on model fairness and accuracy.

***

Generate another set of problem statement for practice, include new dataset

Got it ✅ — you want **another full practice lab problem set** in the same format, but with a **new synthetic dataset** for students to explore AI Ethics \& Core Principles.

Below is a **brand new exercise**, this time built around a **synthetic student admissions dataset**.
It keeps the **Fairness, Transparency, Accountability, Privacy, Safety/Reliability, and Inclusivity** structure, so students can follow the same thinking process while working with different data.

***

# **Problem Set: AI Ethics and Core Principles with University Admissions Dataset**

This lab will help you practice applying **Fairness, Transparency, Accountability, Privacy, Safety/Reliability, and Inclusivity** principles using a **synthetic dataset** related to student admission decisions.

***

## **Dataset Creation**

Run the following code in Google Colab to simulate **1500 student applications** for a university program:

```python
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(123)

# Possible categorical values
genders = ["Male", "Female", "Non-binary"]
majors = ["Computer Science", "Engineering", "Business", "Arts", "Science"]
regions = ["Urban", "Suburban", "Rural"]

# Create dataset
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

# Introduce pattern: higher GPA and SAT = higher admission chance
data["Admitted"] = np.where(
    (data["High_School_GPA"] >= 3.5) & (data["SAT_Score"] >= 1300),
    np.random.choice([1, 0], n, p=[0.8, 0.2]),
    np.random.choice([1, 0], n, p=[0.25, 0.75])
)

# Preview dataset
data.head()
```


***

## **Your Tasks**

### **1. Fairness**

- Calculate admission rates by **Gender** and **Region**.
- Identify any disparities in acceptance rates and possible causes.
- Suggest 2–3 bias-mitigation approaches (e.g., reweighting, blinding sensitive variables).

***

### **2. Transparency**

- Train a **Logistic Regression** model to predict `Admitted`.
- Show feature coefficients and explain what they mean for decision-making.
- In plain language, explain how GPA and SAT score affect admission chances.

***

### **3. Accountability**

- Write a function to log:
    - Applicant features
    - Model prediction
    - Probability score
    - Timestamp
- Describe how logs can help investigate complaints about unfair admission decisions.

***

### **4. Privacy**

- Anonymize `Applicant_ID` using hashing or pseudonyms.
- Discuss the trade-off between anonymization and the ability to track applicants across systems.

***

### **5. Safety \& Reliability**

- Intentionally add:
    - `SAT_Score` values > 1600
    - `High_School_GPA` < 0
- Implement input validation to reject or flag such invalid values before modeling.

***

### **6. Inclusivity**

- Calculate percentage distribution for each **Gender**, **Region**, and **Intended_Major**.
- Identify underrepresented applicant groups.
- Suggest strategies to ensure more balanced participation in the admissions dataset.

***

### **Bonus (Optional)**

- Calculate fairness metrics such as **disparate impact** or **equalized odds** by gender.
- Try a Decision Tree vs. Logistic Regression: compare fairness \& interpretability.
- Experiment with **differential privacy techniques** and analyze the impact on both fairness and accuracy.

***

💡 **Tip for Students:**
Think about **real-world implications** — this dataset mimics actual university admissions, where fairness, privacy, and inclusivity are critically important. Your analysis could directly relate to how real admission systems need ethical oversight.

***