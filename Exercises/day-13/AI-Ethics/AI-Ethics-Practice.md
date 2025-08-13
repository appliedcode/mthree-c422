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
