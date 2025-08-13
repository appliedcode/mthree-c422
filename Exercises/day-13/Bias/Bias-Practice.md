## **Problem Set: Detecting and Mitigating Bias in Hiring Recommendation Algorithms**

### **Scenario**

You are part of a data science team building an **AI hiring recommendation model** that predicts whether a job applicant should be shortlisted.
The dataset contains demographic fields, education details, work history, and the current “shortlisted” status label. Your task is to **identify and mitigate bias** to ensure fair and ethical AI hiring.

***

### **Dataset Setup (Colab‑Ready synthetic)**

```python
import pandas as pd
import numpy as np

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

## **Exercises**

### **1. Understanding Types of Bias**

- **Task 1.1:** Define **data bias**, **algorithmic bias**, and **societal bias** in the context of hiring.
Give an example of each from this dataset or from a real scenario.
- **Task 1.2:** Identify which bias may be present before even training a model.

***

### **2. Detecting Bias in Data**

- **Task 2.1:** Calculate the proportion of applicants by **Gender** and **Ethnicity**; visualize as a bar chart.
- **Task 2.2:** Compute and compare `Shortlisted` rates across groups using statistical tests (e.g., Chi‑square).
- **Task 2.3:** Discuss whether observed disparities might come from unequal representation or historic hiring practices.

***

### **3. Detecting Algorithmic Bias**

- **Task 3.1:** Train a baseline model (e.g., logistic regression / decision tree).
- **Task 3.2:** Compute **Accuracy**, **True Positive Rate (TPR)**, and **False Positive Rate (FPR)** for each **Gender** and **Ethnicity** group.
- **Task 3.3:** Identify if prediction performance is uneven across groups.

***

### **4. Mitigating Bias**

- **Pre‑processing:** Use resampling or reweighting to balance the dataset before model training.
- **In‑processing:** Experiment with a fairness‑aware algorithm or constraints to reduce bias.
- **Post‑processing:** Adjust classification thresholds separately for different demographics to meet fairness criteria.

***

### **5. Fairness Metrics \& Evaluation**

- **Task 5.1:** Implement and compute:
    - **Demographic Parity Difference**
    - **Equal Opportunity Difference**
    - **Disparate Impact Ratio**
- **Task 5.2:** Calculate metrics for the baseline model and for the bias‑mitigated version.
- **Task 5.3:** Compare results, identify trade‑offs between fairness and accuracy, and explain which approach you would recommend to stakeholders.

***

### **6. Reflection \& Reporting**

- Write a summary documenting:
    - Detected biases and their sources
    - Methods used for mitigation
    - Fairness and performance trade‑offs
    - Ethical recommendations for deploying the hiring model in production

***

💡 **Bonus challenges**:

- Visualize ROC curves by demographic group.
- Test fairness metrics over multiple re‑train runs to check stability.
- Simulate societal bias influence by modifying the education distribution between groups.

***