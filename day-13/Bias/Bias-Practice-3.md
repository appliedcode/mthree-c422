# \#\# **Problem Set: Detecting and Mitigating Bias in Hiring Recommendation Algorithms**

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
Generate another set of problem statement for different data set

Here is a new problem set focused on detecting and mitigating bias, but in a different domain and dataset, ready for hands-on exploration:

***

# **Problem Set: Detecting and Mitigating Bias in Credit Approval AI System**

## **Scenario**

You are part of a financial institution’s data science team working on an AI system that recommends credit approvals. The dataset contains applicant demographics, financial history, income details, and loan approval status. Your goal is to identify potential biases that may unfairly impact protected groups and apply mitigation techniques to ensure fair credit decisions.

***

## **Dataset Setup (Colab-Ready synthetic)**

```python
import pandas as pd
import numpy as np

np.random.seed(100)
n = 1500

genders = ["Male", "Female", "Other"]
races = ["White", "Black", "Hispanic", "Asian", "Other"]
employment_status = ["Employed", "Unemployed", "Self-employed", "Retired"]

data = pd.DataFrame({
    "Applicant_ID": range(1, n + 1),
    "Gender": np.random.choice(genders, n, p=[0.48, 0.49, 0.03]),
    "Race": np.random.choice(races, n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    "Age": np.random.randint(21, 70, n),
    "Employment_Status": np.random.choice(employment_status, n),
    "Annual_Income": np.round(np.random.normal(60000, 20000, n), 2),
    "Credit_Score": np.random.randint(300, 850, n)
})

# Introduce bias: Higher loan approval rates for 'White' and 'Employed'
data["Loan_Approved"] = np.where(
    ((data["Race"] == "White") & (data["Credit_Score"] > 650)) |
    ((data["Employment_Status"] == "Employed") & (data["Credit_Score"] > 600)),
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.4, 0.6])
)

data.head()
```


***

## **Exercises**

### 1. Understanding Bias Types

- Define **data bias**, **algorithmic bias**, and **societal bias** in credit approval context, with examples referencing this dataset.


### 2. Detecting Bias in Data

- Calculate applicant proportions by **Gender** and **Race**, visualize as bar charts.
- Compare approval rates by these groups; perform Chi-square tests to assess significance.
- Discuss whether disparities reflect representation imbalances or systemic discrimination.


### 3. Algorithmic Bias Detection

- Train a baseline classifier (e.g., logistic regression) predicting loan approval.
- Calculate **accuracy**, **TPR**, and **FPR** per Gender and Race group.
- Identify uneven predictive performance and discuss potential causes.


### 4. Bias Mitigation

- Apply **pre-processing**: reweight or resample data to balance groups before training.
- Experiment with **in-processing**: use fairness-aware algorithms or constraints.
- Use **post-processing**: adjust decision thresholds for groups to satisfy fairness metrics.


### 5. Fairness Metrics Computation

- Calculate:
    - **Demographic Parity Difference**
    - **Equal Opportunity Difference**
    - **Disparate Impact Ratio**
- Compare metrics before and after mitigation; analyze trade-offs with accuracy.


### 6. Reporting and Recommendations

- Summarize biases found, mitigation methods, and impact on fairness and predictive accuracy.
- Provide ethical guidelines and recommendations for deploying this model fairly.

***

### Bonus

- Visualize ROC curves by demographic groups.
- Perform multiple retrain experiments to check fairness metric stability.
- Simulate societal bias by skewing employment or income distributions for minority groups.

***