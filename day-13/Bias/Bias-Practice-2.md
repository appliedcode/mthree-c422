## **Problem Set: Detecting and Mitigating Bias in University Admissions Algorithms**

### **Scenario**

You are a data analyst working with a university that uses an AI-based system to shortlist students for scholarships.
The system predicts whether an applicant should be awarded a scholarship based on their academic scores, extracurricular activities, and background details.
The university is concerned about **fairness** and **bias**, particularly across gender, socio-economic background, and high school type.

You’ll create a **synthetic dataset** and work through detection, mitigation, and fairness evaluation.

***

### **Dataset Setup (Colab‑Ready Synthetic)**

```python
import pandas as pd
import numpy as np

np.random.seed(101)
n = 1500

genders = ["Male", "Female", "Non-binary"]
school_types = ["Public", "Private", "International"]
income_levels = ["Low", "Medium", "High"]

data = pd.DataFrame({
    "Applicant_ID": range(1, n+1),
    "Gender": np.random.choice(genders, n, p=[0.45, 0.5, 0.05]),
    "School_Type": np.random.choice(school_types, n, p=[0.5, 0.35, 0.15]),
    "Income_Level": np.random.choice(income_levels, n, p=[0.4, 0.4, 0.2]),
    "GPA": np.round(np.random.uniform(2.0, 4.0, n), 2),
    "Extra_Curricular_Score": np.random.randint(0, 100, n)
})

# Introduce bias: Private & High income students slightly more likely to get scholarships
data["Scholarship_Awarded"] = np.where(
    ((data["School_Type"] == "Private") & (data["GPA"] > 3.5)) |
    ((data["Income_Level"] == "High") & (data["Extra_Curricular_Score"] > 70)),
    np.random.choice([1, 0], n, p=[0.75, 0.25]),
    np.random.choice([1, 0], n, p=[0.45, 0.55])
)

data.head()
```


***

## **Exercises**

### **1. Understanding Types of Bias**

- **Task 1.1:** Define and give examples of **data bias**, **algorithmic bias**, and **societal bias** in the **scholarship award** context.
- **Task 1.2:** Identify which types of bias could exist **before model training**.

***

### **2. Detecting Bias in Data**

- **Task 2.1:** Calculate group representation for **Gender**, **School_Type**, and **Income_Level**.
- **Task 2.2:** Compute and visualize `Scholarship_Awarded` rates by each group.
- **Task 2.3:** Perform Chi‑square tests between each sensitive attribute and `Scholarship_Awarded` to see if outcomes differ significantly.

***

### **3. Detecting Algorithmic Bias**

- **Task 3.1:** Train a baseline classifier (e.g., logistic regression or decision tree) to predict `Scholarship_Awarded`.
- **Task 3.2:** Compute **Accuracy**, **True Positive Rate (TPR)**, and **False Positive Rate (FPR)** for each **gender**, **income group**, and **school type**.
- **Task 3.3:** Analyze if performance disparities exist across these groups.

***

### **4. Mitigating Bias**

- **Pre‑processing:** Perform **resampling** or **re‑weighting** to ensure balanced representation of groups in the training data.
- **In‑processing:** Explore fairness‑aware algorithms or add constraints to reduce bias during training.
- **Post‑processing:** Adjust decision thresholds for different groups to achieve fairness targets.

***

### **5. Fairness Metrics \& Evaluation**

- **Task 5.1:** Implement and compute:
    - **Demographic Parity Difference**
    - **Equal Opportunity Difference**
    - **Disparate Impact Ratio**
- **Task 5.2:** Compare these metrics for the baseline vs. bias‑mitigated model.
- **Task 5.3:** Identify trade‑offs between fairness and accuracy.

***

### **6. Reflection \& Recommendations**

- Write a one‑page report containing:
    - Summary of detected biases and sources
    - Mitigation techniques applied
    - Fairness vs. accuracy trade‑offs
    - Ethical recommendations for deploying the admissions algorithm responsibly

***

💡 **Bonus Challenges:**

- Plot per‑group ROC curves to visualize performance differences.
- Simulate an **income‑blind admissions policy** and see effect on fairness metrics.
- Investigate intersectional bias (e.g., *Female + Low Income* group outcomes).

***