## Problem Statement: Predicting Loan Default Risk

### Background

Financial institutions need to assess the risk of customers defaulting on their loans to minimize losses and comply with regulatory requirements. Your task is to build a model that predicts whether a loan applicant will default based on their financial and personal information.

***

### Dataset Description

The dataset contains historical loan application data, including applicants’ demographic and financial details, loan status, and credit history.

**Key features include:**

- **Loan_ID:** Unique loan application identifier
- **Gender:** Male or Female
- **Married:** Whether the applicant is married (Yes/No)
- **Dependents:** Number of dependents (0,1,2,3+)
- **Education:** Applicant education level (Graduate/Not Graduate)
- **Self_Employed:** Whether the applicant is self-employed (Yes/No)
- **ApplicantIncome:** Monthly income of the applicant
- **CoapplicantIncome:** Monthly income of the co-applicant
- **LoanAmount:** Loan amount applied for (in thousands)
- **Loan_Amount_Term:** Term of loan in months
- **Credit_History:** Credit history meets guidelines (1 = yes, 0 = no)
- **Property_Area:** Urban, Semiurban, Rural
- **Loan_Status:** Target variable (Y = loan approved, N = loan not approved/defaulted)

***

### Dataset Source:

Download the dataset from [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

***

### Task:

- Load and preprocess the loan dataset (handle missing values, encode categorical variables).
- Train a classification model to predict loan default risk (Loan_Status).
- Package your model artifacts.
- Create a REST API using Flask to serve the model for inference with input JSON.
- Dockerize the application and run it locally.
- Test the API using sample loan application data.

***

### Learning Outcomes:

- Experience realistic preprocessing including income, credit history analysis.
- Handling imbalanced classification.
- Full ML lifecycle: training, deployment, and serving via Dockerized API.
- Prepare for production-grade ML system development.

***

This dataset and problem simulate typical financial risk prediction workflows and are widely used as practice benchmarks for ML lifecycle projects.

