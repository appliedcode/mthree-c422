## Problem Statement: Predicting Customer Churn for a Telecom Company

### Background

Customer churn—when customers stop using a company's service—is a critical metric for telecom operators. Early identification of customers likely to churn allows targeted retention efforts, reducing revenue loss.

Your task is to build a machine learning model that predicts whether a telecom customer will churn based on their profile and usage data.

***

### Dataset

The dataset contains customer information and usage details recorded over the past year. Each row represents a customer, with features including:

- **CustomerID:** Unique identifier
- **Gender:** Male or Female
- **SeniorCitizen:** Whether the customer is 65 or older (0 = No, 1 = Yes)
- **Partner:** Whether the customer has a partner (Yes/No)
- **Dependents:** Whether the customer has dependents (Yes/No)
- **Tenure:** Number of months the customer has been with the company
- **PhoneService:** Whether the customer has phone service (Yes/No)
- **MultipleLines:** Whether customer has multiple lines (Yes/No/No phone service)
- **InternetService:** Customer’s internet service provider (DSL/Fiber optic/No)
- **OnlineSecurity:** Whether the customer has online security add-on (Yes/No/No internet service)
- **OnlineBackup:** Whether the customer has online backup (Yes/No/No internet service)
- **DeviceProtection:** Whether the customer has device protection (Yes/No/No internet service)
- **TechSupport:** Whether the customer has tech support (Yes/No/No internet service)
- **StreamingTV:** Whether the customer streams TV (Yes/No/No internet service)
- **StreamingMovies:** Whether the customer streams movies (Yes/No/No internet service)
- **Contract:** The contract term of the customer (Month-to-month/One year/Two year)
- **PaperlessBilling:** Whether the customer uses paperless billing (Yes/No)
- **PaymentMethod:** Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
- **MonthlyCharges:** The amount charged monthly
- **TotalCharges:** The total amount charged
- **Churn:** Target variable indicating if the customer has churned (Yes/No)

***

### Dataset Source:

You can download the dataset from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

***

### Task:

- Load and preprocess the dataset (handle missing data, encode categorical variables).
- Train a classification model to predict **Churn**.
- Save the trained model artifacts.
- Build a REST API with Flask that takes customer features as input and returns the churn prediction.
- Containerize the app with Docker and run locally.
- Test the API with sample customer data.

***

### Goals:

This exercise will help you practice the full ML lifecycle locally with Docker, covering:

- Data cleaning and preprocessing
- Model training and evaluation
- API creation and serving
- Docker containerization for portability

***

This problem statement is realistic and business-relevant, with a well-known dataset ensuring easy access and reproducibility.

