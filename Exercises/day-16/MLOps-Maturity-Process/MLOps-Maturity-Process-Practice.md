# Problem Statement: Assessing and Improving ML Process Maturity for Retail Sales Forecasting


***

## Scenario

Your team is developing a machine learning solution for **retail sales forecasting**. You want to ensure your project is robust, reproducible, and mature enough to scale for production. That includes: clear documentation, automated data validation, reproducible pipelines, model versioning/auditing, thorough testing, monitoring model performance, and enforcing responsible deployment.

***

## Dataset

Use the [Rossmann Store Sales Kaggle dataset](https://www.kaggle.com/c/rossmann-store-sales/data), which includes historical sales data for various stores, along with related features (store type, promotions, holidays, etc.).

***

## Tasks

1. **Project Structure \& Documentation**
    - Create a folder with subfolders for data, scripts, notebooks, and models.
    - Write a README.md with instructions for environment setup, running the code, and pipeline steps.
2. **Automated Data Download \& Validation**
    - Download at least a subset (e.g., “train.csv”) from the Kaggle dataset.
    - Use Python and a library (e.g., Great Expectations) to run automated checks for missing data, correct types, and valid ranges for “Sales” and “Store”.
3. **Model Training, Logging, and Versioning**
    - Train a regression model (e.g., RandomForestRegressor or XGBoost) to predict store sales.
    - Log experiment parameters, metrics (RMSE, MAE), and model artifacts using MLflow.
    - Ensure models and their metadata are reproducible and restorable.
4. **Unit and Integration Testing**
    - Write unit tests for data ingestion and feature engineering scripts (e.g., test for null values, correct scaling).
    - Run integration tests verifying the prediction pipeline works end-to-end.
5. **Local CI/CD Automation**
    - Use a Makefile or shell script to automate the steps: data download, validation, training, testing.
6. **Model Performance Monitoring**
    - Simulate live monitoring by writing a small script that logs prediction error as a Prometheus metric.
    - Create a dashboard (or expose localhost metrics) to visualize changes in model RMSE over time.
7. **User Authorization/Governance for Deployment**
    - Require a script that checks if the user running deployment is authorized (list of approved users).
    - Prevent unauthorized model deployment with a clear error message.
8. **Maturity Self-Audit**
    - Write a Python script to check for presence of:
        - Documentation files
        - Automated validation scripts
        - MLflow logs
        - Model files
        - Unit test scripts
        - CI/CD automation files
    - Print a report of strengths and gaps in the ML process.

***

## Goal

By solving this problem, you will:

- Build a reproducible and mature ML workflow for time series regression.
- Address documentation, automation, validation, testing, monitoring, and governance for scalable and production-ready machine learning.

***

This exercise is realistic for enterprise teams and ensures every step of your ML workflow is both automated and auditable—paving the way for industrial ML operations!

