# Problem Statement: Local ML Metadata Tracking Exercise with Customer Churn Dataset

## Overview

In this exercise, you will build a local ML metadata tracking system for a customer churn prediction problem. Using a synthetic customer dataset, you will simulate data collection, train a classification model to predict churn, and track all metadata (data artifacts, training executions, resulting models) in a local SQLite metadata store using ML Metadata (MLMD). You will then visualize and explore the recorded metadata.

***

## Details

- **Objective:** Implement and track an end-to-end churn prediction ML workflow with metadata on your local machine.
- **Dataset:** Synthetic customer data including customer features and churn labels.
- **Model:** Logistic Regression classifier.
- **Tracking:** Use ML Metadata to record datasets, training executions, and saved models along with their lineage.
- **Visualization:** Display execution metadata statistics and lineage information.
- **Tools:** Python, ML Metadata, pandas, scikit-learn, matplotlib, SQLite.

***

## Dataset Description

Simulate a customer churn dataset with the following columns:


| Feature | Description |
| :-- | :-- |
| customer_id | Unique customer identifier |
| monthly_charges | Monthly billing amount |
| tenure | Number of months as customer |
| contract_type | Contract type (Month-to-month, One year, Two year) |
| churn | Target variable: 1 = churned; 0 = stayed |

The dataset will be generated synthetically within the script and saved as a CSV file.

***

## Task Requirements

1. **Data Collection:**
    - Generate and save the synthetic churn dataset locally.
2. **Metadata Registration:**
    - Define artifact types for datasets and models.
    - Define execution types for the training job.
3. **Model Training:**
    - Load dataset, preprocess as needed, train a logistic regression model to predict churn.
    - Save the trained model locally.
4. **Metadata Logging:**
    - Register dataset artifact and model artifact with metadata store.
    - Log training execution lifecycle and link input/output artifacts.
5. **Visualization and Exploration:**
    - Visualize execution states distribution.
    - Query and print input/output artifacts for training executions.
6. **Metadata Store:**
    - Use a local SQLite database file as the ML metadata backend.

***

## Expected Directory Structure

```
customer_churn_metadata/
├── data/
│   └── churn_dataset.csv
├── models/
│   └── churn_model.joblib
├── metadata.db
└── run_churn_metadata.py   # Main Python script for exercise
```


***

## Deliverables

- Python script implementing the entire workflow (data generation, training, metadata registration and logging, visualization).
- Generated dataset CSV and saved model.
- SQLite metadata file (`metadata.db`) persisting metadata.
- Output visualizations and lineage printouts when running the script.

***
