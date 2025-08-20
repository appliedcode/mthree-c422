# Problem Statement: Automated Testing for ML Diabetes Prediction Project with GitHub Actions
***

## Scenario

You are developing a machine learning model to predict the onset of diabetes using the Pima Indians Diabetes Dataset. To ensure the reliability and maintainability of your ML code, you need to implement automated unit tests for your data preprocessing and prediction functions. Furthermore, you want to automate these tests to run on every code push or pull request via GitHub Actions.

***

## Task

1. Implement Python functions for:
    - Data preprocessing (e.g., handling missing values, normalization).
    - A simple rule-based or dummy diabetes risk prediction function based on features (for demonstration).
2. Write unit tests for these functions using `pytest` that cover expected inputs, edge cases, and incorrect inputs.
3. Set up a GitHub Actions workflow that installs dependencies and runs the tests automatically on each push or pull request to the main branch.
4. Use the Pima Indians Diabetes Dataset (or a similar public health dataset) for local testing and validation.

***

## Requirements

- Your preprocessing function should correctly clean and prepare raw data inputs.
- Your prediction function should apply simple logic or dummy rules for diabetes risk classification.
- Pytest test cases must validate function outputs and error handling.
- The GitHub Actions workflow should:
    - Run tests on Python 3.x environment.
    - Install required libraries (e.g., pandas, pytest, scikit-learn).
    - Report test results clearly.
- Provide examples of successful test runs and at least one failing test scenario demonstrating pipeline failure.

***

## Deliverables

- `diabetes_ml.py` with preprocessing and prediction functions.
- `test_diabetes_ml.py` containing pytest test cases.
- `requirements.txt` listing dependencies.
- `.github/workflows/test-ci.yml` GitHub Actions workflow configuration.
- Evidence (screenshots or links) of GitHub Actions runs showing passing and failing tests.

***
