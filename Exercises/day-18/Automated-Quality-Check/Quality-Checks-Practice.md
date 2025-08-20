# Problem Statement: Automated Testing and Code Quality for ML Project with Wine Quality Dataset Using GitHub Actions


***

## Scenario

You are developing a machine learning model to predict wine quality based on physicochemical features from the Wine Quality Dataset (available from UCI Machine Learning Repository). To maintain high code standards and robust functionality, you want to implement automated unit tests for your preprocessing and prediction functions. Additionally, you want to enforce code style and formatting rules. All these checks should run automatically on every push or pull request via GitHub Actions.

***

## Task

1. Implement Python functions for:
    - Preprocessing wine data (e.g., normalization, handling missing data if any).
    - A simple rule-based wine quality predictor based on key features (e.g., alcohol content, acidity).
2. Write comprehensive unit tests for these functions using `pytest` covering typical and edge cases.
3. Integrate code quality tools such as `flake8` for linting and `black` for code formatting checks.
4. Create a GitHub Actions workflow to automatically run tests and code quality checks on every push or pull request to the main branch.
5. Test locally with the Wine Quality dataset or a similar public wine dataset for accuracy and quality validation.

***

## Requirements

- Your preprocessing function should prepare raw data properly for modeling.
- The predictor function can be basic but must implement logical rules on features.
- Provide pytest unit tests with adequate coverage.
- Configure GitHub Actions to:
    - Run tests.
    - Perform `flake8` linting.
    - Check formatting with `black`.
- Demonstrate successful pipeline runs and error detection.

***

## Deliverables

- `wine_quality_ml.py` containing preprocessing and prediction functions.
- `test_wine_quality_ml.py` with pytest unit tests.
- `requirements.txt` including testing and linting dependencies.
- `.github/workflows/ci.yml` GitHub Actions workflow.
- Evidence (logs or screenshots) of GitHub Actions runs showing both success and failure scenarios.

***

