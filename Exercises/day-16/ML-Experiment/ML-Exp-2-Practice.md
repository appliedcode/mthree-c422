# Problem Statement: Track ML Experiments on Wine Quality Dataset

## **Objective**

Implement ML experiment tracking using MLflow with the Wine Quality dataset. Practice logging hyperparameters, metrics, and artifacts by running multiple experiments with different model configurations. Use the MLflow UI to compare, annotate, and register your best performing models.

***

## **Business Context**

Wine producers want to understand which physicochemical properties most strongly influence wine quality scores (rated 0–10). Your team is tasked with building and managing machine learning models to predict wine quality and systematically tracking all experiments for reproducibility and model improvement.

***

## **Dataset**

- **Source:** UCI Wine Quality Dataset ([red](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv), [white](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv))
- **Features:**
    - fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target:**
    - quality (integer 0–10; treat as multiclass classification)

***

## **Tasks**

### **1. Baseline Experiment**

- Train a RandomForestClassifier to predict wine quality (>= 6 good, < 6 bad—convert to binary for simplicity).
- Log hyperparameters (`n_estimators`, `max_depth`) and metrics (accuracy, F1-score).
- Track the run in MLflow.


### **2. Create Multiple Runs**

- Vary model hyperparameters (`n_estimators`, `max_depth`) in several script executions.
- Log results to the same MLflow experiment.
- View and compare accuracy, F1, and confusion matrix from each run in the MLflow UI.


### **3. Log Artifacts**

- Log the following as artifacts:
    - Confusion matrix plot
    - Feature importance plot


### **4. Annotate Runs**

- Add tags (e.g., `note: "baseline run"`, `model_type: "RandomForest"`)
- Add descriptive notes via MLflow’s UI or in code.


### **5. Register Best Model**

- Identify the run with the highest F1-score.
- Register the model in MLflow’s Model Registry.

***

## **Bonus Tasks**

- Try LogisticRegression and XGBClassifier; compare results.
- Track additional metrics (ROC-AUC, precision, recall).
- Experiment with multiclass classification (predict all quality classes).

***

## **Reflection Questions**

- Which features most influence wine quality?
- How did hyperparameter changes affect performance?
- Why is experiment tracking important for ML collaboration?
- How does artifact logging help with model selection and debugging?

***

## **Deliverables**

- MLflow experiment logs with at least 3 model runs
- Artifacts: plots (confusion matrix, feature importances)
- Annotated MLflow entries for each run
- Registered best model in Model Registry

***