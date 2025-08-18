# Problem Statement: Track ML Experiments on Heart Disease UCI Dataset

## **Objective**

Apply ML experiment tracking with MLflow on the Heart Disease dataset. Run multiple experiments with different algorithms and hyperparameters, log metrics and artifacts, and use the tracking UI to compare results and document findings.

***

## **Business Context**

Cardiologists want to predict heart disease risk based on patient data (age, sex, rest ECG, cholesterol, chest pain type, etc.). Your task: Build and manage ML models systematically, tracking all experiments and artifacts to support reproducibility and continuous improvement.

***

## **Dataset**

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Features:** age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol), fbs (fasting blood sugar), restecg, thalach (max heart rate), exang (exercise-induced angina), oldpeak, slope, ca, thal
- **Target:** target (1 = heart disease present, 0 = not present)

***

## **Tasks**

### **1. Baseline Experiment**

- Train a LogisticRegression model to predict heart disease presence.
- Log hyperparameters (e.g., regularization parameter), metrics (accuracy, ROC-AUC), and confusion matrix using MLflow.


### **2. Multiple Runs with Varying Hyperparameters**

- Change the regularization strength in LogisticRegression, rerun and log each experiment.
- Try different algorithms (RandomForest, XGBoost).
- From the MLflow UI, compare each run’s metrics.


### **3. Artifact Logging**

- Log a confusion matrix plot as an artifact.
- Log a ROC curve image for each run.


### **4. Annotate and Organize**

- Add tags to runs (e.g., model_name, experimenter).
- Use notes to document each experiment’s focus or observations.


### **5. Register Best Model**

- Find the run with best ROC-AUC.
- Register the model in Model Registry for future reference/deployment.

***

## **Bonus Tasks**

- Experiment with feature selection and log feature importance bars.
- Track additional metrics (precision, recall, F1-score).
- Test model on a held-out validation set, logging its predictions and accuracy.

***

## **Reflection Questions**

- Which features are the strongest predictors?
- What hyperparameter settings produced the most accurate model?
- How does systematic experiment tracking help clinical ML projects?
- Why is artifact logging important?

***

## **Deliverables**

- MLflow logs for at least 4 runs with different models or hyperparameters.
- Artifacts: confusion matrices, ROC curves
- Annotated/organized MLflow runs
- Registered model in the registry

***
