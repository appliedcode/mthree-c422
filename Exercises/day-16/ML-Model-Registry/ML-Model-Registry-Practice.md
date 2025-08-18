# Problem Statement: Managing Model Versions in MLflow Model Registry with Wine Quality Dataset

## **Objective**

Gain hands-on experience registering, versioning, and promoting machine learning models in the MLflow Model Registry using the Wine Quality dataset. Practice comparing different model types and organizing their lifecycle stages—from “None” to “Staging” and “Production.”

***

## **Business Context**

A beverage analytics team is building machine learning models to predict wine quality for quality control. Multiple models (Random Forest, Logistic Regression) need to be trained, logged, registered, and promoted. The team must ensure only the most suitable models are staged or served for production use.

***

## **Dataset**

- **Source:** UCI Wine Quality Dataset ([red](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv))
- **Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- **Target:** quality (convert to binary: good = quality ≥6, bad = quality <6)

***

## **Tasks**

### **1. Train and Log Multiple Models**

- Train at least two different models (e.g., Logistic Regression, Random Forest) to classify wine as good/bad quality.
- Use MLflow to log their parameters, metrics (accuracy), and model artifacts.


### **2. Register Models in the Registry**

- Register each trained model under a shared model name, e.g., `wine_quality_predictor`.


### **3. Manage Model Versions**

- Promote the best-performing model to the “Staging” stage.
- Archive or keep earlier versions in the “None” stage.
- Add descriptive annotations to each version (e.g., algorithm, performance, special notes).


### **4. Use Model by Stage**

- Write inference code to load and use the model assigned to the “Staging” or “Production” stage for making predictions on new wine samples.


### **5. Discussion and Reflection**

- How does the staging workflow help safe deployment?
- How can you ensure reproducibility and traceability for each model version?
- If your business requirements change (e.g., new quality thresholds), how would you handle model lifecycle changes in the registry?

***

## **Deliverables**

- MLflow experiment and model registry with at least two model versions
- At least one model in “Staging” or “Production” stage
- Sample prediction code that loads the model by its stage
- Brief notes/annotations for each model version

***