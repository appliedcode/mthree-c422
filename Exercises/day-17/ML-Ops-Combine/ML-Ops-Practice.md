# Lab: ML Experiment Tracking, Model Registry \& API Serving with Breast Cancer Dataset


***

## **Objective**

Build an ML workflow to predict breast cancer diagnoses (benign/malignant) from patient measurements.
Systematically track your experiments and compare models, register and promote your best models, and provide a live API endpoint for frontend apps to consume predictions.

***

## **Dataset**

- **Source:** [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features:** 30 numeric attributes computed from digitized images of a breast mass (mean radius, mean texture, mean area, mean smoothness, etc.)
- **Target:** Diagnosis (`M` = malignant, `B` = benign)

***

## **Business Scenario**

A hospital needs a clear, versioned ML pipeline to classify tumors, track improvements across experiments, ensure only authorized models are in production, and make results available to clinicians via a web or mobile frontend.

***

## **Tasks**

### **1. Baseline Experiment Tracking**

- Train at least two classifiers (e.g., Logistic Regression, Random Forest) on the dataset.
- Track each run’s parameters and metrics (**accuracy, ROC-AUC, confusion matrix**) using MLflow.


### **2. Register and Manage Models**

- Register each model version in the **MLflow Model Registry**.
- Annotate versions and promote the best run to “Staging” or “Production”.


### **3. Build and Launch a Prediction API**

- Wrap the model in a **FastAPI** (or Flask) app.
- Expose endpoints:
    - `POST /predict`: Takes numerical features, returns prediction and probability.
    - `GET /health`: For liveness checking.


### **4. Frontend Integration Task (Bonus)**

- Document the API for frontend/UX teams (OpenAPI/Swagger auto-docs).
- Test the API with curl or a small web demo (optional).


### **5. Model Lifecycle Extension**

- Whenever a newer, better model is promoted, ensure the API loads the correct "Production" stage model automatically from the registry.

***

### **Deliverables**

- MLflow experiment logs with at least 2 model types/3 runs
- Registry with promoted "Production"/"Staging" model
- FastAPI Python script serving model predictions live
- API documentation or testing output
- Short README describing your design/choices

***

## **Reflection Questions**

- Which features most influence breast cancer diagnosis in your models?
- How does the registry streamline safe model updates and rollbacks?
- How does experiment tracking make model improvement and debugging easier?
- What are the benefits for frontend engineering of a documented, versioned API endpoint?

***

### **Optional Advanced Tasks**

- Log ROC curves or other artifacts as part of each experiment.
- Integrate pytest or unit testing for the API.
- Use Docker to containerize the full API service for portable deployment.

***
