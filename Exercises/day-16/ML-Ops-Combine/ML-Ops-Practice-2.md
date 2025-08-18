# Lab: Demand Prediction Pipeline — Bike Sharing Dataset


***

## **Objective**

Build and manage a complete ML workflow to predict hourly bike rental demand in a city.
Apply experiment tracking, structured model registry management, and expose live predictions via an API suitable for mobile and web applications.

***

## **Dataset**

- **Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Features:**
    - Hourly weather, temperature, windspeed
    - Time and date attributes (season, holiday, weekday, hour, etc.)
- **Targets:**
    - `cnt` — count of total bikes rented in an hour (regression)
    - Optionally, create a binary target for “High Demand” vs “Normal Demand”

***

## **Business Scenario**

A city transport management office wants to forecast hourly bike rental demand for smart allocation and pricing.
They require a trackable, versioned ML workflow, solid model comparison, and a secure API that can integrate with city dashboards for real-time operations.

***

## **Tasks**

### **1. ML Experiment Tracking**

- Train at least two models (e.g., Random Forest Regressor, Gradient Boosting Regressor) to predict rental count.
- Log model parameters and metrics (e.g., RMSE, MAE, R²) on a test set using MLflow.
- Store plots comparing predictions vs actual values as run artifacts.


### **2. Model Registry \& Lifecycle Management**

- Register each trained model version as `bike_demand_predictor` in the MLflow registry.
- Annotate and tag model versions with notes (e.g., “includes temp + windspeed + hour”).
- Promote the model with the best RMSE to “Production” stage.


### **3. API Development**

- Create a FastAPI (or Flask) app serving the current “Production” model for predictions.
- Endpoints to implement:
    - `POST /predict`: Accepts feature vector for a particular hour, returns predicted bike demand.
    - `GET /health`: Liveness check.


### **4. Testing and Documentation**

- Provide sample API documentation and example requests.
- Test using actual data from the dataset to verify predictions.

***

## **Deliverables**

- MLflow experiment logs for multiple model and hyperparameter runs.
- Registry with promoted model.
- Python API server for predictions.
- Example API documentation or screenshots.
- Short report on feature importance and model accuracy.

***

### **Advanced/Bonus Tasks**

- Add error analysis plots as artifacts in MLflow (e.g., residuals).
- Automate registry promotion with scripts or CI.
- Prepare model for batch inference (accepting multiple rows at once).
- Dockerize the API service for deployment.

***

## **Reflection Questions**

- Which time/weather features most strongly influence demand?
- How does experiment tracking and registry ensure reliable deployment?
- What could be added to the API to make it more robust or scalable?

***
