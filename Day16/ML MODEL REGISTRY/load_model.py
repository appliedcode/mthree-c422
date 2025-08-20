import mlflow.pyfunc

# Load latest Production model
model_uri = "models:/iris_random_forest@production"
model = mlflow.pyfunc.load_model(model_uri)


# Test prediction
print("Prediction:", model.predict([[5.8, 2.7, 5.1, 1.9]]))
