import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set experiment
mlflow.set_experiment("iris_registry_lab")

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# First Model (v1)
with mlflow.start_run(run_name="rf_model_v1") as run1:
    clf1 = RandomForestClassifier(n_estimators=30, random_state=42)
    clf1.fit(X_train, y_train)
    acc1 = clf1.score(X_test, y_test)
    
    # Log parameters & metrics
    mlflow.log_param("n_estimators", 30)
    mlflow.log_metric("accuracy", acc1)
    
    # Log model artifact
    mlflow.sklearn.log_model(clf1, artifact_path="model")

# Second Model (v2)
with mlflow.start_run(run_name="rf_model_v2") as run2:
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2.fit(X_train, y_train)
    acc2 = clf2.score(X_test, y_test)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc2)
    mlflow.sklearn.log_model(clf2, artifact_path="model")

print("✅ Two runs logged. Open MLflow UI to register them.")
