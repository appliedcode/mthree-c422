import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)

# Set MLflow experiment name (creates one if missing)
mlflow.set_experiment("iris_rf_exp_tracking")

with mlflow.start_run():
    n_estimators = 300
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Log hyperparameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)
    
    # Log the model
    mlflow.sklearn.log_model(clf, "model")
    
    print(f"Logged run with accuracy: {acc:.4f}")
    

max_depth = 3
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="micro")

# Log parameters & metrics
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)