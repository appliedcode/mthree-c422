import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and scale dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

mlflow.set_experiment("breast_cancer_registry_demo")
model_name = "breast_cancer_diagnosis"
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, solver='lbfgs')),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42))
]
for model_label, model in models:
    with mlflow.start_run(run_name=model_label) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        # Log params and metrics
        mlflow.log_param("model_type", model_label)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        # Confusion Matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap="Blues", alpha=0.7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], va='center', ha='center')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")
        plt.close()
        # Prepare for signature and input_example logging
        import numpy as np
        input_example = X_test[:2]  # X_test is already a numpy array
        signature = infer_signature(X_train, model.predict(X_train))
        # Register model with all options
        mlflow.sklearn.log_model(
            model,
            name=model_name,
            registered_model_name=model_name,
            input_example=input_example,
            signature=signature
        )
        print(f"{model_label} logged & registered, Run ID: {run.info.run_id}")