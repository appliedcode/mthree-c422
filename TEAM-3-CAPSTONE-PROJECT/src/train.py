# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

NUM = ["age","trestbps","chol","thalach","oldpeak"]
CAT = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

def main(in_path="data/heart_clean.csv", model_dir="models", model_name="heart_pipeline.pkl"):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing cleaned data: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError("heart_clean.csv is empty. Re-run prep_data.py and check logs.")

    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in heart_clean.csv")

    # make sure target is int
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    X = df[NUM + CAT]
    y = df["target"]

    print(f"Dataset rows: {len(df)}")
    print("Target distribution:\n", y.value_counts())

    # if dataset too small, skip splitting
    if len(df) < 5:
        print("⚠️ Too few samples, training without train/test split just for pipeline testing.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    pre = ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), NUM),
        ("cat", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT),
    ])

    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(y.unique()) > 1 else None,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    print("Metrics:", metrics)

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, model_name)
    joblib.dump({"pipeline": pipe, "features": NUM + CAT}, out_path)
    print(f"✅ Saved model to {out_path}")

if __name__ == "__main__":
    main()
