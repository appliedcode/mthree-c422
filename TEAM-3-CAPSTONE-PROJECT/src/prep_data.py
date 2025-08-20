# src/prep_data.py
import os
import pandas as pd

COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

def prepare_data(input_path="data/cleveland.data", output_path="data/heart_clean.csv"):
    print(f"Reading raw dataset: {os.path.abspath(input_path)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Accept commas OR whitespace, tolerant to non-UTF8, skip malformed
    df = pd.read_csv(
        input_path,
        names=COLUMNS,
        header=None,
        sep=r"[,\s]+",
        engine="python",
        encoding="latin1",
        na_values=["?", "NA", ""],
        on_bad_lines="skip"
    )

    print("After parse:", df.shape)
    # Coerce numerics
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows with valid target; convert multi-class to binary (0 vs >0)
    before = len(df)
    df = df[df["target"].notna()].copy()
    print(f"Dropped rows with missing target: {before - len(df)}")
    df["target"] = (df["target"] > 0).astype(int)

    # Keep NaNs in features (your train.py should impute)
    df.reset_index(drop=True, inplace=True)
    print("Final shape:", df.shape)
    print("Target distribution:\n", df["target"].value_counts())

    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned dataset to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    prepare_data()
