## Solution: Loan Default Risk Prediction with Docker


***

### 1. Setup Project Structure

Create the folder layout:

```
loan-default-risk/
  ├── app.py
  ├── train.py
  ├── requirements.txt
  └── Dockerfile
  └── train.csv   # Downloaded Kaggle dataset renamed as train.csv
```


***

### 2. `requirements.txt`

```txt
flask
pandas
scikit-learn
joblib
```


***

### 3. `train.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load dataset
df = pd.read_csv("train.csv")

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode(), inplace=True)
df['Dependents'].fillna(df['Dependents'].mode(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode(), inplace=True)

# Encode categorical variables
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target variable
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

# Features and target
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and columns list
dump(model, 'model.joblib')
dump(list(X_train.columns), 'columns.joblib')

print("Training complete")
```


***

### 4. `app.py`

```python
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)
model = load('model.joblib')
columns = load('columns.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    # Add missing columns with 0
    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[columns]
    prediction = model.predict(input_df)[0]
    return jsonify({'loan_default_prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```


***

### 5. `Dockerfile`

```Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.py app.py train.csv ./

RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
```


***

### 6. Build and Run Docker Container

In your project root directory, run:

```bash
docker build -t loan-default-risk-app .
docker run -p 5000:5000 loan-default-risk-app
```


***

### 7. Test the API Locally

Sample input JSON (feature keys must exactly match the encoded feature names as in training):

```python
import requests

sample_input = {
    "Gender": 1,
    "Married": 0,
    "Dependents": 0,
    "Education": 1,
    "Self_Employed": 0,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 200,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 2
}

response = requests.post('http://localhost:5000/predict', json=sample_input)
print(response.json())
```


***

### Notes:

- Categorical features have been label encoded; ensure client input matches encoded integers.
- The API returns `1` for approved (no default) and `0` for default.
- Extend the solution with validation, logging, monitoring, and improved preprocessing for production readiness.

***

This solution provides a full ML lifecycle setup for loan default risk prediction ready to run locally using Docker.

