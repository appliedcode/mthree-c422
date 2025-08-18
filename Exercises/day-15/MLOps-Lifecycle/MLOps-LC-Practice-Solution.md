## Solution: Customer Churn Prediction with Docker


***

### 1. Setup Project Structure

Create this folder layout:

```
telco-churn/
  ├── app.py
  ├── train.py
  ├── requirements.txt
  └── Dockerfile
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

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing or erroneous TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col != 'customerID' and col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Encode target
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Features and target
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and columns
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
    input_json = request.get_json()
    input_df = pd.DataFrame([input_json])

    # Add missing columns with zeros
    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[columns]
    prediction = model.predict(input_df)[0]
    return jsonify({'churn_prediction': int(prediction)})

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

COPY train.py app.py WA_Fn-UseC_-Telco-Customer-Churn.csv ./

RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
```


***

### 6. Build and Run the Docker Container

In terminal at the project root:

```bash
docker build -t telco-churn-app .
docker run -p 5000:5000 telco-churn-app
```


***

### 7. Test the API Locally

Use Python or tools like Postman:

```python
import requests

sample_input = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 20,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "Contract": 0,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 70.35,
    "TotalCharges": 1397.475
}

response = requests.post('http://localhost:5000/predict', json=sample_input)
print(response.json())
```


***

## Notes

- Categorical columns were label encoded to integers for simplicity.
- Input JSON keys must match the encoded features the model expects.
- This setup ensures training and serving in confined reproducible Docker environment.
- Model evaluation and improved feature engineering can be added as extensions.

***

This completes your full ML lifecycle solution for the Telecom Customer Churn use case, ready to run locally using Docker.

