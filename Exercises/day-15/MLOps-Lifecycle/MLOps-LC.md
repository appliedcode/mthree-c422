## Lab Exercise: ML Lifecycle on Local Machine with Docker


***

### 1. Setup Your Project Directory

**Create this structure:**

```
ml-lifecycle-lab/
  ├── app.py            # Flask API for model serving
  ├── train.py          # Model training script
  ├── requirements.txt  # Python dependencies
  └── Dockerfile        # For building Docker image
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
from joblib import dump

# Load data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv(url, header=None, names=columns, na_values=' ?')
df = df.dropna()
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model and column list
dump(model, 'model.joblib')
dump(list(X_train.columns), 'columns.joblib')
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
    for col in columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[columns]
    prediction = model.predict(input_df)[0]
    return jsonify({'prediction': int(prediction)})

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
COPY train.py app.py ./
RUN python train.py
EXPOSE 5000
CMD ["python", "app.py"]
```


***

### 6. Build and Run the Docker Container

**Open your terminal in the project directory:**

```bash
docker build -t ml-lifecycle-lab .
docker run -p 5000:5000 ml-lifecycle-lab
```


***

### 7. Test the API Locally

**From a second terminal or Postman:**

```python
import requests
sample = {'age': 39, 'fnlwgt': 77516, 'education-num': 13, 'capital-gain': 2174,
          'capital-loss': 0, 'hours-per-week': 40, 
          # Include other one-hot encoded fields as needed for your columns 
          }
response = requests.post('http://localhost:5000/predict', json=sample)
print(response.json())
```

*Tip: Use keys from `columns.joblib` to build a full sample dict.*

***

## Optional Extensions

- Add logging in `app.py` to simulate model monitoring.
- Automate retraining trigger on new data using shell scripts or a CRON job (simulate pipeline triggers/scheduling).
- Use Docker Compose to add Prometheus/Grafana containers for system metrics.

***

## What You Have Practiced

- **Data ingestion, prep, training, evaluation**
- **Model artifact management**
- **Deployment as a REST API**
- **Testing and monitoring basics**
- **Portable reproducibility via Docker**

***
