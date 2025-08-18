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