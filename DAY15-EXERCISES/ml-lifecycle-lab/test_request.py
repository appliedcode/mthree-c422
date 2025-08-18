import requests

sample = {
    'age': 39, 
    'fnlwgt': 77516, 
    'education-num': 13, 
    'capital-gain': 2174,
    'capital-loss': 0, 
    'hours-per-week': 40,
    # You might need to include one-hot encoded fields like workclass_Private=1, etc.
}

response = requests.post('http://localhost:5000/predict', json=sample)
print(response.json())
