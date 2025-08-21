import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load dataset (same as training for consistent split)
data = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset loaded with {len(data)} rows")

# Train/test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('spam_model.joblib')

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")
