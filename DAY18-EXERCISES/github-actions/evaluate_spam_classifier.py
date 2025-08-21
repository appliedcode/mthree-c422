import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Load dataset (same preprocessing as training)
data = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset loaded with {len(data)} rows")

# Select only the first two columns (v1 and v2) and rename them
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

print(f"After cleaning - Dataset shape: {data.shape}")
print(f"Label distribution:\n{data['label'].value_counts()}")

# Train/test split (same as training for consistent evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

print(f"Test set size: {len(X_test)}")

# Load vectorizer and model
try:
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('spam_model.joblib')
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit(1)

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f"\n=== MODEL EVALUATION RESULTS ===")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")

# Detailed classification report
print(f"\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Show some example predictions
print(f"\n=== SAMPLE PREDICTIONS ===")
sample_indices = [0, 1, 2, 3, 4]
for i in sample_indices:
    if i < len(X_test):
        text_sample = X_test.iloc[i][:50] + "..." if len(X_test.iloc[i]) > 50 else X_test.iloc[i]
        print(f"Text: {text_sample}")
        print(f"Actual: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
        print("-" * 50)