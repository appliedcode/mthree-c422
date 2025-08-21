import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset loaded with {len(data)} rows")
print(f"Columns: {list(data.columns)}")
print(f"First few rows:\n{data.head()}")

# Ensure we have enough data
if len(data) < 10:
    raise ValueError(f"Dataset too small: only {len(data)} rows")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'spam_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model trained and saved successfully!")
