import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import argparse
from datetime import datetime

def retrain_with_drift_awareness(drift_score, data_drift, concept_drift):
    """Retrain model with drift-aware strategies"""
    
    print(f"Starting drift-aware retraining...")
    print(f"Drift Score: {drift_score}")
    print(f"Data Drift: {data_drift}")
    print(f"Concept Drift: {concept_drift}")
    
    # Load dataset
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    
    print(f"Dataset loaded: {len(data)} samples")
    
    # Drift-aware preprocessing
    if data_drift == 'true':
        print("Applying data drift mitigation strategies...")
        # Use more robust vectorization parameters
        max_features = 8000  # Increased vocabulary
        min_df = 2  # Minimum document frequency
        max_df = 0.95  # Maximum document frequency
    else:
        max_features = 5000
        min_df = 1
        max_df = 1.0
    
    # Concept drift handling
    if concept_drift == 'true':
        print("Applying concept drift mitigation strategies...")
        # Use stratified sampling to ensure balanced representation
        stratify_param = data['label']
        test_size = 0.25  # Larger test set for better evaluation
    else:
        stratify_param = None
        test_size = 0.2
    
    # Train/test split with drift considerations
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], 
        test_size=test_size, 
        random_state=42,
        stratify=stratify_param
    )
    
    # Vectorize with drift-aware parameters
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        ngram_range=(1, 2) if float(drift_score) > 0.15 else (1, 1)  # Use bigrams for high drift
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Model selection based on drift severity
    if float(drift_score) > 0.2:
        print("High drift detected - using more robust model parameters")
        model = MultinomialNB(alpha=0.5)  # Higher smoothing for stability
    else:
        model = MultinomialNB(alpha=1.0)  # Standard smoothing
    
    # Train model
    print("Training drift-aware model...")
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    joblib.dump(model, 'spam_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    # Save accuracy for GitHub Actions
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")
    
    # Save training metadata with drift info
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'vocabulary_size': len(vectorizer.vocabulary_),
        'drift_info': {
            'drift_score': float(drift_score),
            'data_drift': data_drift,
            'concept_drift': concept_drift,
            'mitigation_applied': True
        },
        'model_parameters': {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': vectorizer.ngram_range,
            'alpha': model.alpha
        }
    }
    
    with open('training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Drift-aware model training completed successfully!")
    
    # Update baseline statistics with new data
    from drift_detection import DriftDetector
    detector = DriftDetector()
    new_baseline = {
        'timestamp': datetime.now().isoformat(),
        'text_features': detector.compute_text_features(data['text']),
        'label_distribution': detector.compute_label_distribution(data['label']),
        'sample_count': len(data),
        'model_version': f"drift-aware-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }
    detector.save_baseline_stats(new_baseline)
    print("Updated baseline statistics")

def main():
    parser = argparse.ArgumentParser(description='Train model with drift awareness')
    parser.add_argument('--drift-score', type=str, required=True, help='Drift score')
    parser.add_argument('--data-drift', type=str, default='false', help='Data drift detected')
    parser.add_argument('--concept-drift', type=str, default='false', help='Concept drift detected')
    args = parser.parse_args()
    
    retrain_with_drift_awareness(args.drift_score, args.data_drift, args.concept_drift)

if __name__ == "__main__":
    main()
