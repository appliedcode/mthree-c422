import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class DriftDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.baseline_file = 'baseline_stats.json'
        
    def load_baseline_stats(self):
        """Load baseline statistics or create new ones"""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_baseline_stats(self, stats):
        """Save baseline statistics"""
        with open(self.baseline_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def compute_text_features(self, texts):
        """Compute text-based features for drift detection"""
        features = {
            'avg_length': np.mean([len(text) for text in texts]),
            'avg_word_count': np.mean([len(text.split()) for text in texts]),
            'avg_uppercase_ratio': np.mean([sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts]),
            'avg_digit_ratio': np.mean([sum(1 for c in text if c.isdigit()) / max(len(text), 1) for text in texts]),
            'avg_special_char_ratio': np.mean([sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1) for text in texts]),
        }
        return features
    
    def compute_label_distribution(self, labels):
        """Compute label distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {label: count/total for label, count in zip(unique, counts)}
    
    def detect_data_drift(self, current_data, baseline_stats):
        """Detect data drift using statistical tests"""
        current_features = self.compute_text_features(current_data['text'])
        baseline_features = baseline_stats.get('text_features', {})
        
        drift_scores = {}
        for feature, current_value in current_features.items():
            baseline_value = baseline_features.get(feature, current_value)
            # Calculate relative change
            if baseline_value != 0:
                drift_score = abs(current_value - baseline_value) / baseline_value
            else:
                drift_score = 0
            drift_scores[feature] = drift_score
        
        avg_drift_score = np.mean(list(drift_scores.values()))
        data_drift_detected = avg_drift_score > self.threshold
        
        return data_drift_detected, avg_drift_score, drift_scores
    
    def detect_concept_drift(self, current_data, baseline_stats):
        """Detect concept drift in label distribution"""
        current_distribution = self.compute_label_distribution(current_data['label'])
        baseline_distribution = baseline_stats.get('label_distribution', current_distribution)
        
        # Calculate KL divergence or similar measure
        drift_score = 0
        for label in set(list(current_distribution.keys()) + list(baseline_distribution.keys())):
            p = current_distribution.get(label, 0.001)  # Small epsilon to avoid log(0)
            q = baseline_distribution.get(label, 0.001)
            drift_score += p * np.log(p / q)
        
        concept_drift_detected = drift_score > self.threshold
        return concept_drift_detected, drift_score
    
    def semantic_drift_detection(self, current_texts, baseline_file='baseline_embeddings.joblib'):
        """Detect semantic drift using text embeddings"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        current_embeddings = vectorizer.fit_transform(current_texts)
        
        if os.path.exists(baseline_file):
            baseline_data = joblib.load(baseline_file)
            baseline_embeddings = baseline_data['embeddings']
            baseline_vectorizer = baseline_data['vectorizer']
            
            # Transform current texts using baseline vocabulary
            try:
                current_embeddings_baseline = baseline_vectorizer.transform(current_texts)
                similarity = cosine_similarity(
                    current_embeddings_baseline.mean(axis=0), 
                    baseline_embeddings.mean(axis=0)
                )[0, 0]
                semantic_drift_score = 1 - similarity
                semantic_drift_detected = semantic_drift_score > self.threshold
            except:
                semantic_drift_detected = True
                semantic_drift_score = 1.0
        else:
            # Save baseline embeddings
            joblib.dump({
                'embeddings': current_embeddings,
                'vectorizer': vectorizer
            }, baseline_file)
            semantic_drift_detected = False
            semantic_drift_score = 0.0
        
        return semantic_drift_detected, semantic_drift_score

def main():
    parser = argparse.ArgumentParser(description='Detect drift in data')
    parser.add_argument('--threshold', type=float, default=0.1, help='Drift detection threshold')
    parser.add_argument('--force-retrain', type=str, default='false', help='Force retraining')
    args = parser.parse_args()
    
    # Load current data
    print("Loading current dataset...")
    data = pd.read_csv('current_data.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    
    # Initialize drift detector
    detector = DriftDetector(threshold=args.threshold)
    
    # Load baseline statistics
    baseline_stats = detector.load_baseline_stats()
    
    if baseline_stats is None:
        print("No baseline found. Creating baseline from current data...")
        # Create baseline statistics
        baseline_stats = {
            'timestamp': datetime.now().isoformat(),
            'text_features': detector.compute_text_features(data['text']),
            'label_distribution': detector.compute_label_distribution(data['label']),
            'sample_count': len(data)
        }
        detector.save_baseline_stats(baseline_stats)
        
        # Set outputs for GitHub Actions
        print("::set-output name=data_drift_detected::false")
        print("::set-output name=concept_drift_detected::false")
        print("::set-output name=drift_score::0.0")
        print("::set-output name=should_retrain::false")
        
        # Save drift report
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_created': True,
            'drift_detected': False,
            'message': 'Baseline statistics created from current data'
        }
        
        with open('drift_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return
    
    print(f"Baseline loaded from {baseline_stats['timestamp']}")
    print(f"Using drift threshold: {args.threshold}")
    
    # Detect data drift
    data_drift_detected, data_drift_score, feature_drift_scores = detector.detect_data_drift(data, baseline_stats)
    
    # Detect concept drift
    concept_drift_detected, concept_drift_score = detector.detect_concept_drift(data, baseline_stats)
    
    # Detect semantic drift
    semantic_drift_detected, semantic_drift_score = detector.semantic_drift_detection(data['text'].tolist())
    
    # Overall drift score (weighted average)
    overall_drift_score = (data_drift_score * 0.4 + concept_drift_score * 0.3 + semantic_drift_score * 0.3)
    
    # Determine if retraining is needed
    should_retrain = (data_drift_detected or concept_drift_detected or semantic_drift_detected or 
                     args.force_retrain.lower() == 'true')
    
    # Print results
    print(f"\n=== DRIFT DETECTION RESULTS ===")
    print(f"Data Drift Detected: {data_drift_detected} (score: {data_drift_score:.4f})")
    print(f"Concept Drift Detected: {concept_drift_detected} (score: {concept_drift_score:.4f})")
    print(f"Semantic Drift Detected: {semantic_drift_detected} (score: {semantic_drift_score:.4f})")
    print(f"Overall Drift Score: {overall_drift_score:.4f}")
    print(f"Should Retrain: {should_retrain}")
    
    # Set outputs for GitHub Actions
    with open(os.environ.get('GITHUB_OUTPUT', '/dev/stdout'), 'a') as f:
        f.write(f"data_drift_detected={str(data_drift_detected).lower()}\n")
        f.write(f"concept_drift_detected={str(concept_drift_detected).lower()}\n")
        f.write(f"drift_score={overall_drift_score:.4f}\n")
        f.write(f"should_retrain={str(should_retrain).lower()}\n")
    
    # Create detailed drift report
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline_timestamp': baseline_stats['timestamp'],
        'threshold': args.threshold,
        'data_drift': {
            'detected': data_drift_detected,
            'score': data_drift_score,
            'feature_scores': feature_drift_scores
        },
        'concept_drift': {
            'detected': concept_drift_detected,
            'score': concept_drift_score
        },
        'semantic_drift': {
            'detected': semantic_drift_detected,
            'score': semantic_drift_score
        },
        'overall_drift_score': overall_drift_score,
        'should_retrain': should_retrain,
        'current_data_stats': {
            'sample_count': len(data),
            'text_features': detector.compute_text_features(data['text']),
            'label_distribution': detector.compute_label_distribution(data['label'])
        }
    }
    
    # Save drift report
    with open('drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create visualization placeholder
    with open('drift_visualizations.png', 'w') as f:
        f.write("Drift visualization would be generated here")
    
    print(f"\nDrift detection completed! Report saved to drift_report.json")

if __name__ == "__main__":
    main()
