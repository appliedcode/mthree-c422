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
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None

    def save_baseline_stats(self, stats):
        with open(self.baseline_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def compute_text_features(self, texts):
        features = {
            'avg_length': np.mean([len(text) for text in texts]),
            'avg_word_count': np.mean([len(text.split()) for text in texts]),
            'avg_uppercase_ratio': np.mean([sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts]),
            'avg_digit_ratio': np.mean([sum(1 for c in text if c.isdigit()) / max(len(text), 1) for text in texts]),
            'avg_special_char_ratio': np.mean([sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1) for text in texts]),
        }
        return features

    def compute_label_distribution(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {label: count / total for label, count in zip(unique, counts)}

    def detect_data_drift(self, current_data, baseline_stats):
        current_features = self.compute_text_features(current_data['text'])
        baseline_features = baseline_stats.get('text_features', {})

        drift_scores = {}
        for feature, current_value in current_features.items():
            baseline_value = baseline_features.get(feature, current_value)
            drift_score = abs(current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
            drift_scores[feature] = drift_score

        avg_drift_score = np.mean(list(drift_scores.values()))
        data_drift_detected = avg_drift_score > self.threshold

        return data_drift_detected, avg_drift_score, drift_scores

    def detect_concept_drift(self, current_data, baseline_stats):
        current_dist = self.compute_label_distribution(current_data['label'])
        baseline_dist = baseline_stats.get('label_distribution', current_dist)

        drift_score = 0
        for label in set(current_dist) | set(baseline_dist):
            p = current_dist.get(label, 1e-3)
            q = baseline_dist.get(label, 1e-3)
            drift_score += p * np.log(p / q)

        concept_drift_detected = drift_score > self.threshold
        return concept_drift_detected, drift_score

    def semantic_drift_detection(self, current_texts, baseline_file='baseline_embeddings.joblib'):
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        current_embeddings = vectorizer.fit_transform(current_texts)

        if os.path.exists(baseline_file):
            data = joblib.load(baseline_file)
            base_emb = data['embeddings']
            base_vec = data['vectorizer']
            try:
                curr_emb_base = base_vec.transform(current_texts)
                sim = cosine_similarity(curr_emb_base.mean(axis=0), base_emb.mean(axis=0))[0, 0]
                semantic_drift_score = 1 - sim
                semantic_drift_detected = semantic_drift_score > self.threshold
            except:
                semantic_drift_detected, semantic_drift_score = True, 1.0
        else:
            joblib.dump({'embeddings': current_embeddings, 'vectorizer': vectorizer}, baseline_file)
            semantic_drift_detected, semantic_drift_score = False, 0.0

        return semantic_drift_detected, semantic_drift_score


def main():
    parser = argparse.ArgumentParser(description='Detect drift in data')
    parser.add_argument('--threshold', type=float, default=0.1, help='Drift detection threshold')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining regardless of drift')
    args = parser.parse_args()

    print("Loading current dataset...")
    data = pd.read_csv('current_data.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']

    detector = DriftDetector(threshold=args.threshold)
    baseline_stats = detector.load_baseline_stats()

    if baseline_stats is None:
        print("No baseline found. Creating baseline from current data...")
        baseline_stats = {
            'timestamp': datetime.now().isoformat(),
            'text_features': detector.compute_text_features(data['text']),
            'label_distribution': detector.compute_label_distribution(data['label']),
            'sample_count': len(data)
        }
        detector.save_baseline_stats(baseline_stats)

        out = os.environ['GITHUB_OUTPUT']
        with open(out, 'a') as fh:
            fh.write("data_drift_detected=false\n")
            fh.write("concept_drift_detected=false\n")
            fh.write("drift_score=0.0\n")
            fh.write("should_retrain=false\n")

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

    data_drift_detected, data_drift_score, _ = detector.detect_data_drift(data, baseline_stats)
    concept_drift_detected, concept_drift_score = detector.detect_concept_drift(data, baseline_stats)
    semantic_drift_detected, semantic_drift_score = detector.semantic_drift_detection(data['text'].tolist())

    overall_drift_score = (data_drift_score * 0.4 +
                           concept_drift_score * 0.3 +
                           semantic_drift_score * 0.3)

    should_retrain = (data_drift_detected or
                      concept_drift_detected or
                      semantic_drift_detected or
                      args.force_retrain)

    print("\n=== DRIFT DETECTION RESULTS ===")
    print(f"Data Drift: {data_drift_detected} (score {data_drift_score:.4f})")
    print(f"Concept Drift: {concept_drift_detected} (score {concept_drift_score:.4f})")
    print(f"Semantic Drift: {semantic_drift_detected} (score {semantic_drift_score:.4f})")
    print(f"Overall Drift Score: {overall_drift_score:.4f}")
    print(f"Should Retrain: {should_retrain}")

    out = os.environ['GITHUB_OUTPUT']
    with open(out, 'a') as fh:
        fh.write(f"data_drift_detected={str(data_drift_detected).lower()}\n")
        fh.write(f"concept_drift_detected={str(concept_drift_detected).lower()}\n")
        fh.write(f"drift_score={overall_drift_score:.4f}\n")
        fh.write(f"should_retrain={str(should_retrain).lower()}\n")

    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline_timestamp': baseline_stats['timestamp'],
        'threshold': args.threshold,
        'data_drift': {'detected': data_drift_detected, 'score': data_drift_score},
        'concept_drift': {'detected': concept_drift_detected, 'score': concept_drift_score},
        'semantic_drift': {'detected': semantic_drift_detected, 'score': semantic_drift_score},
        'overall_drift_score': overall_drift_score,
        'should_retrain': should_retrain,
        'current_data_stats': {
            'sample_count': len(data),
            'text_features': detector.compute_text_features(data['text']),
            'label_distribution': detector.compute_label_distribution(data['label'])
        }
    }
    with open('drift_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    with open('drift_visualizations.png', 'w') as f:
        f.write("Drift visualization placeholder")

    print("\nDrift detection complete; report saved to drift_report.json")


if __name__ == "__main__":
    main()
