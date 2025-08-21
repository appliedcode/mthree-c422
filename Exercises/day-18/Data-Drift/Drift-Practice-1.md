# Lab Exercise: Automated Model Retraining Based on Concept/Data Drift Detection


***

## Objective

Create a GitHub Actions workflow system that:

1. Monitors for concept drift and data drift
2. Automatically triggers model retraining when drift is detected
3. Compares new data distributions with baseline
4. Performs sample predictions and drift analysis

***

## Step 1: Create the Drift Detection Workflow

### File: `.github/workflows/drift-detection.yml`

```yaml
name: Drift Detection and Model Retraining

on:
  workflow_dispatch:
    inputs:
      drift_threshold:
        description: 'Drift detection threshold (0.0-1.0)'
        required: true
        default: '0.1'
        type: string
      force_retrain:
        description: 'Force retraining regardless of drift'
        required: false
        default: false
        type: boolean
  schedule:
    # Run drift detection daily at 6 AM UTC
    - cron: '0 6 * * *'
  repository_dispatch:
    types: [check-drift]

jobs:
  detect_drift:
    runs-on: ubuntu-latest
    
    outputs:
      data_drift_detected: ${{ steps.drift_check.outputs.data_drift_detected }}
      concept_drift_detected: ${{ steps.drift_check.outputs.concept_drift_detected }}
      drift_score: ${{ steps.drift_check.outputs.drift_score }}
      should_retrain: ${{ steps.drift_check.outputs.should_retrain }}

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: 3.10.18

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r app/requirements.txt

    - name: Download current dataset
      run: |
        curl -L -o current_data.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

    - name: Detect drift
      id: drift_check
      run: |
        CMD="python app/drift_detection.py --threshold ${{ github.event.inputs.drift_threshold }}"
        if [ "${{ github.event.inputs.force_retrain }}" = "true" ]; then
          CMD="$CMD --force-retrain"
        fi
        echo "Running: $CMD"
        $CMD

    - name: Upload drift report
      uses: actions/upload-artifact@v4
      with:
        name: drift-analysis-${{ github.run_id }}
        path: |
          drift_report.json
          drift_visualizations.png
          baseline_stats.json

    - name: Comment on drift detection results
      if: github.event_name == 'workflow_dispatch'
      uses: actions/github-script@v6
      with:
        script: |
          const driftDetected = '${{ steps.drift_check.outputs.data_drift_detected }}' === 'true' || 
                               '${{ steps.drift_check.outputs.concept_drift_detected }}' === 'true';
          const driftScore = '${{ steps.drift_check.outputs.drift_score }}';
          
          const message = driftDetected ? 
            `🚨 **Drift Detected!** 
             - Data Drift: ${{ steps.drift_check.outputs.data_drift_detected }}
             - Concept Drift: ${{ steps.drift_check.outputs.concept_drift_detected }}
             - Drift Score: ${driftScore}
             - Retraining will be triggered automatically.` :
            `✅ **No Significant Drift Detected**
             - Drift Score: ${driftScore}
             - Model remains stable.`;
          
          console.log(message);

  trigger_retraining:
    needs: detect_drift
    if: needs.detect_drift.outputs.should_retrain == 'true'
    uses: ./.github/workflows/retrain-with-drift.yml
    with:
      drift_score: ${{ needs.detect_drift.outputs.drift_score }}
      data_drift: ${{ needs.detect_drift.outputs.data_drift_detected }}
      concept_drift: ${{ needs.detect_drift.outputs.concept_drift_detected }}
```


***

## Step 2: Create the Drift-Aware Retraining Workflow

### File: `.github/workflows/retrain-with-drift.yml`

```yaml
name: Drift-Aware Model Retraining

on:
  workflow_call:
    inputs:
      drift_score:
        required: true
        type: string
      data_drift:
        required: true
        type: string
      concept_drift:
        required: true
        type: string
  workflow_dispatch:
    inputs:
      drift_score:
        description: 'Drift score that triggered retraining'
        required: true
        default: '0.15'
        type: string

jobs:
  retrain_model:
    runs-on: ubuntu-latest
    
    outputs:
      model_version: ${{ steps.version.outputs.version }}
      accuracy: ${{ steps.train.outputs.accuracy }}
      drift_score: ${{ inputs.drift_score }}

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: 3.10.18

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r app/requirements.txt

    - name: Set model version with drift info
      id: version
      run: |
        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        DRIFT_SCORE=${{ inputs.drift_score }}
        echo "version=v${TIMESTAMP}-drift${DRIFT_SCORE}" >> $GITHUB_OUTPUT

    - name: Download dataset
      run: |
        curl -L -o spam.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

    - name: Retrain model with drift awareness
      id: train
      run: |
        python app/drift_aware_training.py \
          --drift-score ${{ inputs.drift_score }} \
          --data-drift ${{ inputs.data_drift || 'false' }} \
          --concept-drift ${{ inputs.concept_drift || 'false' }}
        echo "accuracy=$(cat model_accuracy.txt)" >> $GITHUB_OUTPUT

    - name: Generate drift-aware model report
      run: |
        python app/generate_drift_report.py ${{ steps.version.outputs.version }} ${{ inputs.drift_score }}

    - name: Upload model artifacts
      uses: actions/upload-artifact@v4
      with:
        name: drift-retrained-model-${{ steps.version.outputs.version }}
        path: |
          spam_model.joblib
          vectorizer.joblib
          drift_model_report.json
          baseline_stats.json

    - name: Trigger prediction and validation
      if: success()
      uses: peter-evans/repository-dispatch@v2
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        event-type: validate-drift-model
        client-payload: |
          {
            "model_version": "${{ steps.version.outputs.version }}",
            "accuracy": "${{ steps.train.outputs.accuracy }}",
            "drift_score": "${{ inputs.drift_score }}",
            "run_id": "${{ github.run_id }}",
            "retrain_reason": "drift_detection"
          }
```


***

## Step 3: Create Drift Detection Script

### File: `app/drift_detection.py`

```python
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

        # If user requested force-retrain, honor it even on first run
        should_retrain = args.force_retrain

        out = os.environ['GITHUB_OUTPUT']
        with open(out, 'a') as fh:
            fh.write("data_drift_detected=false\n")
            fh.write("concept_drift_detected=false\n")
            fh.write("drift_score=0.0\n")
            fh.write(f"should_retrain={str(should_retrain).lower()}\n")

        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_created': True,
            'drift_detected': False,
            'force_retrain': should_retrain,
            'message': 'Baseline created—or force retrain if requested'
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
```


***

## Step 4: Create Drift-Aware Training Script

### File: `app/drift_aware_training.py`

```python
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
```


***

## Step 5: Create Drift Report Generator

### File: `app/generate_drift_report.py`

```python
import json
import sys
from datetime import datetime

def generate_drift_report(model_version, drift_score):
    """Generate a comprehensive drift-aware model report"""
    
    # Load training metadata
    try:
        with open('training_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
    
    # Load drift report if available
    try:
        with open('drift_report.json', 'r') as f:
            drift_data = json.load(f)
    except FileNotFoundError:
        drift_data = {}
    
    # Create comprehensive report
    report = {
        'model_version': model_version,
        'report_generated': datetime.now().isoformat(),
        'drift_analysis': {
            'trigger_drift_score': float(drift_score),
            'drift_details': drift_data.get('data_drift', {}),
            'concept_drift_details': drift_data.get('concept_drift', {}),
            'semantic_drift_details': drift_data.get('semantic_drift', {}),
            'mitigation_applied': metadata.get('drift_info', {}).get('mitigation_applied', False)
        },
        'model_performance': {
            'accuracy': metadata.get('accuracy', 'N/A'),
            'training_samples': metadata.get('training_samples', 'N/A'),
            'test_samples': metadata.get('test_samples', 'N/A'),
            'vocabulary_size': metadata.get('vocabulary_size', 'N/A')
        },
        'model_parameters': metadata.get('model_parameters', {}),
        'recommendations': generate_recommendations(drift_score, drift_data),
        'status': 'completed',
        'files': [
            'spam_model.joblib',
            'vectorizer.joblib',
            'baseline_stats.json'
        ]
    }
    
    # Save report
    with open('drift_model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Drift-aware model report generated for version: {model_version}")
    print(f"Drift score that triggered retraining: {drift_score}")
    print(f"Model accuracy: {metadata.get('accuracy', 'N/A')}")

def generate_recommendations(drift_score, drift_data):
    """Generate recommendations based on drift analysis"""
    recommendations = []
    
    drift_score_float = float(drift_score)
    
    if drift_score_float > 0.3:
        recommendations.append("High drift detected - consider more frequent monitoring")
        recommendations.append("Evaluate data collection process for potential issues")
    elif drift_score_float > 0.15:
        recommendations.append("Moderate drift detected - monitor closely")
        recommendations.append("Consider feature engineering improvements")
    else:
        recommendations.append("Low drift detected - normal monitoring schedule sufficient")
    
    # Specific recommendations based on drift type
    if drift_data.get('data_drift', {}).get('detected', False):
        recommendations.append("Data distribution changed - review feature preprocessing")
    
    if drift_data.get('concept_drift', {}).get('detected', False):
        recommendations.append("Label distribution changed - review labeling criteria")
    
    if drift_data.get('semantic_drift', {}).get('detected', False):
        recommendations.append("Semantic patterns changed - consider vocabulary updates")
    
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) > 2:
        model_version = sys.argv[1]
        drift_score = sys.argv[2]
    else:
        model_version = "unknown"
        drift_score = "0.0"
    
    generate_drift_report(model_version, drift_score)
```


***

## Step 6: Create Model Validation Workflow

### File: `.github/workflows/validate-drift-model.yml`

```yaml
name: Validate Drift-Retrained Model

on:
  repository_dispatch:
    types: [validate-drift-model]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: 3.10.18

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r app/requirements.txt

    - name: Download model artifacts
      uses: actions/download-artifact@v4
      with:
        name: drift-retrained-model-${{ github.event.client_payload.model_version }}
        github-token: ${{ secrets.GITHUB_TOKEN }}
        run-id: ${{ github.event.client_payload.run_id }}

    - name: Run validation and predictions
      run: |
        python app/validate_drift_model.py \
          --model-version "${{ github.event.client_payload.model_version }}" \
          --drift-score "${{ github.event.client_payload.drift_score }}"

    - name: Upload validation results
      uses: actions/upload-artifact@v4
      with:
        name: validation-results-${{ github.event.client_payload.model_version }}
        path: |
          validation_report.json
          drift_predictions.txt
          model_comparison.json
```


***

## Step 7: Update Requirements

### File: `app/requirements.txt`

```
pandas
scikit-learn
joblib
numpy
scipy
matplotlib
seaborn
```


***

## How to Use This Lab Exercise

### 1. **Manual Drift Detection**:

```bash
# Go to Actions → "Drift Detection and Model Retraining" → "Run workflow"
# Set drift threshold (e.g., 0.1) and optionally force retraining
```


### 2. **Scheduled Monitoring**:

- The workflow runs daily at 6 AM UTC to check for drift
- Automatically triggers retraining if drift is detected


### 3. **Drift Types Detected**:

- **Data Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in label distributions
- **Semantic Drift**: Changes in text patterns and vocabulary


### 4. **Monitoring Results**:

- Check drift reports in workflow artifacts
- Review model comparison metrics
- Monitor retraining frequency and triggers

***
