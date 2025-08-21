# **\# Lab Exercise: Automated Model Retraining Based on Concept/Data Drift Detection**

\*\*\*

**\#\# Objective**

Create a GitHub Actions workflow system that:

1. Monitors for concept drift and data drift
2. Automatically triggers model retraining when drift is detected
3. Compares new data distributions with baseline
4. Performs sample predictions and drift analysis

\*\*\*

**\#\# Step 1: Create the Drift Detection Workflow**

**\#\#\# File: \`.github/workflows/drift-detection.yml\`**

\`\`\`yaml
name: Drift Detection and Model Retraining

on:
workflow\_dispatch:
inputs:
drift\_threshold:
description: 'Drift detection threshold (0.0-1.0)'
required: true
default: '0.1'
type: string
force\_retrain:
description: 'Force retraining regardless of drift'
required: false
default: false
type: boolean
schedule:
\# Run drift detection daily at 6 AM UTC
- cron: '0 6 \* \* \*'
repository\_dispatch:
types: [check-drift]

jobs:
detect\_drift:
runs-on: ubuntu-latest

    outputs:
      data\_drift\_detected: ${{ steps.drift\_check.outputs.data\_drift\_detected }}
      concept\_drift\_detected: ${{ steps.drift\_check.outputs.concept\_drift\_detected }}
      drift\_score: ${{ steps.drift\_check.outputs.drift\_score }}
      should\_retrain: ${{ steps.drift\_check.outputs.should\_retrain }}
    
    
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
        curl -L -o current\_data.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    
    
    - name: Detect drift
      id: drift\_check
      run: |
        CMD="python app/drift\_detection.py --threshold ${{ github.event.inputs.drift\_threshold }}"
        if [ "${{ github.event.inputs.force\_retrain }}" = "true" ]; then
          CMD="$CMD --force-retrain"
        fi
        echo "Running: $CMD"
        $CMD
    
    
    - name: Upload drift report
      uses: actions/upload-artifact@v4
      with:
        name: drift-analysis-${{ github.run\_id }}
        path: |
          drift\_report.json
          drift\_visualizations.png
          baseline\_stats.json
    
    
    - name: Comment on drift detection results
      if: github.event\_name == 'workflow\_dispatch'
      uses: actions/github-script@v6
      with:
        script: |
          const driftDetected = '${{ steps.drift\_check.outputs.data\_drift\_detected }}' === 'true' || 
                               '${{ steps.drift\_check.outputs.concept\_drift\_detected }}' === 'true';
          const driftScore = '${{ steps.drift\_check.outputs.drift\_score }}';
          
          const message = driftDetected ? 
            \`🚨 \*\*Drift Detected!\*\* 
             - Data Drift: ${{ steps.drift\_check.outputs.data\_drift\_detected }}
             - Concept Drift: ${{ steps.drift\_check.outputs.concept\_drift\_detected }}
             - Drift Score: ${driftScore}
             - Retraining will be triggered automatically.\` :
            \`✅ \*\*No Significant Drift Detected\*\*
             - Drift Score: ${driftScore}
             - Model remains stable.\`;
          
          console.log(message);
    trigger\_retraining:
needs: detect\_drift
if: needs.detect\_drift.outputs.should\_retrain == 'true'
uses: ./.github/workflows/retrain-with-drift.yml
with:
drift\_score: \${{ needs.detect\_drift.outputs.drift\_score }}
data\_drift: \${{ needs.detect\_drift.outputs.data\_drift\_detected }}
concept\_drift: \${{ needs.detect\_drift.outputs.concept\_drift\_detected }}
\`\`\`

\*\*\*

**\#\# Step 2: Create the Drift-Aware Retraining Workflow**

**\#\#\# File: \`.github/workflows/retrain-with-drift.yml\`**

\`\`\`yaml
name: Drift-Aware Model Retraining

on:
workflow\_call:
inputs:
drift\_score:
required: true
type: string
data\_drift:
required: true
type: string
concept\_drift:
required: true
type: string
workflow\_dispatch:
inputs:
drift\_score:
description: 'Drift score that triggered retraining'
required: true
default: '0.15'
type: string

jobs:
retrain\_model:
runs-on: ubuntu-latest

    outputs:
      model\_version: ${{ steps.version.outputs.version }}
      accuracy: ${{ steps.train.outputs.accuracy }}
      drift\_score: ${{ inputs.drift\_score }}
    
    
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
        DRIFT\_SCORE=${{ inputs.drift\_score }}
        echo "version=v${TIMESTAMP}-drift${DRIFT\_SCORE}" >> $GITHUB\_OUTPUT
    
    
    - name: Download dataset
      run: |
        curl -L -o spam.csv "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    
    
    - name: Retrain model with drift awareness
      id: train
      run: |
        python app/drift\_aware\_training.py \\
          --drift-score ${{ inputs.drift\_score }} \\
          --data-drift ${{ inputs.data\_drift || 'false' }} \\
          --concept-drift ${{ inputs.concept\_drift || 'false' }}
        echo "accuracy=$(cat model\_accuracy.txt)" >> $GITHUB\_OUTPUT
    
    
    - name: Generate drift-aware model report
      run: |
        python app/generate\_drift\_report.py ${{ steps.version.outputs.version }} ${{ inputs.drift\_score }}
    
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v4
      with:
        name: drift-retrained-model-${{ steps.version.outputs.version }}
        path: |
          spam\_model.joblib
          vectorizer.joblib
          drift\_model\_report.json
          baseline\_stats.json
    
    
    - name: Trigger prediction and validation
      if: success()
      uses: peter-evans/repository-dispatch@v2
      with:
        token: ${{ secrets.GITHUB\_TOKEN }}
        event-type: validate-drift-model
        client-payload: |
          {
            "model\_version": "${{ steps.version.outputs.version }}",
            "accuracy": "${{ steps.train.outputs.accuracy }}",
            "drift\_score": "${{ inputs.drift\_score }}",
            "run\_id": "${{ github.run\_id }}",
            "retrain\_reason": "drift\_detection"
          }
    \`\`\`

\*\*\*

**\#\# Step 3: Create Drift Detection Script**

**\#\#\# File: \`app/drift\_detection.py\`**

\`\`\`python
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from scipy import stats
from sklearn.feature\_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine\_similarity
import joblib
import os

class DriftDetector:
def \_\_init\_\_(self, threshold=0.1):
self.threshold = threshold
self.baseline\_file = 'baseline\_stats.json'

    def load\_baseline\_stats(self):
        if os.path.exists(self.baseline\_file):
            with open(self.baseline\_file, 'r') as f:
                return json.load(f)
        return None
    
    
    def save\_baseline\_stats(self, stats):
        with open(self.baseline\_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    
    def compute\_text\_features(self, texts):
        features = {
            'avg\_length': np.mean([len(text) for text in texts]),
            'avg\_word\_count': np.mean([len(text.split()) for text in texts]),
            'avg\_uppercase\_ratio': np.mean([sum(1 for c in text if c.isupper()) / max(len(text), 1) for text in texts]),
            'avg\_digit\_ratio': np.mean([sum(1 for c in text if c.isdigit()) / max(len(text), 1) for text in texts]),
            'avg\_special\_char\_ratio': np.mean([sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1) for text in texts]),
        }
        return features
    
    
    def compute\_label\_distribution(self, labels):
        unique, counts = np.unique(labels, return\_counts=True)
        total = len(labels)
        return {label: count / total for label, count in zip(unique, counts)}
    
    
    def detect\_data\_drift(self, current\_data, baseline\_stats):
        current\_features = self.compute\_text\_features(current\_data['text'])
        baseline\_features = baseline\_stats.get('text\_features', {})
    
    
        drift\_scores = {}
        for feature, current\_value in current\_features.items():
            baseline\_value = baseline\_features.get(feature, current\_value)
            drift\_score = abs(current\_value - baseline\_value) / baseline\_value if baseline\_value != 0 else 0
            drift\_scores[feature] = drift\_score
    
    
        avg\_drift\_score = np.mean(list(drift\_scores.values()))
        data\_drift\_detected = avg\_drift\_score > self.threshold
    
    
        return data\_drift\_detected, avg\_drift\_score, drift\_scores
    
    
    def detect\_concept\_drift(self, current\_data, baseline\_stats):
        current\_dist = self.compute\_label\_distribution(current\_data['label'])
        baseline\_dist = baseline\_stats.get('label\_distribution', current\_dist)
    
    
        drift\_score = 0
        for label in set(current\_dist) | set(baseline\_dist):
            p = current\_dist.get(label, 1e-3)
            q = baseline\_dist.get(label, 1e-3)
            drift\_score += p \* np.log(p / q)
    
    
        concept\_drift\_detected = drift\_score > self.threshold
        return concept\_drift\_detected, drift\_score
    
    
    def semantic\_drift\_detection(self, current\_texts, baseline\_file='baseline\_embeddings.joblib'):
        vectorizer = TfidfVectorizer(max\_features=1000, stop\_words='english')
        current\_embeddings = vectorizer.fit\_transform(current\_texts)
    
    
        if os.path.exists(baseline\_file):
            data = joblib.load(baseline\_file)
            base\_emb = data['embeddings']
            base\_vec = data['vectorizer']
            try:
                curr\_emb\_base = base\_vec.transform(current\_texts)
                sim = cosine\_similarity(curr\_emb\_base.mean(axis=0), base\_emb.mean(axis=0))[0, 0]
                semantic\_drift\_score = 1 - sim
                semantic\_drift\_detected = semantic\_drift\_score > self.threshold
            except:
                semantic\_drift\_detected, semantic\_drift\_score = True, 1.0
        else:
            joblib.dump({'embeddings': current\_embeddings, 'vectorizer': vectorizer}, baseline\_file)
            semantic\_drift\_detected, semantic\_drift\_score = False, 0.0
    
    
        return semantic\_drift\_detected, semantic\_drift\_score
    def main():
parser = argparse.ArgumentParser(description='Detect drift in data')
parser.add\_argument('--threshold', type=float, default=0.1, help='Drift detection threshold')
parser.add\_argument('--force-retrain', action='store\_true',
help='Force retraining regardless of drift')
args = parser.parse\_args()

    print("Loading current dataset...")
    data = pd.read\_csv('current\_data.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    
    
    detector = DriftDetector(threshold=args.threshold)
    baseline\_stats = detector.load\_baseline\_stats()
    
    
    if baseline\_stats is None:
        print("No baseline found. Creating baseline from current data...")
        baseline\_stats = {
            'timestamp': datetime.now().isoformat(),
            'text\_features': detector.compute\_text\_features(data['text']),
            'label\_distribution': detector.compute\_label\_distribution(data['label']),
            'sample\_count': len(data)
        }
        detector.save\_baseline\_stats(baseline\_stats)
    
    
        # If user requested force-retrain, honor it even on first run
        should\_retrain = args.force\_retrain
    
    
        out = os.environ['GITHUB\_OUTPUT']
        with open(out, 'a') as fh:
            fh.write("data\_drift\_detected=false\\n")
            fh.write("concept\_drift\_detected=false\\n")
            fh.write("drift\_score=0.0\\n")
            fh.write(f"should\_retrain={str(should\_retrain).lower()}\\n")
    
    
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline\_created': True,
            'drift\_detected': False,
            'force\_retrain': should\_retrain,
            'message': 'Baseline created—or force retrain if requested'
        }
        with open('drift\_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        return
    
    
    print(f"Baseline loaded from {baseline\_stats['timestamp']}")
    print(f"Using drift threshold: {args.threshold}")
    
    
    data\_drift\_detected, data\_drift\_score, \_ = detector.detect\_data\_drift(data, baseline\_stats)
    concept\_drift\_detected, concept\_drift\_score = detector.detect\_concept\_drift(data, baseline\_stats)
    semantic\_drift\_detected, semantic\_drift\_score = detector.semantic\_drift\_detection(data['text'].tolist())
    
    
    overall\_drift\_score = (data\_drift\_score \* 0.4 +
                           concept\_drift\_score \* 0.3 +
                           semantic\_drift\_score \* 0.3)
    
    
    should\_retrain = (data\_drift\_detected or
                      concept\_drift\_detected or
                      semantic\_drift\_detected or
                      args.force\_retrain)
    
    
    print("\\n=== DRIFT DETECTION RESULTS ===")
    print(f"Data Drift: {data\_drift\_detected} (score {data\_drift\_score:.4f})")
    print(f"Concept Drift: {concept\_drift\_detected} (score {concept\_drift\_score:.4f})")
    print(f"Semantic Drift: {semantic\_drift\_detected} (score {semantic\_drift\_score:.4f})")
    print(f"Overall Drift Score: {overall\_drift\_score:.4f}")
    print(f"Should Retrain: {should\_retrain}")
    
    
    out = os.environ['GITHUB\_OUTPUT']
    with open(out, 'a') as fh:
        fh.write(f"data\_drift\_detected={str(data\_drift\_detected).lower()}\\n")
        fh.write(f"concept\_drift\_detected={str(concept\_drift\_detected).lower()}\\n")
        fh.write(f"drift\_score={overall\_drift\_score:.4f}\\n")
        fh.write(f"should\_retrain={str(should\_retrain).lower()}\\n")
    
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'baseline\_timestamp': baseline\_stats['timestamp'],
        'threshold': args.threshold,
        'data\_drift': {'detected': data\_drift\_detected, 'score': data\_drift\_score},
        'concept\_drift': {'detected': concept\_drift\_detected, 'score': concept\_drift\_score},
        'semantic\_drift': {'detected': semantic\_drift\_detected, 'score': semantic\_drift\_score},
        'overall\_drift\_score': overall\_drift\_score,
        'should\_retrain': should\_retrain,
        'current\_data\_stats': {
            'sample\_count': len(data),
            'text\_features': detector.compute\_text\_features(data['text']),
            'label\_distribution': detector.compute\_label\_distribution(data['label'])
        }
    }
    with open('drift\_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    
    with open('drift\_visualizations.png', 'w') as f:
        f.write("Drift visualization placeholder")
    
    
    print("\\nDrift detection complete; report saved to drift\_report.json")
    if \_\_name\_\_ == "\_\_main\_\_":
main()
\`\`\`

\*\*\*

**\#\# Step 4: Create Drift-Aware Training Script**

**\#\#\# File: \`app/drift\_aware\_training.py\`**

\`\`\`python
import pandas as pd
from sklearn.feature\_extraction.text import CountVectorizer
from sklearn.model\_selection import train\_test\_split
from sklearn.naive\_bayes import MultinomialNB
from sklearn.metrics import accuracy\_score, classification\_report
import joblib
import json
import argparse
from datetime import datetime

def retrain\_with\_drift\_awareness(drift\_score, data\_drift, concept\_drift):
"""Retrain model with drift-aware strategies"""

    print(f"Starting drift-aware retraining...")
    print(f"Drift Score: {drift\_score}")
    print(f"Data Drift: {data\_drift}")
    print(f"Concept Drift: {concept\_drift}")
    
    # Load dataset
    data = pd.read\_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    
    print(f"Dataset loaded: {len(data)} samples")
    
    # Drift-aware preprocessing
    if data\_drift == 'true':
        print("Applying data drift mitigation strategies...")
        # Use more robust vectorization parameters
        max\_features = 8000  # Increased vocabulary
        min\_df = 2  # Minimum document frequency
        max\_df = 0.95  # Maximum document frequency
    else:
        max\_features = 5000
        min\_df = 1
        max\_df = 1.0
    
    # Concept drift handling
    if concept\_drift == 'true':
        print("Applying concept drift mitigation strategies...")
        # Use stratified sampling to ensure balanced representation
        stratify\_param = data['label']
        test\_size = 0.25  # Larger test set for better evaluation
    else:
        stratify\_param = None
        test\_size = 0.2
    
    # Train/test split with drift considerations
    X\_train, X\_test, y\_train, y\_test = train\_test\_split(
        data['text'], data['label'], 
        test\_size=test\_size, 
        random\_state=42,
        stratify=stratify\_param
    )
    
    # Vectorize with drift-aware parameters
    vectorizer = CountVectorizer(
        max\_features=max\_features,
        min\_df=min\_df,
        max\_df=max\_df,
        stop\_words='english',
        ngram\_range=(1, 2) if float(drift\_score) > 0.15 else (1, 1)  # Use bigrams for high drift
    )
    
    X\_train\_vec = vectorizer.fit\_transform(X\_train)
    X\_test\_vec = vectorizer.transform(X\_test)
    
    print(f"Feature matrix shape: {X\_train\_vec.shape}")
    
    # Model selection based on drift severity
    if float(drift\_score) > 0.2:
        print("High drift detected - using more robust model parameters")
        model = MultinomialNB(alpha=0.5)  # Higher smoothing for stability
    else:
        model = MultinomialNB(alpha=1.0)  # Standard smoothing
    
    # Train model
    print("Training drift-aware model...")
    model.fit(X\_train\_vec, y\_train)
    
    # Evaluate model
    y\_pred = model.predict(X\_test\_vec)
    accuracy = accuracy\_score(y\_test, y\_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\\nDetailed Classification Report:")
    print(classification\_report(y\_test, y\_pred))
    
    # Save model and vectorizer
    joblib.dump(model, 'spam\_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    
    # Save accuracy for GitHub Actions
    with open('model\_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")
    
    # Save training metadata with drift info
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'training\_samples': len(X\_train),
        'test\_samples': len(X\_test),
        'vocabulary\_size': len(vectorizer.vocabulary\_),
        'drift\_info': {
            'drift\_score': float(drift\_score),
            'data\_drift': data\_drift,
            'concept\_drift': concept\_drift,
            'mitigation\_applied': True
        },
        'model\_parameters': {
            'max\_features': max\_features,
            'min\_df': min\_df,
            'max\_df': max\_df,
            'ngram\_range': vectorizer.ngram\_range,
            'alpha': model.alpha
        }
    }
    
    with open('training\_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Drift-aware model training completed successfully!")
    
    # Update baseline statistics with new data
    from drift\_detection import DriftDetector
    detector = DriftDetector()
    new\_baseline = {
        'timestamp': datetime.now().isoformat(),
        'text\_features': detector.compute\_text\_features(data['text']),
        'label\_distribution': detector.compute\_label\_distribution(data['label']),
        'sample\_count': len(data),
        'model\_version': f"drift-aware-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }
    detector.save\_baseline\_stats(new\_baseline)
    print("Updated baseline statistics")
    def main():
parser = argparse.ArgumentParser(description='Train model with drift awareness')
parser.add\_argument('--drift-score', type=str, required=True, help='Drift score')
parser.add\_argument('--data-drift', type=str, default='false', help='Data drift detected')
parser.add\_argument('--concept-drift', type=str, default='false', help='Concept drift detected')
args = parser.parse\_args()

    retrain\_with\_drift\_awareness(args.drift\_score, args.data\_drift, args.concept\_drift)
    if \_\_name\_\_ == "\_\_main\_\_":
main()
\`\`\`

\*\*\*

**\#\# Step 5: Create Drift Report Generator**

**\#\#\# File: \`app/generate\_drift\_report.py\`**

\`\`\`python
import json
import sys
from datetime import datetime

def generate\_drift\_report(model\_version, drift\_score):
"""Generate a comprehensive drift-aware model report"""

    # Load training metadata
    try:
        with open('training\_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
    
    # Load drift report if available
    try:
        with open('drift\_report.json', 'r') as f:
            drift\_data = json.load(f)
    except FileNotFoundError:
        drift\_data = {}
    
    # Create comprehensive report
    report = {
        'model\_version': model\_version,
        'report\_generated': datetime.now().isoformat(),
        'drift\_analysis': {
            'trigger\_drift\_score': float(drift\_score),
            'drift\_details': drift\_data.get('data\_drift', {}),
            'concept\_drift\_details': drift\_data.get('concept\_drift', {}),
            'semantic\_drift\_details': drift\_data.get('semantic\_drift', {}),
            'mitigation\_applied': metadata.get('drift\_info', {}).get('mitigation\_applied', False)
        },
        'model\_performance': {
            'accuracy': metadata.get('accuracy', 'N/A'),
            'training\_samples': metadata.get('training\_samples', 'N/A'),
            'test\_samples': metadata.get('test\_samples', 'N/A'),
            'vocabulary\_size': metadata.get('vocabulary\_size', 'N/A')
        },
        'model\_parameters': metadata.get('model\_parameters', {}),
        'recommendations': generate\_recommendations(drift\_score, drift\_data),
        'status': 'completed',
        'files': [
            'spam\_model.joblib',
            'vectorizer.joblib',
            'baseline\_stats.json'
        ]
    }
    
    # Save report
    with open('drift\_model\_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Drift-aware model report generated for version: {model\_version}")
    print(f"Drift score that triggered retraining: {drift\_score}")
    print(f"Model accuracy: {metadata.get('accuracy', 'N/A')}")
    def generate\_recommendations(drift\_score, drift\_data):
"""Generate recommendations based on drift analysis"""
recommendations = []

    drift\_score\_float = float(drift\_score)
    
    if drift\_score\_float > 0.3:
        recommendations.append("High drift detected - consider more frequent monitoring")
        recommendations.append("Evaluate data collection process for potential issues")
    elif drift\_score\_float > 0.15:
        recommendations.append("Moderate drift detected - monitor closely")
        recommendations.append("Consider feature engineering improvements")
    else:
        recommendations.append("Low drift detected - normal monitoring schedule sufficient")
    
    # Specific recommendations based on drift type
    if drift\_data.get('data\_drift', {}).get('detected', False):
        recommendations.append("Data distribution changed - review feature preprocessing")
    
    if drift\_data.get('concept\_drift', {}).get('detected', False):
        recommendations.append("Label distribution changed - review labeling criteria")
    
    if drift\_data.get('semantic\_drift', {}).get('detected', False):
        recommendations.append("Semantic patterns changed - consider vocabulary updates")
    
    return recommendations
    if \_\_name\_\_ == "\_\_main\_\_":
if len(sys.argv) > 2:
model\_version = sys.argv[1]
drift\_score = sys.argv[2]
else:
model\_version = "unknown"
drift\_score = "0.0"

    generate\_drift\_report(model\_version, drift\_score)
    \`\`\`

\*\*\*

**\#\# Step 6: Create Model Validation Workflow**

**\#\#\# File: \`.github/workflows/validate-drift-model.yml\`**

\`\`\`yaml
name: Validate Drift-Retrained Model

on:
repository\_dispatch:
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
        name: drift-retrained-model-${{ github.event.client\_payload.model\_version }}
        github-token: ${{ secrets.GITHUB\_TOKEN }}
        run-id: ${{ github.event.client\_payload.run\_id }}
    
    
    - name: Run validation and predictions
      run: |
        python app/validate\_drift\_model.py \\
          --model-version "${{ github.event.client\_payload.model\_version }}" \\
          --drift-score "${{ github.event.client\_payload.drift\_score }}"
    
    
    - name: Upload validation results
      uses: actions/upload-artifact@v4
      with:
        name: validation-results-${{ github.event.client\_payload.model\_version }}
        path: |
          validation\_report.json
          drift\_predictions.txt
          model\_comparison.json
    \`\`\`

\*\*\*

**\#\# Step 7: Update Requirements**

**\#\#\# File: \`app/requirements.txt\`**

\`\`\` pandas scikit-learn joblib numpy scipy matplotlib seaborn \`\`\`

\*\*\*

**\#\# How to Use This Lab Exercise**

**\#\#\# 1. \*\*Manual Drift Detection\*\*:**

\`\`\`bash

# Go to Actions → "Drift Detection and Model Retraining" → "Run workflow"

# Set drift threshold (e.g., 0.1) and optionally force retraining

\`\`\`

**\#\#\# 2. \*\*Scheduled Monitoring\*\*:**

- The workflow runs daily at 6 AM UTC to check for drift
- Automatically triggers retraining if drift is detected

**\#\#\# 3. \*\*Drift Types Detected\*\*:**

- **\*\*Data Drift\*\***: Changes in input feature distributions
- **\*\*Concept Drift\*\***: Changes in label distributions
- **\*\*Semantic Drift\*\***: Changes in text patterns and vocabulary

**\#\#\# 4. \*\*Monitoring Results\*\*:**

- Check drift reports in workflow artifacts
- Review model comparison metrics
- Monitor retraining frequency and triggers

\*\*\*

based on above context, generate another exercise for student on another dataset

Perfect — the lab you’ve provided walked students through building a **drift-aware automated retraining system** using the **SMS Spam dataset**.

To deepen their understanding, let’s create **a new lab exercise** on **another dataset**. This time we’ll pick a **numeric dataset** instead of text, so they can practice adapting drift detection for structured/tabular features.

***

# 🚀 Lab Exercise: Automated Model Retraining with Drift Detection (UCI Wine Quality Dataset)


***

## 🎯 Objective

In this exercise, students will extend the drift-aware retraining pipeline to handle **structured dataset drift detection** with **numerical features**.

They will implement:

1. Daily monitoring for data drift and concept drift
2. Automatic retraining of a regression model
3. Baseline updating with feature distribution statistics
4. Creation of drift-aware validation and reports

***

## Dataset

We will use the **UCI Wine Quality Dataset**:

- URL (Red Wine Quality CSV):
`https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`

This dataset contains 11 physicochemical properties (numerical features) and a **quality score (0–10)** used as the target.

***

## Step 1: Drift Detection Workflow

### File: `.github/workflows/wine-drift-detection.yml`

👉 Similar to the SMS lab, but with **winequality-red.csv** as the data source.
👉 Students should compute drift on numerical features instead of text features.

```yaml
name: Wine Drift Detection and Model Retraining

on:
  workflow_dispatch:
    inputs:
      drift_threshold:
        description: 'Drift detection threshold (0.0-1.0)'
        required: true
        default: '0.15'
        type: string
      force_retrain:
        description: 'Force retraining regardless of drift'
        required: false
        default: false
        type: boolean
  schedule:
    - cron: '0 6 * * *' # Run daily at 6 AM UTC

jobs:
  detect_drift:
    runs-on: ubuntu-latest
    outputs:
      data_drift_detected: ${{ steps.drift_check.outputs.data_drift_detected }}
      concept_drift_detected: ${{ steps.drift_check.outputs.concept_drift_detected }}
      drift_score: ${{ steps.drift_check.outputs.drift_score }}
      should_retrain: ${{ steps.drift_check.outputs.should_retrain }}

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: 3.10
    - run: |
        pip install -r app/requirements.txt
    - run: |
        curl -L -o winequality.csv \
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    - name: Run drift detection
      id: drift_check
      run: |
        CMD="python app/wine_drift_detection.py --threshold ${{ github.event.inputs.drift_threshold }}"
        if [ "${{ github.event.inputs.force_retrain }}" = "true" ]; then
          CMD="$CMD --force-retrain"
        fi
        $CMD

    - uses: actions/upload-artifact@v4
      with:
        name: wine-drift-report-${{ github.run_id }}
        path: |
          wine_drift_report.json
          wine_baseline_stats.json

  trigger_retraining:
    needs: detect_drift
    if: needs.detect_drift.outputs.should_retrain == 'true'
    uses: ./.github/workflows/wine-retrain-with-drift.yml
    with:
      drift_score: ${{ needs.detect_drift.outputs.drift_score }}
      data_drift: ${{ needs.detect_drift.outputs.data_drift_detected }}
      concept_drift: ${{ needs.detect_drift.outputs.concept_drift_detected }}
```


***

## Step 2: Wine-Specific Drift Detection Script

### File: `app/wine_drift_detection.py`

Key changes from SMS example:

- Compute drift using **Kolmogorov–Smirnov test** for numeric distributions
- Track **mean and std** of each numeric feature
- Concept drift detection → monitor changes in label (`quality`) distribution

```python
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import argparse, json, os
from datetime import datetime

class WineDriftDetector:
    def __init__(self, threshold=0.15):
        self.threshold = threshold
        self.baseline_file = "wine_baseline_stats.json"

    def load_baseline(self):
        return json.load(open(self.baseline_file)) if os.path.exists(self.baseline_file) else None

    def save_baseline(self, stats):
        with open(self.baseline_file, "w") as f:
            json.dump(stats, f, indent=2)

    def compute_feature_stats(self, df):
        return {col: {"mean": df[col].mean(), "std": df[col].std()} for col in df.columns[:-1]}

    def label_distribution(self, labels):
        counts = labels.value_counts(normalize=True).to_dict()
        return counts

    def detect_data_drift(self, df, baseline):
        drift_scores = {}
        df_labels = df["quality"]
        total_score = 0
        for col in df.columns[:-1]:
            stat, pval = ks_2samp(df[col], np.random.normal(
                baseline["features"][col]["mean"],
                baseline["features"][col]["std"],
                len(df[col])
            ))
            drift_scores[col] = 1 - pval
            total_score += (1 - pval)
        avg_score = total_score / len(drift_scores)
        return avg_score > self.threshold, avg_score, drift_scores

    def detect_concept_drift(self, df, baseline):
        new_dist = self.label_distribution(df["quality"])
        old_dist = baseline["label_distribution"]
        labels = set(new_dist) | set(old_dist)
        score = sum(abs(new_dist.get(l,0)-old_dist.get(l,0)) for l in labels)
        return score > self.threshold, score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv("winequality.csv", sep=";")

    detector = WineDriftDetector(args.threshold)
    baseline = detector.load_baseline()

    if baseline is None:
        baseline = {
            "timestamp": datetime.now().isoformat(),
            "features": detector.compute_feature_stats(df),
            "label_distribution": detector.label_distribution(df["quality"]),
            "sample_count": len(df)
        }
        detector.save_baseline(baseline)
        should_retrain = args.force_retrain
        data_drift, concept_drift, drift_score = False, False, 0.0
    else:
        data_drift, data_score, data_details = detector.detect_data_drift(df, baseline)
        concept_drift, concept_score = detector.detect_concept_drift(df, baseline)
        drift_score = (0.6*data_score + 0.4*concept_score)
        should_retrain = data_drift or concept_drift or args.force_retrain

    # Write action outputs
    out = os.environ["GITHUB_OUTPUT"]
    with open(out,"a") as f:
        f.write(f"data_drift_detected={str(data_drift).lower()}\n")
        f.write(f"concept_drift_detected={str(concept_drift).lower()}\n")
        f.write(f"drift_score={drift_score:.4f}\n")
        f.write(f"should_retrain={str(should_retrain).lower()}\n")

    json.dump({
        "timestamp": datetime.now().isoformat(),
        "data_drift": data_drift, "concept_drift": concept_drift,
        "drift_score": drift_score, "should_retrain": should_retrain
    }, open("wine_drift_report.json","w"), indent=2)

if __name__ == "__main__":
    main()
```


***

## Step 3: Retraining Script

### File: `app/wine_drift_training.py`

- Train **RandomForestRegressor**
- Adjust tree depth if drift is high

***

## Step 4: Updated Report + Validation

- Adapt the retraining and validation workflow to output **wine model accuracy (RMSE, MAE, R²)**
- Update `.github/workflows/wine-retrain-with-drift.yml` and validation pipeline analogously

***

## ✅ Student Tasks

1. Adapt the **drift detection script** for wine quality dataset using **numeric features**
2. Implement **RandomForestRegressor retraining** in `wine_drift_training.py`
3. Generate and compare **baseline vs current feature statistics**
4. Validate drift-aware retrained models via `wine-validation.yml`

***
