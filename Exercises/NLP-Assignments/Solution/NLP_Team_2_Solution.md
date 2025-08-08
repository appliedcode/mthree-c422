## Solution: Multi-Label Text Classification using Transformers on the Reuters-21578 Dataset

### 1. Setup Environment and Install Libraries

```python
!pip install transformers datasets scikit-learn nltk
```


### 2. Load and Preprocess Dataset

You can use `nltk.corpus.reuters` along with `datasets` library to fetch and prepare the Reuters-21578 dataset. Here, we focus on handling multi-label targets and selecting a manageable subset of the most frequent categories.

```python
import nltk
nltk.download("reuters")
nltk.download("punkt")

from nltk.corpus import reuters
import pandas as pd
from collections import Counter

# Extract file IDs and their categories
fileids = reuters.fileids()

# Gather documents and their multi-labels
texts = [reuters.raw(fid) for fid in fileids]
labels = [reuters.categories(fid) for fid in fileids]

# Flatten labels and find most common categories for manageable subset
all_labels = [label for sublist in labels for label in sublist]
top_labels = [label for label, count in Counter(all_labels).most_common(20)]  # top 20 categories

# Filter documents for those having at least one top category label
filtered_texts = []
filtered_labels = []
for text, lbls in zip(texts, labels):
    filtered_lbls = [l for l in lbls if l in top_labels]
    if filtered_lbls:
        filtered_texts.append(text)
        filtered_labels.append(filtered_lbls)

# Create a binary multi-label format dataframe
import numpy as np
mlb_labels = []
for lbls in filtered_labels:
    mlb = [1 if cat in lbls else 0 for cat in top_labels]
    mlb_labels.append(mlb)

df = pd.DataFrame({
    'text': filtered_texts,
    'labels': mlb_labels
})

print(f"Total samples after filtering: {df.shape[0]}")
print(f"Top categories considered: {top_labels}")
```


### 3. Train-Test Split

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```


### 4. Tokenization

Use the BERT tokenizer with padding and truncation.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

train_encodings = tokenize_function(train_df['text'].tolist())
test_encodings = tokenize_function(test_df['text'].tolist())
```


### 5. Create PyTorch Dataset

```python
import torch

class ReutersDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.FloatTensor(self.labels[idx])
        return item

train_dataset = ReutersDataset(train_encodings, train_df['labels'].tolist())
test_dataset = ReutersDataset(test_encodings, test_df['labels'].tolist())
```


### 6. Load Pretrained Model for Multi-Label Classification

Adapt output layer for multi-label classification using sigmoid activation with 20 categories.

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(top_labels),
    problem_type="multi_label_classification"
)
```


### 7. Define Evaluation Metrics for Multi-Label

We use thresholding on sigmoid outputs to get predictions and compute micro and macro precision, recall, and F1-score.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    # Note: accuracy in multi-label is more complex; here we report subset accuracy for exact match
    subset_acc = accuracy_score(labels, preds)

    return {
        'accuracy': subset_acc,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
```


### 8. Training Setup with Trainer API

Set batch size, epochs, and evaluation strategy.

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
```


### 9. Train the Model

```python
trainer.train()
```


### 10. Evaluate the Model

```python
eval_results = trainer.evaluate()
print(eval_results)
```


### 11. Optional: Analyze Confusion Matrix per Category

For multi-label, confusion matrix per label can be computed separately to analyze true positives, false positives, etc.

```python
import matplotlib.pyplot as plt
import seaborn as sns

predictions_output = trainer.predict(test_dataset)
logits, labels = predictions_output.predictions, predictions_output.label_ids
probs = sigmoid(logits)
preds = (probs >= 0.5).astype(int)

for i, category in enumerate(top_labels):
    cm = confusion_matrix(labels[:, i], preds[:, i])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix for category: {category}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
```


### 12. Optional Extension: Audio Feature Extraction Insight

Brief demonstration of MFCC feature extraction useful for audio emotion or sentiment recognition.

```python
import librosa
import numpy as np

print("Example MFCC feature extraction:")
audio_path = librosa.ex('trumpet')
y, sr = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"MFCCs shape: {mfccs.shape}")
```


### Summary

This solution equips you to:

- Load and prepare Reuters-21578 data for multi-label problem.
- Select top N labels and encode multi-label targets in binary format.
- Tokenize text data with BERT tokenizer.
- Fine-tune a pre-trained BERT model adapted for multi-label classification.
- Evaluate model using appropriate multi-label metrics.
- Visualize confusion matrices per label.
- Gain insight into related audio-based sentiment/emotion recognition using MFCC extraction.

