## Solution: Multi-Class Sentiment Classification using Transformers on the Amazon Reviews Dataset

### 1. Setup Environment and Install Libraries

```python
# Install required libraries
!pip install transformers datasets scikit-learn
```


### 2. Load Dataset

We use the Huggingface `datasets` library to load a suitable Amazon reviews dataset, such as `"amazon_review_full"` which has reviews with ratings 1 to 5.

```python
from datasets import load_dataset

# Load full Amazon review dataset with ratings 1 to 5
dataset = load_dataset("amazon_review_full")

print(dataset)
```


### 3. Dataset Inspection and Cleaning (optional)

Look at some examples and optionally clean text (remove special characters etc.) if desired.

```python
# Example review and label
print(dataset['train'][0]['review'])
print("Label (rating):", dataset['train'][0]['label'])
```

If needed, add a function to clean reviews (this is optional and depends on your use case).

```python
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars except words and spaces
    return text

# Apply cleaning during preprocessing if required
```


### 4. Tokenization using BERT Tokenizer

Use `BertTokenizer` to tokenize texts with padding and truncation.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    # If cleaning needed, apply here: examples['review'] = [clean_text(r) for r in examples['review']]
    return tokenizer(examples['review'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```


### 5. Prepare Datasets for PyTorch and Remove Text Column

```python
tokenized_datasets = tokenized_datasets.remove_columns(['review'])
tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']
```


### 6. Load Pretrained BERT Model with Classification Head

Configure output layer for 5 classes (ratings 1 to 5).

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
```


### 7. Define Metrics for Evaluation

Use `sklearn.metrics` to compute accuracy, precision, recall, F1-score, and also generate a confusion matrix after evaluation.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
```


### 8. Define Training Arguments and Trainer

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
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


### 11. Confusion Matrix Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions on test set
predictions_output = trainer.predict(test_dataset)
y_pred = np.argmax(predictions_output.predictions, axis=1)
y_true = predictions_output.label_ids

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix: Amazon Reviews Sentiment Classification")
plt.show()
```


### 12. Optional Extension: Audio Feature Extraction Insight

Briefly demonstrate audio feature extraction using MFCCs which is relevant for audio-based sentiment/emotion recognition.

```python
import librosa
import numpy as np

print("Demonstrating MFCC feature extraction from example audio...")
audio_path = librosa.ex('trumpet')
y, sr = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"MFCC feature shape: {mfccs.shape}")
```


### Summary

This solution guides you through:

- Loading and preprocessing the multi-class Amazon sentiment reviews dataset.
- Tokenizing text using BERT tokenizer with padding/truncation.
- Fine-tuning the pretrained BERT model for 5-class sentiment classification.
- Evaluating with accuracy, precision, recall, F1-score, and confusion matrix.
- Extending understanding by linking text and audio sentiment recognition via MFCC feature extraction.

