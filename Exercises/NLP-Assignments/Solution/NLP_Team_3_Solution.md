## Solution: Binary Sentiment Classification using Transformers on the Yelp Reviews Polarity Dataset

### 1. Setup Environment and Install Required Libraries

```python
!pip install transformers datasets scikit-learn
```


### 2. Load the Dataset

We use Huggingface's `datasets` library to load the `yelp_polarity` dataset, which is balanced with two labels: positive (1) and negative (0).

```python
from datasets import load_dataset

dataset = load_dataset("yelp_polarity")
print(dataset)
```

Expected splits:

- Train: 560,000 samples
- Test: 38,000 samples


### 3. Dataset Preview and Optional Cleaning

You can preview a few samples:

```python
print(dataset['train'][^0])
```

Optionally, you can define a simple text cleaning function (e.g., lowercasing, removing extra spaces, special characters), but this is not mandatory as BERT tokenizer handles raw text well.

### 4. Tokenization

Use the pretrained BERT tokenizer to convert raw text to input IDs with padding and truncation.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_fn, batched=True)
```


### 5. Prepare Datasets for PyTorch

Remove the raw text column and set format for PyTorch tensors:

```python
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format('torch')
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']
```


### 6. Load Pretrained BERT Model for Binary Classification

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```


### 7. Define Evaluation Metrics

Use `sklearn.metrics` to compute accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```


### 8. Setup Trainer and Training Arguments

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
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


### 11. Confusion Matrix Visualization (Optional)

Visualize the confusion matrix to better understand classification errors:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Yelp Polarity Sentiment Classification')
plt.show()
```


### 12. Optional Extension: Audio-Based Sentiment Detection Insight

Briefly demonstrate MFCC (Mel Frequency Cepstral Coefficients) feature extraction used in audio sentiment recognition, for multimodal understanding:

```python
import librosa
import numpy as np

print("Demo: Extracting MFCC features from example audio...")
audio_path = librosa.ex('trumpet')
y, sr = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(f"MFCC feature shape: {mfccs.shape}")
```

This solution covers all the key aspects of the assignment:

- Loading and tokenizing a large-scale binary sentiment dataset
- Fine-tuning a pretrained BERT model for binary classification
- Evaluating model performance with standard metrics
- Optionally visualizing confusion matrix and discussing audio sentiment approaches

If you want, I can prepare a ready-to-run Google Colab notebook with this entire solution. Just let me know!

<div style="text-align: center">⁂</div>

[^1]: https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews

[^2]: https://hex.tech/templates/sentiment-analysis/sentiment-analysis/

[^3]: https://github.com/wutianqidx/Yelp-Review-Sentiment-Analysis

[^4]: https://metatext.io/datasets/yelp-polarity-reviews

[^5]: https://www.kaggle.com/datasets/irustandi/yelp-review-polarity

[^6]: https://www.kaggle.com/code/anasofiauzsoy/yelp-review-sentiment-analysis-tensorflow-tfds

