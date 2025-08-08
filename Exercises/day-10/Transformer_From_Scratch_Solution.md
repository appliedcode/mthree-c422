<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# could you generate the solution?

Certainly! Below is a complete solution for **sentiment classification on the IMDb Movie Reviews dataset using BERT**, based on the problem statement and dataset from Hugging Face `datasets`. This code covers:

- Loading and exploring the dataset
- Cleaning the text
- Tokenization and dataset preparation
- Fine-tuning `BertForSequenceClassification`
- Evaluating the model with classification report and confusion matrix
- Inspecting some predictions

***

```python
# Install required packages (uncomment if running from scratch in Colab/other)
# !pip install transformers datasets sklearn

import re
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding,
    logging
)
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Check for GPU availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 1. Load IMDb dataset from Hugging Face
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
print(f"Training set size: {len(train_df)}")
print(f"Test set size    : {len(test_df)}")

# Optional: check label distribution
print("Label distribution (train):\n", train_df["label"].value_counts())
print("Label distribution (test):\n", test_df["label"].value_counts())

# 2. Text cleaning function
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric chars except spaces
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Apply cleaning
train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

# 3. Tokenization and Dataset class
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from torch.utils.data import Dataset

class IMDbDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(
            df["cleaned_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = IMDbDataset(train_df)
test_dataset = IMDbDataset(test_df)

# 4. Load BERT sequence classification model
num_labels = 2
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device);

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # disable external logging by default
    seed=42,
    save_total_limit=1,
)

# 6. Data collator (helps dynamic padding in batches)
data_collator = DataCollatorWithPadding(tokenizer)

# 7. Metrics function for evaluation
import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    return {"accuracy": acc["accuracy"]}

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9. Start training
print("Starting training...")
trainer.train()

# 10. Evaluation on test set
print("\nEvaluating model on test set...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 11. Predictions & classification report
print("\nGenerating predictions on test set...")
pred_output = trainer.predict(test_dataset)
preds = np.argmax(pred_output.predictions, axis=1)
labels = pred_output.label_ids

print("\nClassification Report:\n")
print(classification_report(labels, preds, target_names=["negative", "positive"]))

# 12. Confusion Matrix plot
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 13. Inspect some predictions
compare_df = pd.DataFrame({
    "text": test_df["cleaned_text"],
    "actual_label": labels,
    "predicted_label": preds
})

print("\nSample predictions:\n")
print(compare_df.sample(10).to_string(index=False))
```


***

### Explanation and Next Steps:

- **Training:** This fine-tunes BERT for 2 epochs. Feel free to increase `num_train_epochs` for better accuracy if you have more compute time.
- **Max sequence length:** 128 tokens balances performance and speed.
- **Evaluation:** Includes metrics and confusion matrix.
- **Prediction inspection:** Helps to analyze cases where model succeeds or fails.

***

If you want, I can help you extend this with:

- Hyperparameter tuning
- Saving/loading model checkpoints
- Using evaluation metrics like ROC-AUC
- Handling class imbalance or data augmentation

Just let me know!

