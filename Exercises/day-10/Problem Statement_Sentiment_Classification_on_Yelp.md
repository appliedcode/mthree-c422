<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Problem Statement: Sentiment Classification on Yelp Polarity Reviews Using BERT

## Objective

Fine-tune a Transformer-based model (BERT) to classify Yelp restaurant reviews into positive or negative sentiment. Build data preprocessing pipelines, tokenize text, train and evaluate the model, and analyze classification results.

## Dataset

**Yelp Polarity Reviews**

- **Access:** Publicly available via Hugging Face Datasets (no sign-in required)
- **Loading Code:**

```python
from datasets import load_dataset
dataset = load_dataset("yelp_polarity")
train_df = dataset["train"].to_pandas()
test_df  = dataset["test"].to_pandas()
```

- **Structure:**
    - `text`: review text
    - `label`: sentiment (0 = negative, 1 = positive)
- **Size:** 560,000 training reviews, 38,000 test reviews


## Learning Objectives

- Clean and preprocess customer review text (remove noise, URLs, punctuation)
- Tokenize and encode text for BERT with `BertTokenizer`
- Fine-tune `BertForSequenceClassification` on Yelp data
- Evaluate performance using precision, recall, F1-score, and confusion matrix
- Perform error analysis on misclassified examples


## Tasks

1. **Data Loading \& Exploration**
    - Load the Yelp Polarity dataset using `load_dataset("yelp_polarity")`
    - Examine class balance and review length distribution
2. **Data Cleaning**
    - Remove URLs, non-alphanumeric characters, and extra whitespace
    - Add a `cleaned_text` column to both DataFrames
3. **Dataset Preparation**
    - Implement a PyTorch `Dataset` class that tokenizes and encodes reviews with `BertTokenizer`
    - Apply padding and truncation to a maximum of 128 tokens
4. **Model Setup**
    - Initialize `BertForSequenceClassification` with `bert-base-uncased` and `num_labels=2`
    - Configure `TrainingArguments` for 2 epochs, batch size 16, and appropriate logging
5. **Training \& Evaluation**
    - Use Hugging Face’s `Trainer` API to train on the train split and evaluate on the test split
    - Generate a classification report and confusion matrix
6. **Error Analysis**
    - Create a DataFrame comparing `cleaned_text`, actual labels, and predicted labels
    - Display several misclassified reviews and propose strategies (e.g., handling irony, data augmentation)

## Deliverables

- A Python notebook with all code implementations
- Classification report and confusion matrix plot
- Written analysis of error patterns and improvement suggestions

