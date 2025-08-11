# Problem Statement: Sentiment Classification on IMDb Movie Reviews Using BERT

## Objective

Build and fine-tune a Transformer-based model (BERT) to classify movie reviews from the IMDb dataset into positive or negative sentiment. This task involves data cleaning, tokenization, model training, evaluation, and analysis, following a similar pipeline demonstrated in the transformer tweet sentiment example.

## Dataset

**IMDb Movie Reviews**

- Publicly accessible via the Hugging Face Datasets library (no manual download or sign-in required).
- Loading code snippet:

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
train = dataset["train"]
test = dataset["test"]
```

- Dataset size: 25,000 training samples and 25,000 testing samples.
- Structure: Each example contains a `text` field (the movie review) and a `label` field (0 = negative, 1 = positive).


## Learning Objectives

- Clean and preprocess natural language movie reviews (remove HTML tags, special characters, unwanted whitespace).
- Tokenize and encode text using `BertTokenizer`.
- Fine-tune `BertForSequenceClassification` for binary sentiment classification.
- Evaluate model with classification metrics (precision, recall, F1-score).
- Analyze model predictions, including inspection of correctly and incorrectly classified samples.


## Tasks

1. **Data Loading \& Exploration**
    - Load the IMDb dataset directly using Hugging Face’s `load_dataset("imdb")` function.
    - Analyze dataset distribution and sample texts to understand the data.
2. **Data Cleaning**
    - Clean the review texts to remove noise such as HTML tags and punctuation.
    - Prepare the cleaned text for tokenization.
3. **Dataset Preparation**
    - Implement a PyTorch Dataset class similar to `TweetDataset`, which performs tokenization, padding, and truncation using `BertTokenizer`.
    - Ensure token sequences have a max length (e.g., 128) for efficient batching.
4. **Model Setup and Training**
    - Load the pretrained BERT base uncased model configured for sequence classification with two output labels.
    - Define training parameters such as batch size, epochs, and logging setup.
    - Use the Hugging Face `Trainer` API to train and validate the model on the IMDb data.
5. **Evaluation and Reporting**
    - Generate a detailed classification report with precision, recall, and F1-score.
    - Create a DataFrame comparing review texts, actual labels, and predicted labels for sample inspection.

## Deliverables

- Python notebook or script containing fully documented code for the entire pipeline.
- Classification report and insights into model performance and errors.
- Examples of correct and incorrect predictions with analysis.


## Getting Started Example

```python
from datasets import load_dataset

# Load IMDb dataset
dataset = load_dataset("imdb")
train = dataset["train"]
test = dataset["test"]

print(f"Number of training samples: {len(train)}")
print(f"Number of test samples: {len(test)}")

# Sample review and label
print("Sample text:", train[0]["text"][:200])
print("Sample label:", train[0]["label"])
```


***

This problem statement ensures you use a reliable, easy-to-access dataset with no external sign-in or manual downloads, perfectly fitting into a Transformer fine-tuning workflow.
