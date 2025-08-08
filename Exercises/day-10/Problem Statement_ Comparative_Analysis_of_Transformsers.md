# Problem Statement: Comparative Analysis of Transformer Components on AG News Headlines

## Objective

Implement and compare the three core transformer components—**Self-Attention**, **Multi-Head Attention**, and **Feedforward Neural Networks**—using real-world news headlines from the AG News dataset. Visualize and analyze how each component processes and transforms text across four news categories.

## Dataset

**AG News**

- Loaded via the Hugging Face `datasets` library (no sign-in required).
- 120,000 training headlines, 7,600 test headlines.
- Four balanced categories: World (0), Sports (1), Business (2), Sci/Tech (3).

Load with:

```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
train = dataset["train"]
test  = dataset["test"]
```


## Learning Objectives

- Understand and implement self-attention, multi-head attention, and feedforward layers.
- Visualize attention weight patterns on real news headlines.
- Compare single-head vs. multi-head attention behaviors across categories.
- Analyze how feedforward networks refine token representations after attention.


## Tasks

### Task 1: Data Preparation

1. Load AG News via `load_dataset("ag_news")`.
2. Sample 10–15 headlines from each category (“World,” “Sports,” “Business,” “Sci/Tech”).
3. Tokenize each headline and convert tokens to embeddings (random vectors or pretrained embeddings).

### Task 2: Self-Attention

- Adapt the provided `SimpleAttention` class to compute attention weights over headline embeddings.
- For each category’s sample headlines, produce and plot the self-attention weight matrix as a heatmap.
- Interpret which tokens receive highest attention within each headline.


### Task 3: Multi-Head Attention

- Use the provided `MultiHeadAttention` class with four heads on the same headline embeddings.
- For a representative headline from each category, visualize each head’s attention weight matrix.
- Compare patterns across heads and categories, noting heads that focus on entities vs. actions vs. other tokens.


### Task 4: Feedforward Neural Network

- Implement a positionwise feedforward network (two linear layers with ReLU activation).
- Apply it to the outputs of multi-head attention.
- Visualize or inspect how token embeddings change before vs. after the feedforward step for selected headlines.


### Task 5: Comparative Analysis Report

- Compile self-attention, multi-head attention, and feedforward visualizations side by side.
- Describe differences in attention patterns across categories (e.g., sports headlines focusing on player names).
- Explain how multi-head attention captures diverse linguistic features and how feedforward layers refine representations.
- Reflect on implications for transformer-based NLP tasks such as classification or summarization.


## Deliverables

1. **Code**: Well-documented implementations of Self-Attention, Multi-Head Attention, and Feedforward layers applied to AG News headlines.
2. **Visualizations**: Heatmaps of attention weights and examples of embedding transformations.
3. **Report**: Comparative analysis detailing observations for each transformer component across the four news categories.

## Getting Started

```bash
pip install datasets torch matplotlib seaborn
```

```python
from datasets import load_dataset
dataset = load_dataset("ag_news")
train = dataset["train"]

# Example: fetch 5 World headlines
world_headlines = [ex["text"] for ex in train if ex["label"] == 0][:5]
for h in world_headlines:
    print("•", h)
```

Use these samples to implement and compare attention and feedforward mechanisms as outlined above.

