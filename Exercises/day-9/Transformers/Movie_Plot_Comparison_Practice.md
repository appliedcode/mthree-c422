# Problem Statement: Comparative Analysis of Transformer Components Using Movie Plot Summaries

### Objective

Implement and compare the three core transformer components—**Self-Attention**, **Multi-Head Attention**, and **Feedforward Neural Networks**—using real-world textual data (movie plot summaries). You will visualize and analyze how each component processes and transforms token embeddings on natural language text.

### Dataset Options

1. **CMU Movie Summary Corpus**
– Description: Plot summaries for more than 42,000 movies, each linked to Wikipedia.
– Download: https://www.cs.cmu.edu/~ark/personas/ (click “Download dataset”)
2. **IMDb Movie Plot Summaries (via Kaggle)**
– Description: 5,043 user-submitted plot summaries from IMDb, labeled with genres.
– Download: https://www.kaggle.com/PromptCloudHQ/imdb-data

(You must sign in to Kaggle to download.)

Either dataset provides ample text length and variety for attention visualization exercises.

### Learning Objectives

- Build and visualize **self-attention** weights on realistic text sequences.
- Extend to **multi-head attention** and observe diverse attention patterns.
- Apply **feedforward layers** to refine attention outputs.
- Compare and explain how these components differ in processing natural language.


### Tasks

**Task 1: Data Preparation**

- Download one of the provided datasets.
- Extract a small sample subset (e.g., 10–20 plot summaries).
- Clean and tokenize text; convert tokens to embeddings (random or pretrained).

**Task 2: Self-Attention**

- Implement a self-attention mechanism on token embeddings.
- Visualize the attention weight matrix as a heatmap for each summary.
- Interpret which words each token attends to and why.

**Task 3: Multi-Head Attention**

- Implement multi-head attention with at least 4 heads.
- Visualize each head’s attention patterns separately.
- Compare multi-head attention to single-head self-attention.

**Task 4: Feedforward Neural Networks**

- Apply a position-wise feedforward network to the multi-head attention outputs.
- Compare token embeddings before and after feedforward processing.
- Explain how feedforward layers refine the representations.

**Task 5: Comparative Analysis Report**

- Compile your visualizations and observations.
- Describe the unique roles and differences of self-attention, multi-head attention, and feedforward layers.
- Reflect on how these components enable transformer models to understand and generate language.


### Deliverables

- Well-documented code for each component.
- Heatmap visualizations of attention weights.
- Comparative analysis write-up explaining your findings.
- (Optional) Suggestions for extensions: positional encodings, full transformer layer stacking, or using pretrained embeddings.

With the direct dataset links above, you can begin downloading and experimenting immediately. Let me know if you need starter code or further guidance!

