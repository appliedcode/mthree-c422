<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# News Headlines NLP Lab: Bag-of-Words and Document Similarity

## Objective

Analyze a collection of news headlines by building a Bag-of-Words representation to extract features, explore word frequency, and compute document similarity.

## Dataset

Use the following list of news headlines:

```python
headlines = [
    "AI outperforms doctors in diagnosing rare diseases",
    "Stock markets hit new record highs amid global optimism",
    "New vaccine shows promise in early trials",
    "Climate change impacts agriculture across multiple continents",
    "Scientists develop biodegradable plastic from seaweed",
    "Sports teams adapt strategies with big data analytics",
    "Electric vehicles set new sales record worldwide",
    "Breakthrough in quantum computing boosts encryption security"
]
```


## Tasks

1. **Preprocessing**
    - Write a function to lowercase all text, remove punctuation, and normalize whitespace in each headline.
2. **Bag-of-Words Analysis**
    - Use scikit-learn’s `CountVectorizer` with stop word removal and vocabulary limited to 50 words.
    - Fit and transform the preprocessed headlines into a Bag-of-Words matrix.
    - Display the vocabulary, shape, and sparsity of the matrix.
3. **Word Frequency and Visualization**
    - Compute total word frequency across all headlines.
    - Plot the top 10 most frequent words using `matplotlib` or `seaborn`.
4. **Document Similarity**
    - Calculate cosine similarity between headline vectors.
    - Display the similarity matrix in tabular form.
    - Identify the two most similar headlines and explain their similarity based on shared vocabulary.

## Deliverables

- A notebook implementing the preprocessing function and Bag-of-Words construction.
- Printed output showing vocabulary and matrix characteristics.
- A bar chart of the top 10 words by frequency.
- A similarity matrix with highlighted most similar headline pairs.
- A short commentary explaining the results.

This exercise provides hands-on experience with core NLP techniques including text cleaning, feature extraction via Bag-of-Words, and comparing documents using cosine similarity on vectorized features.

