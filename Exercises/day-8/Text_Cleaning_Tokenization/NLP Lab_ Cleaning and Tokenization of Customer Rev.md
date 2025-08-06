# NLP Lab: Cleaning and Tokenization of Customer Reviews

## Objective

Build a reusable text-preprocessing pipeline to clean, normalize, and tokenize a set of “messy” product reviews. You will compare multiple tokenization methods and visualize token frequencies.

## Dataset

Create a list of 10 messy customer review strings, for example:

```python
reviews = [
  "Absolutely LOVED this product!!! Will buy again 😊 Visit http://shop.example.com",
  "Worst purchase ever... arrived broken, no response from support @helpdesk",
  "Ok quality; does the job. 5/5 stars! #satisfied",
  "Email me at user@example.org for details about bulk order!!!",
  "Super overpriced!! Paid $299 but performance is meh...",
  "<div>Great build quality</div><p>But shipping was slow</p>",
  "Contact: +44 20 7946 0958 or (020)79460958",
  "MixedCASE and random123numbers and symbols %^&*",
  "Line1\nLine2\tTabbed text\r\nEnd of review",
  "Contractions—can't, won't, shouldn't—are common here."
]
```


## Tasks

1. **Implement Cleaning Functions**
    - Remove URLs, email addresses, HTML tags, phone numbers, social media mentions/hashtags, and non-alphanumeric symbols (except whitespace and apostrophes).
    - Normalize whitespace and convert text to lowercase.
2. **Compare Tokenization Methods**
    - Apply at least three methods to one cleaned review:
        - Simple `split()`
        - NLTK `word_tokenize`
        - spaCy tokenization
        - *(Bonus)* Regex-based tokenization
3. **Aggregate and Visualize**
    - Tokenize all cleaned reviews using your chosen method (e.g., regex).
    - Compute the frequency distribution of tokens across the dataset.
    - Plot the top 10 most frequent tokens with `matplotlib` or `seaborn`.
4. **Analysis**
    - Briefly comment on the differences observed between tokenization methods.
    - Discuss any surprising tokens in the top 10 and suggest why they appear so often.

## Deliverables

- A **Colab notebook** implementing the cleaning pipeline as reusable functions.
- Demonstrations of each tokenization method on one sample review.
- A bar chart of the top 10 tokens after processing all reviews.
- Written comments summarizing your observations and analysis.

