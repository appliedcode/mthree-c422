# Transformer-Based Encoder–Decoder Practice Exercises with Solutions

The following exercises build on your summarization program and cover key concepts in Transformer encoder–decoder models. Each exercise includes a concise solution outline.

## Exercise 1: Parameter Effects on Summarization

**Task:**
Run the summarizer on the same input text with these settings and compare the outputs:

1. `max_length=30, min_length=10`
2. `max_length=60, min_length=20`
3. `do_sample=True, temperature=0.7, max_length=45, min_length=20`

**Solution Outline:**

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "…(your paragraph)…"

configs = [
    {"max_length":30, "min_length":10, "do_sample":False},
    {"max_length":60, "min_length":20, "do_sample":False},
    {"max_length":45, "min_length":20, "do_sample":True, "temperature":0.7}
]

for cfg in configs:
    summary = summarizer(text, **cfg)[0]["summary_text"]
    print(cfg, "\n", summary, "\n")
```

- **Expected Observations:**
    - Shorter `max_length` yields very concise summaries, potentially missing details.
    - Longer `max_length` captures more context but risks verbosity.
    - Sampling (`do_sample=True`) introduces variation; temperature controls randomness.


## Exercise 2: Summarization on Different Text Styles

**Task:**
Summarize three different text types—news article, recipe, and email—using the same BART pipeline settings (`max_length=45,min_length=20,do_sample=False`). Compare how well the model adapts.

```python
texts = {
    "news": "Scientists at MIT have developed a new battery technology…",
    "recipe": "To make cookies: preheat oven to 375°F. Mix flour, sugar, …",
    "email": "Hi team, quarterly sales exceeded targets by 15%…"
}

for label, txt in texts.items():
    s = summarizer(txt, max_length=45, min_length=20, do_sample=False)[0]["summary_text"]
    print(label, "→", s)
```

- **Solution:**
    - **News:** Clear, factual summary.
    - **Recipe:** May drop procedural steps; summarizer struggles with instructions.
    - **Email:** Captures key points (dates, metrics) but may omit casual phrasing.


## Exercise 3: Model Comparison (BART vs. T5)

**Task:**
Compare summaries from:

- `facebook/bart-large-cnn`
- `google/t5-small` (prefix input with `"summarize: "`)

```python
from transformers import pipeline

bart = pipeline("summarization", model="facebook/bart-large-cnn")
t5  = pipeline("summarization", model="google/t5-small")

for name, mdl in [("BART", bart), ("T5", t5)]:
    inp = "summarize: " + text if name=="T5" else text
    s = mdl(inp, max_length=45, min_length=20, do_sample=False)[0]["summary_text"]
    print(f"{name} → {s}\n")
```

- **Solution:**
    - **BART:** Generally more detailed and coherent on summarization tasks.
    - **T5-small:** More concise but sometimes omits important details; must prefix with `"summarize: "`.


## Exercise 4: Simple Chunked Summarization Function

**Task:**
Implement `smart_summarize(text, chunk_size=200, overlap=50)` that:

1. Splits `text` into overlapping chunks of `chunk_size`.
2. Summarizes each chunk.
3. Concatenates chunk summaries.
```python
def smart_summarize(text, chunk_size=200, overlap=50):
    sentences = text.split('. ')
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent + ". "
        else:
            chunks.append(current)
            current = " ".join(current.split()[-overlap:]) + sent + ". "
    chunks.append(current)
    
    summaries = [summarizer(c, max_length=45, min_length=20, do_sample=False)[0]["summary_text"]
                 for c in chunks]
    return " ".join(summaries)

# Usage
print(smart_summarize(text))
```

- **Solution Explanation:**
    - **Chunking:** Ensures no chunk exceeds token limits.
    - **Overlap:** Maintains context between chunks.
    - **Concatenation:** Builds overall summary from partial summaries.


## Exercise 5: Beam Search vs. Nucleus Sampling

**Task:**
Generate summaries with different decoding strategies and compare:

```python
methods = {
    "greedy": {"do_sample": False},
    "beam":   {"do_sample": False, "num_beams":4},
    "nucleus":{"do_sample": True, "top_p":0.9, "temperature":0.8},
    "top_k":  {"do_sample": True, "top_k":50, "temperature":0.7}
}

for name, params in methods.items():
    s = summarizer(text, max_length=45, min_length=20, **params)[0]["summary_text"]
    print(f"{name} → {s}\n")
```

- **Solution Observations:**
    - **Greedy:** Very deterministic; may be repetitive.
    - **Beam:** Balanced quality; more comprehensive than greedy.
    - **Nucleus/Top-k:** Introduces diversity; may generate varied phrasings or less focused summaries.

**Tips for Students:**

- Run each cell and inspect outputs closely.
- Note trade-offs between brevity, coherence, and creativity.
- Adjust parameters further to fine-tune behavior.
- Document your findings to reinforce learning.

