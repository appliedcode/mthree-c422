## Exercise A: Machine Translation (English → French)

**Task:**
Translate these three English sentences into French using `Helsinki-NLP/opus-mt-en-fr`:

1. “The weather is beautiful today.”
2. “I would like to order a coffee, please.”
3. “What time does the train to Paris depart?”

Initialize a translation pipeline and print each English sentence alongside its French translation.

**Solution:**

```python
from transformers import pipeline

# Initialize translation pipeline
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

# Sentences to translate
sentences = [
    "The weather is beautiful today.",
    "I would like to order a coffee, please.",
    "What time does the train to Paris depart?"
]

# Translate and print
for eng in sentences:
    fr = translator(eng)[0]['translation_text']
    print(f"EN: {eng}")
    print(f"FR: {fr}\n")
```


## Exercise B: Paraphrasing with T5

**Task:**
Use `google/t5-small` as a paraphrasing model. Prefix each input with `"paraphrase: "` and generate two paraphrases for this sentence:
> “Deep learning models require large amounts of data to train effectively.”

Set `num_return_sequences=2`, `do_sample=True`, and `temperature=0.8`. Print both paraphrased outputs.

**Solution:**

```python
from transformers import pipeline

# Initialize paraphrase pipeline
paraphraser = pipeline(
    "text2text-generation",
    model="google/t5-small"
)

# Input sentence
sentence = "Deep learning models require large amounts of data to train effectively."

# Generate two paraphrases
outputs = paraphraser(
    "paraphrase: " + sentence,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.8,
    max_length=64
)

# Print results
for i, out in enumerate(outputs, 1):
    print(f"Paraphrase {i}: {out['generated_text']}")
```


## Exercise C: Question Answering over a Context

**Task:**
Use `deepset/roberta-base-squad2` as an encoder–decoder question-answering pipeline. Given the context:
> “Marie Curie was the first woman to win a Nobel Prize, and she remains the only person awarded Nobel Prizes in two different scientific fields.”
Answer the question:
> “In how many fields did Marie Curie win Nobel Prizes?”

**Solution:**

```python
from transformers import pipeline

# Initialize QA pipeline
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Context and question
context = (
    "Marie Curie was the first woman to win a Nobel Prize, "
    "and she remains the only person awarded Nobel Prizes "
    "in two different scientific fields."
)
question = "In how many fields did Marie Curie win Nobel Prizes?"

# Get answer
result = qa(question=question, context=context)
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
```


## Exercise D: Code Generation from Description

**Task:**
Use `Salesforce/codet5-small` as an encoder–decoder code-generation pipeline. Provide this prompt:
> “Write a Python function that takes a list of numbers and returns the list sorted in ascending order.”

Generate and print the complete function code.

**Solution:**

```python
from transformers import pipeline

# Initialize code generation pipeline
codegen = pipeline("text2text-generation", model="Salesforce/codet5-small")

# Prompt
prompt = (
    "Write a Python function that takes a list of numbers and returns "
    "the list sorted in ascending order."
)

# Generate code
output = codegen(prompt, max_length=128, do_sample=False)
print(output[0]['generated_text'])
```


## Exercise E: Text Infilling with BART

**Task:**
Use `facebook/bart-large` for text infilling. Provide this incomplete sentence with `[MASK]` tokens:
> “To make the recipe, first preheat the oven to [MASK] degrees, then [MASK] the butter and sugar together.”

Set `task="fill-mask"` and print the top 3 filled-in options for each mask.

*Hints for all exercises:*

- Initialize the appropriate `pipeline` with the correct `model` and `task`.
- For translation and paraphrasing, adjust sampling parameters.
- For fill-mask, the pipeline will return a list of possible tokens for each mask.
- Print inputs and outputs clearly for comparison.

**Solution:**

```python
from transformers import pipeline

# Initialize fill-mask pipeline
fill_mask = pipeline("fill-mask", model="facebook/bart-large")

# Incomplete sentence with masks
sentence = (
    "To make the recipe, first preheat the oven to <mask> degrees, "
    "then <mask> the butter and sugar together."
)

# Get top 3 options for each mask
results = fill_mask(sentence, top_k=3)

# Print results
for res in results:
    print(f"Token: {res['token_str']}, Score: {res['score']:.4f}")
    print(f"Sequence: {res['sequence']}\n")
```

