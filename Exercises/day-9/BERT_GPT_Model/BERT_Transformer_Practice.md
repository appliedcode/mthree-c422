# Additional Transformer-Based Encoder–Decoder Practice Exercises with Solutions
## Exercise 6: Interactive Summarizer CLI

**Task:**
Build a command-line interface that repeatedly prompts the user for text input and prints its summary until the user types “exit”. Handle empty inputs gracefully.

```python
def interactive_summarizer():
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Enter text to summarize (type 'exit' to quit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "exit":
            print("Goodbye!")
            break
        if not text:
            print("Please enter some text.")
            continue
        summary = summarizer(text, max_length=45, min_length=20, do_sample=False)[0]["summary_text"]
        print("Summary:", summary, "\n")

# Run:
interactive_summarizer()
```

**Solution Explanation:**

- Uses a loop to prompt until “exit.”
- Checks for empty input.
- Invokes the summarizer and prints results.


## Exercise 7: Multilingual Summarization

**Task:**
Use a multilingual model (`facebook/mbart-large-50-many-to-many-mmt`) to summarize texts in English, French, and Spanish. Include language codes in input.

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def mbart_summarize(text, src_lang, tgt_lang="en_XX"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    generated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                               max_length=45, min_length=20)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

samples = {
    "en_XX": "The quick brown fox jumps over the lazy dog.",
    "fr_XX": "Le renard brun rapide saute par-dessus le chien paresseux.",
    "es_XX": "El rápido zorro marrón salta sobre el perro perezoso."
}
for lang, txt in samples.items():
    print(lang, "→", mbart_summarize(txt, src_lang=lang))
```

**Solution Explanation:**

- Sets `src_lang` and `forced_bos_token_id` for target language.
- Works for any supported language pair.


## Exercise 8: Summarization Quality Evaluation

**Task:**
Implement a simple ROUGE-1 score calculation (unigram overlap) between generated and reference summaries.

```python
def rouge_1(reference, generated):
    ref_unigrams = set(reference.lower().split())
    gen_unigrams = set(generated.lower().split())
    overlap = ref_unigrams & gen_unigrams
    precision = len(overlap) / len(gen_unigrams) if gen_unigrams else 0
    recall = len(overlap) / len(ref_unigrams) if ref_unigrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# Example:
ref = "The battery technology stores twice as much energy."
gen = summarizer(news_article, max_length=30, min_length=10)[0]["summary_text"]
print(rouge_1(ref, gen))
```

**Solution Explanation:**

- Computes set overlap of words.
- Calculates precision, recall, and F1 for ROUGE-1.


## Exercise 9: Chunked Translation and Summarization

**Task:**
For a long English text, first translate it to French, then summarize the French translation using the same BART summarizer. Handle texts exceeding token limits by chunking.

```python
from transformers import pipeline
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def translate_and_summarize(text, chunk_size=200, overlap=50):
    # Chunk English text
    sentences = text.split('. ')
    chunks, buf = [], ""
    for sent in sentences:
        if len(buf) + len(sent) < chunk_size:
            buf += sent + ". "
        else:
            chunks.append(buf); buf = sent + ". "
    chunks.append(buf)
    # Process each chunk
    outputs = []
    for chunk in chunks:
        fr = translator(chunk)[0]["translation_text"]
        summary = summarizer(fr, max_length=45, min_length=20, do_sample=False)[0]["summary_text"]
        outputs.append(summary)
    return " ".join(outputs)

# Run:
print(translate_and_summarize(long_text))
```

**Solution Explanation:**

- Splits long text into manageable chunks.
- Translates each chunk, then summarizes the translation.
- Concatenates results.


## Exercise 10: Knowledge Distillation for Summarization

**Task (Advanced):**
Distill a smaller BART model (`facebook/mbart-base`) from a larger teacher (`facebook/bart-large-cnn`) using a small dataset. Use teacher logits as soft targets.

```python
# Pseudocode outline (detailed implementation requires training loop)
from transformers import BartForConditionalGeneration, BartTokenizer
teacher = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
student = BartForConditionalGeneration.from_pretrained("facebook/mbart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# 1. Tokenize your small dataset of (text, summary)
# 2. For each batch:
#    - teacher_outputs = teacher(**inputs, labels=labels, output_logits=True)
#    - student_outputs = student(**inputs, labels=labels)
#    - loss_soft = KLDivLoss(student_logits, teacher_logits / T)
#    - loss_hard = CrossEntropyLoss(student_logits, labels)
#    - total_loss = alpha * loss_soft + (1-alpha) * loss_hard
#    - Backpropagate total_loss for student only
```

**Solution Explanation:**

- Uses a combination of soft-target loss (distillation) and traditional loss.
- Temperature `T` and weight `alpha` control distillation strength.
- Produces a smaller model with similar behavior.

These exercises, with provided solution outlines, will deepen your hands-on understanding of Transformer encoder–decoder models, covering interactivity, multilingual processing, evaluation, and advanced model compression techniques.

