# \# -*- coding: utf-8 -*-

### Exercise 1: Translate Multiple Sentences

**Problem:**
Create a list of at least 5 English sentences of your choice (e.g., about daily routines, hobbies, or technology). Write code to translate each sentence into French using the `translator` pipeline. Print each original sentence alongside its French translation.

### Exercise 2: Experiment with Different Translation Models

**Problem:**
Try using a different pre-trained translation model available on Hugging Face, such as:

- `"Helsinki-NLP/opus-mt-en-de"` (English to German)
- `"Helsinki-NLP/opus-mt-en-es"` (English to Spanish)

Translate the same sample sentence `"He works at Apple and eats an apple every day."` into German and Spanish, then compare the outputs.

### Exercise 3: Reverse Translation (French to English)

**Problem:**
Load the French to English translation pipeline (`"Helsinki-NLP/opus-mt-fr-en"`). Translate the French output from your original example back to English. Compare the two English sentences and analyze any changes or loss of meaning.

### Exercise 4: Handle Long Text Translation

**Problem:**
Write a function to translate long English paragraphs by splitting the text into chunks of approximately 100 tokens (or characters), translate each chunk separately, and then join the translated chunks to get the full French translation.

Use this function on a paragraph of at least 150-200 words and observe how chunking affects translation coherence.

### Exercise 5: Translation with Controlled Output Length

**Problem:**
Translate the sentence `"The quick brown fox jumps over the lazy dog."` into French, experimenting with the arguments:

- `max_length=10`
- `max_length=20`
- `max_length=40`

Compare the translations and discuss how `max_length` influences the length and completeness of the translated output.


