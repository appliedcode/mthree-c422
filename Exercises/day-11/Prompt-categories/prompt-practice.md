# Lab Exercise 1 — Zero-Shot Prompt Engineering

### Objective

- Classify text into target categories without any training, by crafting a natural language instruction.
"""


# Install Hugging Face Transformers

!pip install transformers -q

from transformers import pipeline

# Load zero-shot classification pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- PROMPT / INPUT ---

texts = [
"The prime minister met foreign leaders to discuss trade agreements.",
"The local team won the championship after a thrilling final.",
"Stock markets surged after tech companies posted strong earnings."
]
candidate_labels = ["sports", "politics", "business"]

# -------------------------------

for text in texts:
print("=== PROMPT SENT TO MODEL ===")
print(text)
result = classifier(text, candidate_labels)
print(f"Predicted Label: {result['labels'][0]} (Score: {result['scores'][0]:.2f})")
print("-" * 60)

"""\# Lab Exercise 2 — Few-Shot Prompting

### Objective

- Show how adding a few examples to the prompt improves the model’s output accuracy.
"""


# Exercise

# Ask a model to classify movie review sentiment using few-shot prompting.

!pip install transformers accelerate -q

from transformers import pipeline

generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

# --- THIS IS THE PROMPT ---

prompt = """Classify the sentiment of the following movie reviews as Positive or Negative.

Review: "I loved this film! The story was captivating."
Sentiment: Positive

Review: "This was the worst acting I have ever seen."
Sentiment: Negative

Review: "Absolutely wonderful soundtrack and direction."
Sentiment: Positive

Review: "The plot was dull and uninteresting."
Sentiment:"""

# --------------------------

print("=== PROMPT SENT TO MODEL ===")
print(prompt)

response = generator(prompt, max_new_tokens=20, temperature=0.2)

print("\n=== MODEL RESPONSE ===")
print(response[0]['generated_text'])

"""\# Lab Exercise 3 — Chain-of-Thought Prompting

## Objective

- Guide the model to output step-by-step reasoning before giving the final answer.
"""


# Exercise

# Solve math word problems with explicit reasoning.

# Use the same Falcon-7b-instruct model as above for reasoning

# --- THIS IS THE PROMPT ---

prompt = """
Q: Tom has 3 apples. He buys 2 more packs with 4 apples each. Then he eats 3 apples. How many apples does he have left?
Let's think step-by-step.

1. Tom starts with 3 apples.
2. He buys 2 packs of 4 apples = 2 * 4 = 8 apples.
3. Total apples now = 3 + 8 = 11 apples.
4. He eats 3 apples, so remaining = 11 - 3 = 8 apples.
Answer: 8

Q: Sarah has 10 pencils. She gives 3 to John and 2 to Mary. How many does she have left?
Let's think step-by-step.
"""

# --------------------------

print("=== PROMPT SENT TO MODEL ===")
print(prompt)

response = generator(prompt, max_new_tokens=50, temperature=0)

print("\n=== MODEL RESPONSE ===")
print(response[0]['generated_text'])

Generate few execises for student to try on their own.

Absolutely! Here are **self-practice exercises** (with realistic prompts) for students to try Zero-shot, Few-shot, and Chain-of-Thought (CoT) prompting on their own — using Colab and Hugging Face, as in your examples.
Each exercise suggests a realistic scenario and gives instructions for how to construct the prompt and evaluate responses.

***

# 📝 Lab Student Exercises: Prompting Styles


***

## **Zero-Shot Prompting Exercises**

**Objective:** Practice giving a model classification or extraction tasks **without** providing any examples.

### Exercise Z1: News Headline Classification

Replace the `texts` and `candidate_labels` in your zero-shot pipeline.

```python
texts = [
    "NASA launches new mission to search for exoplanets.",
    "The city council approved new recycling rules yesterday.",
    "Barcelona defeats Real Madrid in a thrilling match."
]
candidate_labels = ["science", "sports", "politics", "environment"]

# Run using your zero-shot classification pipeline.
```

- **Task:** For each headline, write down which category you think it should be, then compare to the model.
- **Reflect:** Did the model choose as you expected? Were any surprising?

***

### Exercise Z2: Sentiment Classification (Zero-Shot)

```python
texts = [
    "What a disappointing experience, nothing went right.",
    "Absolutely loved the atmosphere and the food was perfect!",
    "It was okay, nothing stood out as good or bad."
]
candidate_labels = ["positive", "negative", "neutral"]

# Run using your zero-shot classification pipeline.
```

- **Task:** How does the model rate each? Are there any edge cases ("It was okay")?
- **Bonus:** Try adding new candidate labels like "mixed" or "not sure."

***

## **Few-Shot Prompting Exercises**

**Objective:** Give a few examples in your prompt to help the model learn a pattern, then ask it to continue.

### Exercise F1: Intent Classification (Few-Shot)

Compose a text-generation prompt like:

```python
prompt = """Classify each customer support inquiry as Order, Technical Issue, or Complaint.

Inquiry: "My package hasn't arrived and it's a week late."
Intent: Complaint

Inquiry: "Can I change my delivery address after ordering?"
Intent: Order

Inquiry: "The app crashes when I try to log in."
Intent: Technical Issue

Inquiry: "The product is broken, please help."
Intent:
"""
response = generator(prompt, max_new_tokens=10, temperature=0.2)
print(response[0]['generated_text'])
```

- **Task:** Add 1 or 2 more inquiries of your own and assess the model's outputs for those new items.

***

### Exercise F2: Movie Genre Classification (Few-Shot)

Prepare a prompt such as:

```python
prompt = """Classify the following movie plots as Comedy, Drama, or Action.

Plot: "A group of friends go on a hilarious road trip and get into wild situations."
Genre: Comedy

Plot: "A detective teams up with a spy to stop an international heist."
Genre: Action

Plot: "A family comes together during a difficult time to heal old wounds."
Genre: Drama

Plot: "Two rival chefs compete to win the city's food festival."
Genre:
"""
# Send to the text-generation pipeline
```

- **Bonus:** Try inventing your own plot and see if the model classifies it correctly!

***

## **Chain-of-Thought (Step-by-Step) Prompting Exercises**

**Objective:** Guide the model to show its reasoning, not just answers. Use "Let's think step by step."

### Exercise C1: Math Word Problem (CoT)

Continue this pattern with your own question.

```python
prompt = """
Q: There are 18 oranges. If 6 are eaten and then 4 more are bought, how many are there now?
Let's think step by step.

1. Start with 18 oranges.
2. 6 are eaten: 18 - 6 = 12 oranges left.
3. 4 more are bought: 12 + 4 = 16 oranges.

Answer: 16

Q: John has 24 candies. He gives 7 to his friend and then buys 5 more. How many candies does John have?
Let's think step by step.
"""
response = generator(prompt, max_new_tokens=50, temperature=0)
print(response[0]['generated_text'])
```

- **Task:** Write at least 1 new word problem and try to see if the model solves it step by step.

***

### Exercise C2: Everyday Reasoning (CoT)

Try a real-life reasoning question. Prompt example:

```python
prompt = """
Q: Amy has 20 dollars. She spends 8 dollars on lunch and 5 dollars on a gift. How much money does she have left?
Let's think step by step.

1. Start with 20 dollars.
2. Spend 8 dollars: 20 - 8 = 12 dollars left.
3. Spend 5 dollars: 12 - 5 = 7 dollars left.

Answer: 7

Q: Ben has 15 chocolate bars. He shares 5 with his friends and then buys 3 more. How many does he have now?
Let's think step by step.
"""
# Complete as above
```

- **Bonus:** Try a logic puzzle or "If-Then" reasoning and see if you can get stepwise explanations.

***

## **Reflection Questions**

- Did step-by-step (CoT) prompting improve the model's explanation or accuracy?
- In which scenario did few-shot examples matter most? Why?
- What surprised you most about the zero-shot result?

***

## **Extra Challenge**

- Swap out the genres, sentiment labels, or categories in any example and invent new test examples yourself. Experiment with the order of instructions and see what changes!

***

