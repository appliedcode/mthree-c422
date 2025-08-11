# **💡 Solutions: Prompting Practice Lab**

***

## **1️⃣ Zero‑Shot Prompting Solutions**

### **Z1: News Headline Classification**

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

texts = [
    "NASA launches new mission to search for exoplanets.",
    "The city council approved new recycling rules yesterday.",
    "Barcelona defeats Real Madrid in a thrilling match."
]
candidate_labels = ["science", "sports", "politics", "environment"]

for text in texts:
    result = classifier(text, candidate_labels)
    print(f"Text: {text}")
    print(f"Predicted Label: {result['labels'][0]} (Score: {result['scores'][0]:.2f})\n")
```

**💭 Expected Output:**

```
Text: NASA launches new mission to search for exoplanets.
Predicted Label: science (Score: ~0.97)

Text: The city council approved new recycling rules yesterday.
Predicted Label: environment (Score: ~0.88)  # politics might be 2nd

Text: Barcelona defeats Real Madrid in a thrilling match.
Predicted Label: sports (Score: ~0.99)
```


***

### **Z2: Sentiment Classification (Zero‑Shot)**

```python
texts = [
    "What a disappointing experience, nothing went right.",
    "Absolutely loved the atmosphere and the food was perfect!",
    "It was okay, nothing stood out as good or bad."
]
candidate_labels = ["positive", "negative", "neutral"]

for text in texts:
    result = classifier(text, candidate_labels)
    print(f"Text: {text}")
    print(f"Predicted Label: {result['labels'][0]} (Score: {result['scores'][0]:.2f})\n")
```

**💭 Expected Output:**

```
Text: What a disappointing experience, nothing went right.
Predicted Label: negative (Score: ~0.95)

Text: Absolutely loved the atmosphere and the food was perfect!
Predicted Label: positive (Score: ~0.99)

Text: It was okay, nothing stood out as good or bad.
Predicted Label: neutral (Score: ~0.85)  # positive/negative lower
```


***

## **2️⃣ Few‑Shot Prompting Solutions**

Using a text‑generation model like Falcon‑7B‑Instruct:

```python
from transformers import pipeline
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")
```


***

### **F1: Intent Classification**

```python
prompt = """Classify each customer support inquiry as Order, Technical Issue, or Complaint.

Inquiry: "My package hasn't arrived and it's a week late."
Intent: Complaint

Inquiry: "Can I change my delivery address after ordering?"
Intent: Order

Inquiry: "The app crashes when I try to log in."
Intent: Technical Issue

Inquiry: "The product is broken, please help."
Intent:"""

response = generator(prompt, max_new_tokens=10, temperature=0.2)
print(response[0]['generated_text'])
```

**💭 Expected Continuation:**

```
...Intent: Complaint
```

**Explanation:** Matches the few‑shot examples — broken product is clearly a complaint.

***

### **F2: Movie Genre Classification**

```python
prompt = """Classify the following movie plots as Comedy, Drama, or Action.

Plot: "A group of friends go on a hilarious road trip and get into wild situations."
Genre: Comedy

Plot: "A detective teams up with a spy to stop an international heist."
Genre: Action

Plot: "A family comes together during a difficult time to heal old wounds."
Genre: Drama

Plot: "Two rival chefs compete to win the city's food festival."
Genre:"""

response = generator(prompt, max_new_tokens=10, temperature=0.2)
print(response[0]['generated_text'])
```

**💭 Expected Continuation:**

```
...Genre: Comedy
```

**Reasoning:** A food festival with rival chefs generally fits “Comedy” in tone.

***

## **3️⃣ Chain‑of‑Thought (CoT) Solutions**

We add step-by-step logic:

***

### **C1: Math Word Problem**

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

**💭 Expected Step-by-Step Reasoning:**

```
1. Start with 24 candies.
2. Gives away 7: 24 - 7 = 17.
3. Buys 5 more: 17 + 5 = 22.

Answer: 22
```


***

### **C2: Everyday Reasoning**

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

response = generator(prompt, max_new_tokens=40, temperature=0)
print(response[0]['generated_text'])
```

**💭 Expected Step-by-Step Reasoning:**

```
1. Start with 15 chocolate bars.
2. Gives away 5: 15 - 5 = 10.
3. Buys 3 more: 10 + 3 = 13.

Answer: 13
```


***

## **📌 Summary Table of Expected Answers**

| Exercise | Prompting Type | Expected Answer (Top Result) |
| :-- | :-- | :-- |
| Z1 | Zero-Shot | Science / Environment / Sports |
| Z2 | Zero-Shot | Negative / Positive / Neutral |
| F1 | Few-Shot | Complaint |
| F2 | Few-Shot | Comedy |
| C1 | CoT | 22 |
| C2 | CoT | 13 |


***

