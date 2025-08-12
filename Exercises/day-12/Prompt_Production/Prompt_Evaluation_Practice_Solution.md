## ** Solution - Self‑Practice Problem Set – Prompt Integration \& Performance Evaluation**

***

## **🔹 Step 0 – Colab Setup (Run First)**

**Save your API key once** in Colab secrets:

```python
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-your_api_key_here")
```

**Install \& initialize OpenAI:**

```python
!pip install --quiet openai

from google.colab import userdata
import os, time
from openai import OpenAI

api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Please set it in Colab Secrets (userdata).")

os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()
print("✅ OpenAI client ready")
```


***

## **Helper Function**

```python
def get_ai_response(prompt, model="gpt-4o-mini", temperature=0.7):
    try:
        start_time = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        latency = time.time() - start_time
        return resp.choices[0].message.content.strip(), latency
    except Exception as e:
        return f"Error: {e}", None
```


***

# **SECTION 1 – Integrating Prompts in Applications \& Pipelines**


***

## **Problem 1 – Multi‑Stage Prompt Pipeline**

```python
# Step 1: Summarize article
article = """
Artificial intelligence is being used to improve renewable energy efficiency.
Smart algorithms help predict electricity demand, optimize grid usage,
and integrate solar and wind power into existing infrastructure.
"""

summary_prompt = f"Summarize this article in 2 concise sentences:\n\n{article}"
summary, _ = get_ai_response(summary_prompt)
print("Summary:", summary, "\n")

# Step 2: Generate quiz questions from summary
quiz_prompt = f"From this summary, generate 3 quiz questions:\n\n{summary}"
quiz, _ = get_ai_response(quiz_prompt)
print("Quiz Questions:\n", quiz)
```


***

## **Problem 2 – Role Switching in Prompt Pipelines**

```python
base_question = "Explain the importance of cybersecurity for small businesses."

role_explainer = f"You are a technical explainer. Break the following topic into bullet points:\n{base_question}"
role_coach = f"You are a motivational mentor. Encourage the reader to learn about:\n{base_question}"

print("=== Technical Explainer ===")
print(get_ai_response(role_explainer)[0], "\n")

print("=== Motivational Mentor ===")
print(get_ai_response(role_coach)[0])
```


***

## **Problem 3 – Prompt Version Control Simulation**

```python
prompt_v1 = "Give travel tips for visiting Japan."
prompt_v2 = "You are a travel assistant. Suggest 3 budget-friendly travel tips for Japan."
prompt_v3 = """You are a helpful travel assistant.
Example:
1. Visit free attractions like parks
2. Eat at local markets
3. Use public transport
Now give 3 budget-friendly Japan travel tips in similar style."""

versions = [prompt_v1, prompt_v2, prompt_v3]

for i, pv in enumerate(versions, 1):
    print(f"--- Version {i} ---")
    print(get_ai_response(pv)[0], "\n")
```


***

# **SECTION 2 – Prompt Performance Measurement \& Evaluation**


***

## **Problem 4 – Keyword-Based Scoring**

```python
from collections import Counter

prompt = "Explain how photosynthesis works."
expected_keywords = ["photosynthesis", "sunlight", "chlorophyll", "glucose", "oxygen"]

response, _ = get_ai_response(prompt)
print("Response:\n", response, "\n")

score = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
print(f"Keyword Match Score: {score}/{len(expected_keywords)}")
```


***

## **Problem 5 – Response Length Evaluation**

```python
runs = [get_ai_response("What are the benefits of exercise?")[0] for _ in range(3)]
lengths = [len(r.split()) for r in runs]

for i, r in enumerate(runs, 1):
    print(f"Run {i} length: {lengths[i-1]} words")

print("Average length:", sum(lengths) / len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
```


***

## **Problem 6 – Consistency Measurement**

```python
import difflib

prompt = "Define cloud computing."
responses = [get_ai_response(prompt)[0] for _ in range(5)]

scores = []
for i in range(len(responses) - 1):
    sim = difflib.SequenceMatcher(None, responses[i], responses[i+1]).ratio()
    scores.append(sim)
    print(f"Similarity Run {i+1} vs Run {i+2}: {sim:.2f}")

print("Average similarity:", sum(scores) / len(scores))
```


***

## **Problem 7 – Simulated User Feedback Loop**

```python
import random

prompt_versions = [
    "Explain blockchain in simple terms.",
    "Explain blockchain in simple terms with a real-world analogy."
]

log = []

for pv in prompt_versions:
    response, _ = get_ai_response(pv)
    rating = random.randint(1, 5)  # Simulated feedback
    log.append({"prompt": pv, "excerpt": response[:60], "rating": rating})

for entry in log:
    print(entry)

avg_ratings = {}
for pv in prompt_versions:
    ratings = [e["rating"] for e in log if e["prompt"] == pv]
    avg_ratings[pv] = sum(ratings) / len(ratings)

print("Average ratings:", avg_ratings)
```


***

## **Problem 8 – Latency Tracking for Prompt Performance**

```python
test_prompts = [
    "Summarize the plot of Romeo and Juliet in 2 sentences.",
    "Write a 200-word essay on climate change impacts."
]

for p in test_prompts:
    resp, latency = get_ai_response(p)
    print(f"Prompt: {p}")
    print(f"Latency: {latency:.2f} sec | Word count: {len(resp.split())}")
    print()
```


***

## ✅ What Students See

With this **solution + code**:

- They run both **integration workflows** (pipelines, role switches, versioning).
- They measure **prompt quality** via keywords, length, consistency, ratings, and latency.
- They can **tweak prompts** and re-run metrics to see tangible changes.

***

