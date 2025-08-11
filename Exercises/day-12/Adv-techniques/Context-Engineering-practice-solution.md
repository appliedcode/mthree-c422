## **Setup – Run This First in Colab**

Make sure you’ve securely stored your API key in Colab Secrets (only once per account/session):

```python
# Run this ONCE to store key securely (do NOT hardcode it in notebooks)
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-YOURKEYHERE")
```

Then, in a fresh cell:

```python
!pip install --quiet openai

from google.colab import userdata
import os
from openai import OpenAI

# Load key from Colab Secrets
api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key not found in Colab Secrets. Please set it first.")

# Set env variable for OpenAI
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()
print("✅ OpenAI API Key loaded and client initialized.")
```


***

## **Helper Function**

```python
def generate_response(prompt, model="gpt-4o-mini", temperature=0.7):
    """
    Sends prompt to chosen OpenAI model and returns string response
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
```


***

## **Part A – Context Management Solutions**


***

### **Problem 1 – Story Continuation Without Losing Details**

```python
# WITHOUT Context
print("=== Story WITHOUT Context ===")
story_start = "Once upon a time, a boy found a mysterious glowing stone."
prompts = [
    story_start,
    "Add more events to the story.",
    "Introduce a new character.",
    "Create a twist ending."
]
for p in prompts:
    print(f"User: {p}")
    print("AI:", generate_response(p), "\n")

# WITH Context
print("\n=== Story WITH Context ===")
context = ""
for p in prompts:
    full_prompt = context + f"\nUser: {p}\nAI:"
    reply = generate_response(full_prompt)
    print(f"User: {p}\nAI: {reply}\n")
    context += f"\nUser: {p}\nAI: {reply}"
```


***

### **Problem 2 – FAQ Bot Simulation**

```python
topic_questions = [
    "What is Python used for?",
    "What are data types in Python?",
    "Explain Python lists in one sentence."
]

print("=== FAQ Bot WITHOUT Context ===")
for q in topic_questions:
    print(f"Q: {q}\nAI: {generate_response(q)}\n")

print("\n=== FAQ Bot WITH Context ===")
context = ""
for q in topic_questions:
    full_prompt = context + f"\nUser: {q}\nAI:"
    ans = generate_response(full_prompt)
    print(f"Q: {q}\nAI: {ans}\n")
    context += f"\nUser: {q}\nAI: {ans}"
```


***

### **Problem 3 – Context Truncation Challenge**

```python
MAX_CONTEXT_CHARS = 500  # Simulated limit
context = ""
turns = [f"Turn {i}: Add a new event to our ongoing fantasy story." for i in range(1, 11)]

for turn in turns:
    # If context too long, truncate oldest part
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
    
    full_prompt = context + f"\nUser: {turn}\nAI:"
    reply = generate_response(full_prompt)
    print(f"{turn}\nAI: {reply}\n")
    context += f"\nUser: {turn}\nAI: {reply}"
```


***

## **Part B – Prompt Optimization Solutions**


***

### **Problem 4 – Recipe Request Optimization**

```python
vague = "Give me a recipe for pasta"
optimized = """Give me a vegetarian pasta recipe for a beginner cook, 
serves 2 people, includes spinach and mushrooms, excludes cheese. 
Provide the answer as an 'Ingredients' list and 'Numbered Steps'."""

print("--- Vague Prompt ---")
print(generate_response(vague), "\n")

print("--- Optimized Prompt ---")
print(generate_response(optimized))
```


***

### **Problem 5 – Study Guide Creation**

```python
subject = "Physics"
vague = f"Tell me about {subject}"
optimized = f"""Create a short study guide for {subject} containing:
1. 5 bullet points summarizing key concepts
2. A simple daily study plan for 1 week
3. One easy real-life example for each concept."""

print("--- Vague ---")
print(generate_response(vague), "\n")

print("--- Optimized ---")
print(generate_response(optimized))
```


***

### **Problem 6 – Summarization with Constraints**

```python
paragraph = """Albert Einstein was a theoretical physicist who developed the theory of relativity,
one of the two pillars of modern physics. His work is also known for its influence on
the philosophy of science. In 1921, he received the Nobel Prize in Physics for
his explanation of the photoelectric effect, a pivotal step in the development of quantum theory."""

vague = f"Summarize this paragraph:\n{paragraph}"
optimized = f"""Summarize this paragraph in exactly 3 bullet points.
Each bullet must have fewer than 12 words.
Keep only the most important factual details.
Text:\n{paragraph}"""

print("--- Vague Summary ---")
print(generate_response(vague), "\n")

print("--- Optimized Summary ---")
print(generate_response(optimized))
```


***

## **Part C – Combined Challenge**


***

### **Problem 7 – Multi-Turn Optimized Interview**

```python
context = ""
conversation = [
    "Write a job description for a Machine Learning Engineer role.",
    "Based on that job description, list 3 common interview questions.",
    "For the second question above, provide a detailed, ideal answer."
]

for turn in conversation:
    optimized_turn = f"{context}\nUser: {turn}\nAI:"
    ans = generate_response(optimized_turn)
    print(f"User: {turn}\nAI: {ans}\n")
    context += f"\nUser: {turn}\nAI: {ans}"
```


***

## **How Students Should Use This**

- Run each problem twice (vague vs optimized, with vs without context).
- Write **2–3 sentence observation summaries** for each:
    - What improved?
    - Did the AI stay consistent?
    - Was the output clearer or more relevant?

***


