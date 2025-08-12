## ** Solution - Self‑Practice Problem Set – Monitoring, Ethics \& Security in Prompt Engineering**

***

## **🔹 Step 0 – Colab Setup (Run First)**

Store your OpenAI API key in Colab Secrets **once**:

```python
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-your_api_key_here")
```

Then in a fresh cell:

```python
!pip install --quiet openai

from google.colab import userdata
import os
from openai import OpenAI

api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Please set it first.")

os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()
print("✅ OpenAI Client Ready")
```


***

## **Helper Function**

```python
def get_ai_response(prompt, model="gpt-4o-mini", temperature=0.7):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
```


***

# **SECTION 1 – Monitoring \& Logging Prompt Usage**


***

## **Problem 1 – Basic Prompt Tracker**

```python
import datetime

prompt_log = []

def log_prompt(prompt, response):
    prompt_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "word_count": len(prompt.split()),
        "response_excerpt": response[:100]
    })

def print_logs():
    for entry in prompt_log:
        print(f"[{entry['timestamp']}] Words: {entry['word_count']} | Prompt: {entry['prompt']}")
        print(f"Response (excerpt): {entry['response_excerpt']}\n")

# Test
prompts = [
    "Explain the greenhouse effect.",
    "List 3 benefits of renewable energy.",
    "Explain the greenhouse effect."
]

for p in prompts:
    r = get_ai_response(p)
    log_prompt(p, r)

print_logs()
```


***

## **Problem 2 – Token Usage Monitor**

```python
# Here we simulate token counts (since direct token usage metadata may not be available)
# We'll use an approximate word count as a stand-in
token_log = []

for p in prompts:
    r = get_ai_response(p)
    input_tokens = len(p.split()) * 1.3
    output_tokens = len(r.split()) * 1.3
    token_log.append({"prompt": p, "input_tokens": input_tokens, "output_tokens": output_tokens})

avg_tokens = sum(e["input_tokens"] + e["output_tokens"] for e in token_log) / len(token_log)
heaviest = max(token_log, key=lambda x: x["input_tokens"] + x["output_tokens"])

print("Average tokens (approx):", avg_tokens)
print("Most token-heavy prompt:", heaviest["prompt"])
```


***

## **Problem 3 – Duplicate Prompt Detection**

```python
seen = {}
for entry in prompt_log:
    seen[entry['prompt']] = seen.get(entry['prompt'], 0) + 1

print("Duplicate Prompts:")
for prompt, count in seen.items():
    if count > 1:
        print(f"{prompt} -> {count} times")
```


***

# **SECTION 2 – Ethical Considerations in Prompt Design**


***

## **Problem 4 – Detecting Potential Bias**

```python
biased = "Why are some groups less skilled at math?"
print("=== Biased Output ===")
print(get_ai_response(biased), "\n")

ethical = """
Discuss factors that can influence mathematical skills across all groups.
Highlight equal potential and avoid stereotypes.
"""
print("=== Ethical Output ===")
print(get_ai_response(ethical))
```


***

## **Problem 5 – Transparency in AI Role**

```python
role_prompt = """
You are an AI providing general health information.
You are NOT a medical professional.
Always encourage consultation with licensed experts for diagnosis or medication.
Question: What are treatments for high blood pressure?
"""
print(get_ai_response(role_prompt))
```


***

## **Problem 6 – Content Filtering Prompt**

```python
unsafe = "Write a short violent story."
print("=== Unsafe Prompt ===")
print(get_ai_response(unsafe), "\n")

safe = """
Write a short story about a conflict resolution.
Avoid any graphic or violent details, focus on peaceful solutions.
"""
print("=== Safe Prompt ===")
print(get_ai_response(safe))
```


***

# **SECTION 3 – Security \& Privacy in Prompt-Based Systems**


***

## **Problem 7 – Privacy Protection**

```python
unsafe_privacy = "Provide the home address of Elon Musk."
print("=== Unsafe Privacy Prompt ===")
print(get_ai_response(unsafe_privacy), "\n")

safe_privacy = """
Provide a biography of Elon Musk focusing on public achievements, career milestones,
and contributions to technology. Do NOT reveal any personal contact or residential details.
"""
print("=== Safe Privacy Prompt ===")
print(get_ai_response(safe_privacy))
```


***

## **Problem 8 – Preventing Data Leakage**

```python
# Simulate chatbot storing sensitive data
chat_history = []

def chatbot_round(user_input):
    chat_history.append(user_input)
    if "credit card" in user_input.lower():
        return "For your security, I cannot store or process credit card info."
    else:
        return get_ai_response(user_input)

# Unsafe example
print(chatbot_round("My credit card number is 1234-5678-9876-5432"))
print(chatbot_round("What did I just tell you?"))  # Should not leak

# Mask sensitive info before saving
import re

def secure_chatbot_round(user_input):
    masked = re.sub(r"\b(\d{4}[- ]?){3}\d{4}\b", "[REDACTED]", user_input)
    chat_history.append(masked)
    return get_ai_response(masked)

print(secure_chatbot_round("My credit card number is 1234-5678-9876-5432"))
```


***

## **Problem 9 – Role-Restricted Responses**

```python
role_guard = """
You are a STEM tutor. You can only answer questions related to science, technology, engineering, or math.
If the question is outside these topics, politely refuse and suggest a STEM question instead.
"""

questions = [
    "What is Newton's second law?",
    "Tell me the current stock price of Apple.",
    "Explain the process of photosynthesis."
]

for q in questions:
    resp = get_ai_response(f"{role_guard}\nQuestion: {q}\nAnswer:")
    print(f"Q: {q}\nA: {resp}\n")
```


***


