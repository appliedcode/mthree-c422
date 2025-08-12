<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Generate the solution+code for the same

Got it ✅ — you want me to take the **student self‑practice problems** I just gave you for **Handling Ambiguity, Troubleshooting, and Prompt Failures** and now produce **full worked solutions + runnable Colab code** using the secure API key method and the **new OpenAI Python SDK syntax**.

Here’s the complete **solved lab** — each problem includes the **“bad” (vague/ambiguous) prompt**, the **refined/fixed prompt(s)**, the code to run them in Google Colab, and notes on expected improvements.

***

## **🔹 Colab Setup – Run This First**

Before any exercise, store your OpenAI key in Colab **once per account/session**:

```python
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-YOUR_API_KEY_HERE")
```

Then in a fresh cell:

```python
!pip install --quiet openai

from google.colab import userdata
import os
from openai import OpenAI

api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Please set it in Colab Secrets.")

os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()
print("✅ OpenAI client ready.")
```


***

## **Helper Function**

```python
def get_ai_response(prompt, model="gpt-4o-mini", temperature=0.7):
    """
    Sends a prompt to OpenAI and returns the string output.
    """
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

## **Problem 1 – Ambiguous Prompt Clarification**

```python
# Ambiguous version
ambiguous = "Describe the process."

# Refined version
refined = "Describe the water purification process focusing on filtration and disinfection."

print("=== Ambiguous Prompt ===")
print(get_ai_response(ambiguous), "\n")

print("=== Refined Prompt ===")
print(get_ai_response(refined))
```

**Expected:** Refined version is specific and relevant vs. vague mixed responses.

***

## **Problem 2 – Avoiding Hallucinations**

```python
# Loose prompt (likely to speculate)
vague = "Tell me about the most recent discovery in black hole research."

# Anchored to facts & date
anchored = """Based only on confirmed scientific findings up to 2023, 
summarize the latest known discovery about black holes. 
If not certain, reply with: "I don't know."
"""

print("--- Vague ---")
print(get_ai_response(vague), "\n")

print("--- Anchored ---")
print(get_ai_response(anchored))
```

**Expected:** Anchored prompt should avoid guessing and admit uncertainty.

***

## **Problem 3 – Troubleshooting Irrelevant Output**

```python
# Initial vague version
vague = "Write something interesting about the ocean."

# Refined with constraints
refined = """Write 5 bullet points about deep-sea creatures 
for a 10-year-old student in simple language."""

print("--- Vague ---")
print(get_ai_response(vague), "\n")

print("--- Refined ---")
print(get_ai_response(refined))
```

**Expected:** Vague = generic fun facts; refined = targeted to age, topic, format.

***

## **Problem 4 – Overloaded Task Decomposition**

```python
# Overloaded
overloaded = """Summarize the causes, effects, and significance of World War 2,
and suggest three fictional what-if scenarios about if it never happened."""

# Decomposed into smaller prompts
p1 = "Summarize the main causes of World War 2 in 5 bullet points."
p2 = "List the major effects and historical significance of World War 2."
p3 = "Suggest 3 fictional what-if scenarios for an alternate world where WW2 never happened."

print("--- Overloaded ---")
print(get_ai_response(overloaded), "\n")

print("--- Step 1 ---")
print(get_ai_response(p1), "\n")

print("--- Step 2 ---")
print(get_ai_response(p2), "\n")

print("--- Step 3 ---")
print(get_ai_response(p3))
```

**Expected:** Overloaded = mixed structure; split version = clear \& comprehensive.

***

## **Problem 5 – Meta-Prompting to Clarify Requirements**

```python
# Ambiguous
ambiguous = "Create an efficient search algorithm."

# Meta-prompt
meta = """Before writing any code, explain what 'efficient search algorithm' could mean in this context.
List possible interpretations and complexities. If unsure, ask me for clarification."""

# Refined after meta understanding
refined = """Write a Python implementation of binary search (O(log n) complexity) 
with clear comments explaining each step."""

print("--- Ambiguous ---")
print(get_ai_response(ambiguous), "\n")

print("--- Meta-Prompt ---")
print(get_ai_response(meta), "\n")

print("--- Refined ---")
print(get_ai_response(refined))
```

**Expected:** Meta-step surfaces assumptions before final refined task.

***

## **Problem 6 – Prompt Failure Recovery**

```python
# Broad prompt
broad = "Create a game."

# Iteratively refined
refined1 = "Create a simple text-based number guessing game in Python."
refined2 = """Create a Python text-based number guessing game:
- Number range 1-100
- User guesses until correct
- Provide high/low hints after each guess
- End with congratulations message
"""

print("--- Broad ---")
print(get_ai_response(broad), "\n")

print("--- First Refinement ---")
print(get_ai_response(refined1), "\n")

print("--- Second Refinement ---")
print(get_ai_response(refined2))
```

**Expected:** Specific iterations lead from random ideas to working code logic.

***

## **Problem 7 – Length \& Truncation Management**

```python
# Long prompt with many mixed questions
long_prompt = """
1. Explain quantum entanglement.
2. Summarize World War 1.
3. Give a recipe for apple pie.
4. Tell a joke.
5. Describe how rainbows form.
6. Name 10 programming languages.
(Answer all in one response)
"""

# Shorter, chunked approach
chunk1 = "Explain quantum entanglement in simple terms."
chunk2 = "Summarize World War 1 in 5 bullet points."
chunk3 = "Give me a 5-step apple pie recipe."
chunk4 = "Tell me a science-related joke."
chunk5 = "Describe how rainbows form."
chunk6 = "Name 10 popular programming languages."

print("--- Overloaded ---")
print(get_ai_response(long_prompt), "\n")

print("--- Chunked ---")
for c in [chunk1, chunk2, chunk3, chunk4, chunk5, chunk6]:
    print(f"> {c}")
    print(get_ai_response(c), "\n")
```

**Expected:** Overloaded = incomplete/skips parts; chunked = complete per topic.

***

## **📚 What Students Learn From These Solutions**

- How vague or overloaded instructions hurt output quality.
- How to **iteratively refine** prompts to remove ambiguity.
- How to **limit hallucinations** with factual anchoring.
- How to **split large tasks** for better performance.
- How **meta-prompting** can clarify hidden assumptions.

***

If you want, I can next make a **side-by-side automatic comparison logger** so each vague vs refined output is paired visually for training purposes.
Do you want me to prepare that enhanced comparison notebook?

