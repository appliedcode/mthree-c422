<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Could you generate the solution code for the same

Got it ✅ — you’re asking for **solution code in Colab** for the *self-practice lab* we just defined, so students can actually run their prompts for **Clarity, Context, Constraints, Organization, Prefixes \& Labels, and Decomposition** and compare outputs.

Here’s a **complete Colab-ready solution notebook section** using `gpt-4o-mini` with example answers for each exercise so they can check their work.

***

## 🧪 Prompt Structure Lab – Solution Code (gpt‑4o‑mini)

### 0️⃣ Setup \& API Key

Make sure your key is stored in Colab secrets:

```python
!pip install --quiet openai
from google.colab import userdata
import os
from openai import OpenAI

api_key = userdata.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("❌ API key not found — set with userdata.set('OPENAI_API_KEY','your-key')")
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```


***

### 1️⃣ Clarity

```python
# Vague prompt
prompt1 = "Tell me about machine learning."
# Clear prompt
prompt2 = "Explain machine learning in 3 simple sentences for a high-school science student."

for p in [prompt1, prompt2]:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}],
        max_tokens=100
    )
    print(f"\nPrompt:\n{p}\n---\n{r.choices[0].message.content.strip()}\n")
```


***

### 2️⃣ Context

```python
# No context
prompt_nc = "Describe climate change."
# With context
prompt_c = (
    "You are an environmental educator preparing a talk for 10-year-old students. "
    "Explain climate change with a relatable example in 3-4 sentences."
)

for p in [prompt_nc, prompt_c]:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}],
        max_tokens=120
    )
    print(f"\nPrompt:\n{p}\n---\n{r.choices[0].message.content.strip()}\n")
```


***

### 3️⃣ Constraints

```python
# No constraints
p1 = "List the benefits of clean energy."
# Exact constraints
p2 = "List exactly 4 benefits of clean energy, each 6 words or fewer, in bullet points."

for p in [p1, p2]:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": p}],
        max_tokens=60
    )
    print(f"\nPrompt:\n{p}\n---\n{r.choices[0].message.content.strip()}\n")
```


***

### 4️⃣ Organization

```python
prompt = """
### Instruction:
Explain how solar panels work.

### Context:
Audience is 8th-grade science students.

### Output Format:
Give the answer in exactly 3 short bullet points.
"""
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=80
)
print(r.choices[0].message.content.strip())
```


***

### 5️⃣ Prefixes \& Labels

```python
prompt = """Explain photosynthesis.

Example 1:
Q: What is photosynthesis?
A: It is how plants make their food from sunlight, water, and air.

Example 2:
Q: Give an example with a tree.
A: A tree uses sunlight to turn water and air into sugars, releasing oxygen.

Now:
Q: Give an example with grass.
A:"""

r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=60
)
print(r.choices[0].message.content.strip())
```


***

### 6️⃣ Decomposition

```python
prompt = (
    "Step 1: Define Artificial Intelligence in one sentence.\n"
    "Step 2: Give two bullet-point examples (one in healthcare, one in education)."
)

r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=120
)
print(r.choices[0].message.content.strip())
```


***

### ✅ How to Use

- Students run each block **twice** — once as given, and then change prompt wording.
- They can compare how changes in **clarity, context, constraints, organization, labels, and decomposition** affect the model’s responses.
- Encourage keeping a “before/after” log to see improvements.

***

If you want, I can now **bundle this into a polished Colab notebook** with markdown instructions, placeholder prompts for students, and separate “Run Solution” buttons — so it’s perfectly classroom-ready.

Do you want me to prepare that?

