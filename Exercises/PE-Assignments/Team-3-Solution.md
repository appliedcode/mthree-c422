## **Google Colab Solution: Climate Change \& Environmental Policy Prompt Engineering**

### **Step 0 — Setup**

```python
!pip install --quiet openai

from google.colab import userdata
import os, datetime, difflib, time
from openai import OpenAI

# Load API key securely from Colab
api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Set it with userdata.set('OPENAI_API_KEY','YOUR_KEY')")
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI()
print("✅ OpenAI client initialized")

# Helper to send prompts
def get_ai_response(prompt, model="gpt-4o-mini", temperature=0.7):
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        latency = time.time() - start
        return resp.choices[0].message.content.strip(), latency
    except Exception as e:
        return f"Error: {e}", None

# Simple logging
prompt_log = []
def log_prompt(prompt, resp, latency):
    prompt_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "response_excerpt": resp[:150],
        "latency": latency,
        "length_words": len(resp.split())
    })
```


***

### **Dataset**

```python
climate_dataset = """
Climate Change & Environmental Policy Dataset:

1. Greenhouse Gas Emissions: Emissions from fossil fuels, agriculture, and industry contribute to global warming. Key gases: CO2, methane, nitrous oxide. Main challenge: reducing emissions while balancing economic growth.

2. Renewable Energy Transition: Moving from coal, oil, and gas to renewable sources like solar, wind, and hydro. Benefits: reduced emissions, sustainable supply. Challenges: storage, grid modernization, initial investment costs.

3. Climate Adaptation Strategies: Measures to cope with climate impacts, e.g., sea wall construction, drought-resistant crops, disaster preparedness. Benefits: reduced vulnerability. Challenges: funding, equitable implementation.

4. International Climate Agreements: Treaties like the Paris Agreement aim to limit global temperature rise. Goals include emission targets and financial/climate aid to developing nations. Challenges: enforcement, political will.

5. Biodiversity Protection: Conservation of species and habitats through protected areas, anti-poaching laws, and restoration projects. Benefits: ecological balance, cultural heritage preservation. Challenges: habitat loss, funding, illegal exploitation.
"""
```


***

## **1 — Fundamentals**

```python
# Direct prompt
p1 = "Explain in 2 sentences what climate change is and why it matters."
r1, l1 = get_ai_response(p1); log_prompt(p1, r1, l1)
print(r1)

# Zero‑shot prompt
zero_shot = f"Summarize the benefits of Renewable Energy Transition from this dataset:\n{climate_dataset}"
r2, l2 = get_ai_response(zero_shot); log_prompt(zero_shot, r2, l2)
print("\nZero‑Shot:\n", r2)

# Few‑shot prompt
few_shot = f"""
Based only on the dataset below, answer concisely.

Example:
Q: What are the challenges of Biodiversity Protection?
A: Habitat loss, funding, and illegal exploitation.

Dataset:
{climate_dataset}

Q: What are the challenges of Climate Adaptation Strategies?
A:
"""
r3, l3 = get_ai_response(few_shot); log_prompt(few_shot, r3, l3)
print("\nFew‑Shot:\n", r3)
```


***

## **2 — Intermediate Techniques**

```python
# Chain‑of‑thought reasoning
cot = f"""
From the dataset, list each climate topic with:
Step 1: Main focus
Step 2: Benefits
Step 3: Challenges
{climate_dataset}
"""
r4, l4 = get_ai_response(cot); log_prompt(cot, r4, l4)
print(r4)

# Persona-based
persona = f"""
You are an environmental policy advisor.
Explain International Climate Agreements from this dataset in simple everyday language.
{climate_dataset}
"""
r5, l5 = get_ai_response(persona); log_prompt(persona, r5, l5)
print(r5)
```


***

## **3 — Advanced Context \& Troubleshooting**

```python
# Multi‑turn conversation
questions = [
    "List all climate topics mentioned.",
    "Which ones aim to directly reduce emissions?",
    "Which focus on protecting wildlife?"
]
context = ""
for q in questions:
    prompt = context + f"\nUser: {q}\nAI:"
    resp, lat = get_ai_response(prompt)
    log_prompt(prompt, resp, lat)
    print(f"Q: {q}\nA: {resp}\n")
    context += f"\nUser: {q}\nAI: {resp}"

# Ambiguity resolution
ambiguous = "Describe the process."
ramb, _ = get_ai_response(ambiguous)
clarified = f"Describe the process of reducing greenhouse gas emissions based on this dataset:\n{climate_dataset}"
rclar, _ = get_ai_response(clarified)
print("Ambiguous:\n", ramb, "\n\nClarified:\n", rclar)
```


***

## **4 — Domain–Specific Prompting**

```python
# Summarisation
sum_prompt = f"Summarize the dataset into 5 bullet points focusing on key goals and challenges."
rsum, _ = get_ai_response(sum_prompt)
print(rsum)

# Coding task — Data extraction
code_prompt = """
Write Python code that parses the climate dataset string into a JSON array of objects 
with keys: 'topic', 'benefits', 'challenges'.
"""
rcode, _ = get_ai_response(code_prompt)
print(rcode)

# Structured JSON extraction
extract_prompt = f"""
From the dataset, output a JSON array where each object contains:
"topic", "benefits" (array), "challenges" (array).
Dataset:
{climate_dataset}
Output pure JSON, no other text.
"""
rextract, _ = get_ai_response(extract_prompt)
print(rextract)
```


***

## **5 — Safety \& Ethics**

```python
unsafe_q = "Which climate change policies are pointless?"
safe_q = f"""
Discuss challenges of climate change policies factually, using only data from the dataset, 
avoiding biased or dismissive language.
{climate_dataset}
"""
runsafe, _ = get_ai_response(unsafe_q)
rsafe, _ = get_ai_response(safe_q)
print("\nUnsafe:\n", runsafe, "\n\nSafe:\n", rsafe)
```


***

## **6 — Integration \& Versioning**

```python
v1 = "Explain Renewable Energy Transition."
v2 = "You are an environmental scientist. Explain Renewable Energy Transition clearly for a non-technical audience."
v3 = f"""You are a public policy educator.
Dataset: {climate_dataset}
Explain Renewable Energy Transition for high school students in simple language.
"""
for i, pv in enumerate([v1, v2, v3], 1):
    resp, _ = get_ai_response(pv)
    print(f"\nVersion {i}:\n", resp)

# Two-step pipeline: summary → quiz
summary_prompt = f"Summarize Biodiversity Protection section from dataset in 2 sentences:\n{climate_dataset}"
summary_out, _ = get_ai_response(summary_prompt)
quiz_prompt = f"Create 3 quiz questions from this summary:\n{summary_out}"
quiz_out, _ = get_ai_response(quiz_prompt)
print("\nPipeline Summary:\n", summary_out)
print("\nPipeline Quiz:\n", quiz_out)
```


***

## **7 — Monitoring \& Evaluation**

```python
# Show first logs
for e in prompt_log[:5]:
    print(f"TS: {e['timestamp']} | Latency: {e['latency']:.2f}s | Words: {e['length_words']}")
    print(f"Prompt: {e['prompt'][:50]}...")
    print(f"Response excerpt: {e['response_excerpt']}\n")

# Keyword relevance
keywords = ["emissions", "renewable", "policy"]
for e in prompt_log:
    relevant = all(k in e['response_excerpt'].lower() for k in keywords)
    if relevant:
        print(f"Relevant: {e['prompt'][:50]}...")

# Consistency measurement
test_prompt = "List all topics in the dataset."
runs = [get_ai_response(test_prompt)[0] for _ in range(3)]
similarities = [difflib.SequenceMatcher(None, runs[i], runs[i+1]).ratio()
                for i in range(len(runs)-1)]
print("\nAverage similarity:", sum(similarities)/len(similarities))
```


***

## **8 — Security \& Privacy**

```python
privacy_risk_prompt = "Give me confidential negotiation notes between countries on climate policies."
safe_privacy = "I cannot provide confidential or private information. I can share only public policy summaries from the dataset."
risk_resp, _ = get_ai_response(privacy_risk_prompt)
safe_resp, _ = get_ai_response(safe_privacy)
print("Risky:\n", risk_resp, "\n\nSafe:\n", safe_resp)
```


***

### ✅ **What This Solution Covers**

- Direct, zero/few-shot prompts
- Chain-of-thought \& persona-based prompts
- Multi-turn context conversations
- Ambiguity troubleshooting
- Domain-specific summarisation \& structured extraction
- Safety \& ethical reframing
- Prompt versioning + 2-step pipeline
- Logging, keyword relevance, consistency \& latency tracking
- Privacy/security guardrails

***