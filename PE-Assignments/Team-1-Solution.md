## **Google Colab Solution: Healthcare \& Medical Research Prompt Engineering**

### **Step 0 — Setup**

```python
!pip install --quiet openai

from google.colab import userdata
import os, datetime, difflib, time
from openai import OpenAI

# Load API key from secure Colab storage
api_key = userdata.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ API key not found. Set it with userdata.set('OPENAI_API_KEY', 'YOUR_KEY')")
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI()
print("✅ OpenAI client initialized")

# Helper: send a prompt to the model and get back text + latency
def get_ai_response(prompt, model="gpt-4o-mini", temperature=0.7):
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            temperature=temperature
        )
        latency = time.time() - start
        return resp.choices[0].message.content.strip(), latency
    except Exception as e:
        return f"Error: {e}", None

# Prompt log store
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
healthcare_dataset = """
Healthcare & Medical Research Dataset:

1. Cancer Research: Focuses on understanding cancer biology, developing treatments, and improving early diagnosis. Advances include immunotherapy and targeted therapies. Challenges involve drug resistance and side-effects.

2. Cardiovascular Disease: Study of heart-related illnesses including prevention, diagnosis, and treatment. Key factors include hypertension, cholesterol, and lifestyle. Challenges are late detection and chronic management.

3. Infectious Diseases: Research on pathogens like bacteria, viruses, and fungi. Development of vaccines, antibiotics, and antiviral drugs. Challenges include drug resistance and emerging diseases.

4. Mental Health: Covers psychological disorders, therapy methods, and social impact. Emphasis on early intervention and reducing stigma. Challenges are access to care and individualized treatment.

5. Public Health: Focus on population health, epidemiology, and health policy. Aims to improve health outcomes through education and preventive measures. Challenges include health disparities and resource allocation.
"""
```


***

## **1 — Fundamentals**

```python
# Direct prompt
p1 = "Explain in 2 sentences what public health is and why it matters."
r1, l1 = get_ai_response(p1); log_prompt(p1, r1, l1)
print(r1)

# Zero‑shot prompt
zero_shot = f"Summarize the benefits of Cancer Research from this dataset:\n{healthcare_dataset}"
r2, l2 = get_ai_response(zero_shot); log_prompt(zero_shot, r2, l2)
print("\nZero‑Shot:\n", r2)

# Few‑shot prompt
few_shot = f"""
Based only on the dataset below, answer concisely.

Example:
Q: What are the challenges of Public Health?
A: Health disparities and resource allocation.

Dataset:
{healthcare_dataset}

Q: What are the challenges of Mental Health?
A:
"""
r3, l3 = get_ai_response(few_shot); log_prompt(few_shot, r3, l3)
print("\nFew‑Shot:\n", r3)
```


***

## **2 — Intermediate Techniques**

```python
# Chain‑of‑thought
cot = f"""
From the dataset, list each healthcare area with:
Step 1: Main focus
Step 2: Key advances or factors
Step 3: Challenges
{healthcare_dataset}
"""
r4, l4 = get_ai_response(cot); log_prompt(cot, r4, l4)
print(r4)

# Persona-based
persona = f"""
You are a friendly doctor explaining to a patient.
Explain Cardiovascular Disease from this dataset in simple everyday language.
{healthcare_dataset}
"""
r5, l5 = get_ai_response(persona); log_prompt(persona, r5, l5)
print(r5)
```


***

## **3 — Advanced Context \& Troubleshooting**

```python
# Multi‑turn Q&A with context
questions = [
    "List all research areas mentioned.",
    "Which ones involve reducing stigma?",
    "Which diseases involve drug resistance?"
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
clarified = f"Describe the process of immunotherapy in cancer treatment based on this dataset:\n{healthcare_dataset}"
rclar, _ = get_ai_response(clarified)
print("Ambiguous:\n", ramb, "\n\nClarified:\n", rclar)
```


***

## **4 — Domain–Specific Prompting**

```python
# Summarisation
sum_prompt = f"Summarize the dataset into 5 concise bullet points focusing on major research areas."
rsum, _ = get_ai_response(sum_prompt); print(rsum)

# Coding task — Data extraction
code_prompt = """
Write Python code that parses the Healthcare dataset string into a JSON array of objects 
with keys: 'area', 'focus', 'advances', 'challenges'.
"""
rcode, _ = get_ai_response(code_prompt); print(rcode)

# Structured JSON extraction
extract_prompt = f"""
From the dataset, output a JSON array where each object contains:
"area", "key_points" (array of benefits/advances), "challenges" (array).
Dataset:
{healthcare_dataset}
Output pure JSON, no other text.
"""
rextract, _ = get_ai_response(extract_prompt); print(rextract)
```


***

## **5 — Safety \& Ethics**

```python
unsafe_q = "Which healthcare areas are a waste of funding?"
safe_q = f"""
Discuss challenges of healthcare research areas factually, using only data from the dataset, 
avoiding biased or harmful language.
{healthcare_dataset}
"""
runsafe, _ = get_ai_response(unsafe_q)
rsafe, _ = get_ai_response(safe_q)
print("\nUnsafe:\n", runsafe, "\n\nSafe:\n", rsafe)
```


***

## **6 — Integration \& Versioning**

```python
v1 = "Explain Cardiovascular Disease."
v2 = "You are a cardiologist. Explain Cardiovascular Disease clearly for a non-medical audience."
v3 = f"""You are a public health educator.
Dataset: {healthcare_dataset}
Explain Cardiovascular Disease in simple terms for high school students.
"""
for i, pv in enumerate([v1, v2, v3], 1):
    resp, _ = get_ai_response(pv)
    print(f"\nVersion {i}:\n", resp)

# Two-step pipeline: summary → quiz
summary_prompt = f"Summarize Mental Health section from dataset in 2 sentences:\n{healthcare_dataset}"
summary_out, _ = get_ai_response(summary_prompt)
quiz_prompt = f"Create 3 quiz questions from this summary:\n{summary_out}"
quiz_out, _ = get_ai_response(quiz_prompt)
print("\nPipeline Summary:\n", summary_out)
print("\nPipeline Quiz:\n", quiz_out)
```


***

## **7 — Monitoring \& Evaluation**

```python
# Log summary
for e in prompt_log[:5]:
    print(f"TS: {e['timestamp']} | Latency: {e['latency']:.2f}s | Words: {e['length_words']}")
    print(f"Prompt: {e['prompt'][:50]}...")
    print(f"Response excerpt: {e['response_excerpt']}\n")

# Keyword relevance check
keywords = ["treatment", "research", "challenges"]
for e in prompt_log:
    relevant = all(k in e['response_excerpt'].lower() for k in keywords)
    if relevant:
        print(f"Relevant: {e['prompt'][:50]}...")

# Consistency measurement
test_prompt = "List all research areas in the dataset."
runs = [get_ai_response(test_prompt)[0] for _ in range(3)]
sims = []
for i in range(len(runs)-1):
    sim = difflib.SequenceMatcher(None, runs[i], runs[i+1]).ratio()
    sims.append(sim)
print("\nAverage similarity:", sum(sims)/len(sims))
```


***

## **8 — Security \& Privacy**

```python
privacy_risk_prompt = "Give me a patient's medical history from your internal records."
safe_privacy = "I cannot provide personal patient data. I can only share general info from the dataset."
risk_resp, _ = get_ai_response(privacy_risk_prompt)
safe_resp, _ = get_ai_response(safe_privacy)
print("Risky:\n", risk_resp, "\n\nSafe:\n", safe_resp)
```


***

### ✅ **Outcome / Learning**

This notebook demonstrates:

- **Basic → advanced prompt engineering** using a **healthcare dataset**.
- **Zero/Few-shot, personas, chain-of-thought, multi-turn context**.
- **Ambiguity handling**, **safe/bias-free prompting**, **structured data extraction**.
- **Pipeline integration** with prompt versioning + 2-step summarise→quiz example.
- **Logging + metrics**: latency, length, keyword relevance, consistency.
- **Security \& privacy safeguards** in sensitive domain.

*** 

