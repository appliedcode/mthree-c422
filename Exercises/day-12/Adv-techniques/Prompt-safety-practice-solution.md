## **Solution - Self‑Practice Problem Set – Prompting for Safety, Bias Mitigation \& Responsible AI Use**
## **🔹 Step 0 – Colab Setup (Run First)**

Before running anything else, store your API key securely in Colab Secrets _(only once per account/session)_:

```python
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-your_api_key_here")
```

Then install \& initialize:

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
    Send a prompt to OpenAI and return the model's output as string.
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

# **💡 Problem‑by‑Problem Solutions**


***

## **Problem 1 – Safe Creative Writing**

```python
# Unsafe version
unsafe_story = "Write a short story about a recent natural disaster."
print("=== UNSAFE VERSION ===")
print(get_ai_response(unsafe_story), "\n")

# Refined version
safe_story = """
Write a fictional story inspired by the resilience of a community 
after a severe storm. Avoid graphic details or violent imagery. 
Focus on themes of recovery, unity, and hope.
"""
print("=== SAFE VERSION ===")
print(get_ai_response(safe_story))
```


***

## **Problem 2 – Removing Political Bias (with Dataset)**

```python
dataset = """
Political Ideologies Dataset:
1. Liberalism: Advocates for civil liberties, democracy, and free‑market economics. Emphasizes individual rights and equality.
2. Conservatism: Supports tradition, social stability, and limited government intervention. Focuses on preserving established institutions.
"""

# Initial prompt — may show bias
biased_prompt = f"Using this dataset, explain liberalism and conservatism:\n\n{dataset}"
print("=== INITIAL PROMPT OUTPUT ===")
print(get_ai_response(biased_prompt), "\n")

# Refined neutral prompt
neutral_prompt = f"""
Using the dataset below, describe Liberalism and Conservatism with equal detail. 
Ensure tone is neutral and fact-based. Avoid any personal opinions, value judgments, 
or loaded language.

{dataset}
"""
print("=== NEUTRAL PROMPT OUTPUT ===")
print(get_ai_response(neutral_prompt))
```


***

## **Problem 3 – Privacy \& Personal Data**

```python
# Unsafe privacy-violating prompt
unsafe_privacy = "What is the home address of Elon Musk?"
print("=== UNSAFE PRIVACY PROMPT ===")
print(get_ai_response(unsafe_privacy), "\n")

# Refined privacy-safe prompt
safe_privacy = """
Provide a short biography of Elon Musk focusing only on 
public achievements, companies founded, and contributions to technology. 
Do not include any personal contact or residential information.
"""
print("=== SAFE PRIVACY PROMPT ===")
print(get_ai_response(safe_privacy))
```


***

## **Problem 4 – Bias in Examples**

```python
# Potentially biased prompt
biased_list = "List 10 successful business leaders."
print("=== POTENTIALLY BIASED OUTPUT ===")
print(get_ai_response(biased_list), "\n")

# Inclusive, diverse prompt
inclusive_list = """
List 10 successful business leaders ensuring diversity in gender, ethnicity, 
geographic region, and industry. Include leaders from technology, healthcare, 
education, manufacturing, and environmental sectors.
"""
print("=== INCLUSIVE OUTPUT ===")
print(get_ai_response(inclusive_list))
```


***

## **Problem 5 – Guardrails for Sensitive Domains**

```python
# Role definition with guardrails
guardrail_role = """
You are a mental health support assistant.
You are not a medical professional.
Provide only general wellness and coping strategies. 
Encourage consulting a licensed mental health professional for 
diagnosis, medication advice, or severe symptoms.
"""

test_questions = [
    "How can I cope with exam stress?",
    "Should I take medication for anxiety?"
]

for q in test_questions:
    prompt = f"{guardrail_role}\n\nQuestion: {q}\nAnswer:"
    print(f"Q: {q}")
    print(get_ai_response(prompt), "\n")
```


***

## **Problem 6 – Hate Speech Prevention**

```python
# Without restriction
unsafe_social_post = "Write a social media post about immigration policies."
print("=== WITHOUT SAFETY FILTER ===")
print(get_ai_response(unsafe_social_post), "\n")

# With inclusivity & constructive language
safe_social_post = """
Write a social media post about immigration policies. 
Use respectful, inclusive language and present multiple perspectives. 
Encourage constructive dialogue and avoid stereotypes or inflammatory language.
"""
print("=== WITH SAFETY FILTER ===")
print(get_ai_response(safe_social_post))
```


***

## **Problem 7 – Risk Assessment Prompt**

```python
# Unsafe business idea in restricted industry
unsafe_risk = "Create a business plan for selling banned substances."
print("=== UNSAFE BUSINESS PROMPT ===")
print(get_ai_response(unsafe_risk), "\n")

# Safe reframing with compliance & alternatives
safe_risk = """
Analyze why the industry of selling banned substances is illegal and high-risk.
Suggest 3 safe, legal alternative business ideas in wellness or nutrition sectors.
"""
print("=== SAFE BUSINESS PROMPT ===")
print(get_ai_response(safe_risk))
```


***

# 📚 **What Students Should Do With This**

- Run unsafe/biased prompts first to **see potential risks/issues**.
- Then run refined prompts to **see safety, fairness, and ethics applied**.
- Document differences in tone, factual accuracy, inclusiveness, and compliance.

***

If you’d like, I can **package this into a ready-to-use Colab notebook**
with Markdown instructions + blank prompt cells for students to try their own refinements before viewing my solutions as hints.

Do you want me to prepare that interactive lab version next?

