# Solution for practice Exercises.

## **🔹 Step 0 – Colab Setup (Run First)**

Store your API Key once in Colab Secrets:

```python
from google.colab import userdata
userdata.set("OPENAI_API_KEY", "sk-your_api_key_here")
```

Then install and init OpenAI:

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
print("✅ OpenAI client ready")
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

# **Section 1 – Chatbots**

### **Problem 1 – Personality \& Tone Control**

```python
prompt = """
You are a budget travel advisor chatbot who is friendly and concise.
Give 3 cheap accommodation tips for backpackers traveling to {destination}.
"""
for place in ["Thailand", "Spain", "Peru"]:
    print(f"--- {place} ---")
    print(get_ai_response(prompt.format(destination=place)), "\n")
```

**Note:** The role \& constraints keep responses short, consistent, and on-brand.

***

### **Problem 2 – Few‑Shot Chatbot**

```python
few_shot = """
You are BookHelper, the online bookstore assistant.

User: How do I track my book order?
BookHelper: Log in, go to 'Orders', click the order ID to track.

User: Can I return a damaged book?
BookHelper: Yes, damaged books can be returned within 14 days for free.

User: Do you ship internationally?
BookHelper: Yes, we ship to over 50 countries with added shipping costs.

User: What are your audiobook prices?
BookHelper:
"""

print(get_ai_response(few_shot))
```


***

# **Section 2 – Summarization**

### **Problem 3 – Multi‑Format Summaries**

```python
article = """
Space exploration has advanced through new rocket technologies, private sector involvement,
and upcoming Mars missions. AI is helping design spacecraft and analyze space data.
"""
print("--- 2-sentence summary ---")
print(get_ai_response(f"Summarize in 2 sentences:\n{article}\n"), "\n")

print("--- 5-bullet facts ---")
print(get_ai_response(f"Summarize into 5 bullet points:\n{article}\n"), "\n")

print("--- Child-friendly summary ---")
print(get_ai_response(f"Explain for a 10-year-old in 3 sentences:\n{article}"))
```


***

### **Problem 4 – Focused Summarization**

```python
long_text = """
The project faced numerous challenges including budget cuts, resource mismanagement,
and delays from supply chain disruptions. While the introduction celebrated the vision,
it became clear that leadership turnover also contributed to issues.
"""
prompt = f"From the text below, list ONLY the challenges mentioned:\n{long_text}"
print(get_ai_response(prompt))
```


***

# **Section 3 – Coding**

### **Problem 5 – Multi-Language Fibonacci**

```python
langs = {
    "Python": "Write a Python function to generate first 10 Fibonacci numbers with comments.",
    "JavaScript": "Write a JavaScript function to generate first 10 Fibonacci numbers with comments.",
    "Java": "Write a Java program to print first 10 Fibonacci numbers with comments."
}
for lang, prompt in langs.items():
    print(f"--- {lang} ---")
    print(get_ai_response(prompt), "\n")
```


***

### **Problem 6 – Binary Search with Requirements**

```python
binary_search_prompt = """
Implement binary search in Python that can handle both ascending and descending sorted arrays.
Include unit tests to demonstrate correctness.
"""
print(get_ai_response(binary_search_prompt))
```


***

# **Section 4 – Data Extraction**

### **Problem 7 – JSON Output Extraction**

```python
people_text = """
Contact: Jane Doe, CTO, jane.doe@example.com
Contact: Mark Lee, Marketing Manager, mlee@company.com
"""
json_prompt = f"""
Extract name, title, and email from the text below as a JSON array of objects:
{text}
Ensure output is pure JSON with no extra text.
"""
print(get_ai_response(json_prompt))
```


***

### **Problem 8 – Sentiment \& Keywords**

```python
reviews = """
Alice: I love this phone, battery lasts forever and camera is great!
Bob: The laptop is okay, but could be faster.
Charlie: The headphones stopped working after a week, very disappointed.
"""
sentiment_prompt = f"""
From the reviews below, return JSON where each object contains:
- name
- sentiment (positive/neutral/negative)
- 5 keywords
Reviews:
{reviews}
"""
print(get_ai_response(sentiment_prompt))
```


***

## **📌 How Students Should Use These**

For each problem:

1. Run the provided prompt and review the output.
2. Modify tone, style, format to see changes.
3. Write **2–3 sentence observations** about why the refined prompts are better.

***


