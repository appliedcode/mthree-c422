# 📝 **Code Implementation Practice: Advanced Prompt Engineering**

## **Problem Statement**

**Objective:**
Using Python in Google Colab and the OpenAI API (`gpt-4o-mini`), implement prompts for four advanced techniques:

1. **Chain-of-Thought Prompting**
2. **Persona-Based Prompting**
3. **Controlling Verbosity, Style \& Tone**
4. **Prompting for Creativity vs. Accuracy**

**Task for each technique:**

- Write Python code that sends your prompt to the model.
- Capture and display the output.
- Compare with a variation that changes the prompting style.
- Add a short note in comments about differences in the outputs.

**Requirements:**

- Use OpenAI Python SDK (`openai` package)
- Store prompts as Python strings
- At least **2 variations** per technique (e.g., with/without CoT, high/low verbosity, etc.)
- Print the outputs clearly with headings

**Deliverables:**

- Python code for all 4 techniques
- Example prompts and outputs
- Observations in comments

***

## 💻 **Solution – Colab‑Ready Code**

> **Note:** Replace `"YOUR_API_KEY"` with your actual OpenAI API key.

```python
# Install and import OpenAI SDK
!pip install openai -q

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

def ask_gpt(prompt, temp=0.7, max_tokens=300):
    """Helper function to send prompt to GPT-4o-mini."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
```


***

### **1️⃣ Chain-of-Thought Prompting**

```python
# CoT OFF
prompt_no_cot = """
A store sells pencils at 3 for $1.50.
How many pencils can you buy with $6?
"""
print("=== Without Chain-of-Thought ===")
print(ask_gpt(prompt_no_cot, temp=0))

# CoT ON
prompt_cot = """
A store sells pencils at 3 for $1.50.
How many pencils can you buy with $6?
Let's think step by step.
"""
print("\n=== With Chain-of-Thought ===")
print(ask_gpt(prompt_cot, temp=0))

# Observation: CoT version should show interim steps like:
# 1. Price per pencil calculation
# 2. Division into $6 budget
```


***

### **2️⃣ Persona-Based Prompting**

```python
# No persona
prompt_no_persona = "Give me 3 must-visit spots in Kyoto, Japan."
print("=== Without Persona ===")
print(ask_gpt(prompt_no_persona))

# With persona
prompt_persona = """
You are a seasoned Japan travel guide.
Create an exciting 3-point list of must-visit spots in Kyoto, Japan,
including cultural context and insider tips.
"""
print("\n=== With Persona ===")
print(ask_gpt(prompt_persona))

# Observation: Persona version should sound warmer, more descriptive,
# with tips only a 'guide' would give.
```


***

### **3️⃣ Controlling Verbosity, Style \& Tone**

```python
# Concise & Formal
prompt_concise_formal = "Explain global warming in exactly 2 formal sentences."
print("=== Concise & Formal ===")
print(ask_gpt(prompt_concise_formal))

# Detailed & Casual
prompt_detailed_casual = "Explain global warming in a detailed but casual, friendly tone."
print("\n=== Detailed & Casual ===")
print(ask_gpt(prompt_detailed_casual))

# Observation: Expect big differences in sentence length, vocabulary,
# and reader engagement.
```


***

### **4️⃣ Prompting for Creativity vs. Accuracy**

```python
# Creative (High temp)
prompt_creative = "Imagine a day in 2080 where cities float above oceans. Describe creatively."
print("=== Creative (Temp=0.9) ===")
print(ask_gpt(prompt_creative, temp=0.9))

# Accurate (Low temp)
prompt_accurate = """
Describe realistic expectations for global urban life in 2080
based on current scientific and economic data.
"""
print("\n=== Accurate (Temp=0.2) ===")
print(ask_gpt(prompt_accurate, temp=0.2))

# Observation:
# High temp = imaginative, novel scenarios
# Low temp = grounded, factual predictions
```


***

## **📊 Expected Outcome Summary Table**

| Technique | Variation 1 | Variation 2 | Key Difference |
| :-- | :-- | :-- | :-- |
| Chain-of-Thought | Direct answer | Step-by-step reasoning | Transparency \& accuracy |
| Persona-Based | Neutral output | Role-specific style | More tone, context, tips |
| Verbosity/Style \& Tone | 2-sentence formal | Detailed casual | Clarity vs engagement |
| Creativity vs Accuracy | Creative, temp=0.9 | Factual, temp=0.2 | Imagination vs facts |


***

This lab **forces students to implement prompt-engineering concepts in code** and actually **compare** how the API responds to changes, building both technical and conceptual skills.

***

