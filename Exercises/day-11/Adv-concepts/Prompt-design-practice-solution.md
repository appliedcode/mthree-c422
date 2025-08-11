## 📓 **Solution Code: Advanced Prompt Design Lab**

> **Note:** Replace `"YOUR_API_KEY"` with your real OpenAI API Key before running in Colab.

```python
# ==== Setup ====
!pip install openai -q

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

def ask_gpt(prompt, temperature=0.7, max_tokens=500):
    """Send prompt to GPT-4o-mini and return model's reply text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
```


***

### **1️⃣ Interview Pattern**

```python
prompt_interview = """
Simulate an interview between a journalist and the CEO of a space tourism company in the year 2045.
Include at least 5 Q&A pairs covering:
- History of the company
- Spacecraft technology
- Safety measures
- Cost to customers
- Plans for the next decade
Format clearly as:
Q: ...
A: ...
"""

print("=== Interview Pattern Output ===\n")
print(ask_gpt(prompt_interview, temperature=0.6))
```


***

### **2️⃣ Tree‑of‑Thoughts Prompting**

```python
prompt_tree = """
I want to choose the best startup idea to pitch.
Generate 3 different business ideas.
For each idea, list:
- 2 pros
- 2 cons
After listing, analyze all 3 ideas and choose the best with reasons.
"""

print("=== Tree-of-Thoughts Output ===\n")
print(ask_gpt(prompt_tree, temperature=0.7))
```


***

### **3️⃣ Step‑by‑Step Reasoning \& Decomposition**

```python
# Direct Prompt
prompt_direct = """
A company produces 500 gadgets. Each sells for $40.
Production cost per gadget is $25.
Calculate total revenue, total cost, and total profit.
"""
print("=== Direct Calculation ===\n")
print(ask_gpt(prompt_direct, temperature=0))

# Step-by-Step Prompt
prompt_steps = """
A company produces 500 gadgets. Each sells for $40.
Production cost per gadget is $25.
Let's calculate step-by-step:
1. Total revenue
2. Total cost
3. Total profit
Show the working for each step.
"""
print("\n=== Step-by-Step Calculation ===\n")
print(ask_gpt(prompt_steps, temperature=0))
```


***

### **4️⃣ Combining Prompts \& Managing Context**

```python
prompt_combined = """
You are a climate change consultant.

Part 1: Summarize 3 benefits of urban green roofs.
Part 2: List 3 main challenges to implementing them.
Part 3: Suggest 3 evidence-based solutions to these challenges.

Label each part clearly in your answer.
"""

print("=== Combined Prompt Output ===\n")
print(ask_gpt(prompt_combined, temperature=0.7))
```


***

### **5️⃣ Prompt Chaining \& Multi‑Turn Conversations**

```python
# Turn 1: Basic explanation
turn1 = "Explain what e-commerce is and its current global scope."
res1 = ask_gpt(turn1)
print("=== Turn 1 ===\n", res1)

# Turn 2: Build on previous answer
turn2 = f"Based on your previous explanation: {res1} Explain how AI can improve e-commerce operations."
res2 = ask_gpt(turn2)
print("\n=== Turn 2 ===\n", res2)

# Turn 3: Use all context for strategy
turn3 = f"""
Using the previous answers:
1. {res1}
2. {res2}

Outline a practical AI implementation strategy for a small online retailer to improve customer experience.
"""
res3 = ask_gpt(turn3)
print("\n=== Turn 3 ===\n", res3)
```


***

## 📌 **How to Use in Colab**

1. Copy all the above cells into a Colab notebook.
2. Run the **Setup** cell first (install + API key).
3. Run each section to see an example of the advanced technique.
4. Compare variations in prompt style and note differences in:
    - Structure
    - Depth of reasoning
    - Context handling between turns

***

