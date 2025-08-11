# Solution
**⚠️ Prerequisites:**

- Make sure you've set your OpenAI API key in Colab using:

```python
from google.colab import userdata
userdata.set('OPENAI_API_KEY', 'your-api-key-here')
```

- Install and import OpenAI's client:

```python
!pip install --quiet openai
from openai import OpenAI
import os
from google.colab import userdata

api_key = userdata.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("❌ API key not found.")
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```


***

## 1️⃣ Instruction Clarity and Detail

**a. Brief Explanation**

```python
prompt = "Explain how photosynthesis works in three sentences."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

**b. Detailed, Bullets**

```python
prompt = (
    "Explain how photosynthesis works in detail, using bullet points with specific steps and examples."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150
)
print(response.choices[0].message.content)
```


***

## 2️⃣ Few-Shot Learning

```python
few_shot = """
English: Thank you for your help!
French: Merci pour votre aide !

English: I cannot attend the meeting.
French: Je ne peux pas assister à la réunion.

English: Please call me tomorrow.
French:
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot}],
    max_tokens=50
)
print(response.choices[0].message.content)
```


***

## 3️⃣ Persona Roleplay

**a. Friendly Travel Guide**

```python
prompt = (
    "You are a friendly travel guide for Tokyo. Recommend places to visit and food to try to a first-time visitor."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=120
)
print(response.choices[0].message.content)
```

**b. Strict Tour Manager**

```python
prompt = (
    "You are a strict Tokyo tour manager. Plan a tightly scheduled day with timings and specific stops."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=120
)
print(response.choices[0].message.content)
```


***

## 4️⃣ Recipe and Alternatives

```python
prompt = (
    "Share a recipe for improving daily productivity. Then provide two alternative approaches, each with its advantages and disadvantages."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)
print(response.choices[0].message.content)
```


***

## 5️⃣ Flipped Interaction / Ask-for-Input

```python
prompt = (
    "You are an interviewer. Ask me three questions to understand my favorite hobbies. Then, based on my answers, create a personalized weekend plan and summarize my hobbies first."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=220
)
print(response.choices[0].message.content)
```

*(Let the model ask questions, then you reply in another cell; repeat the process to complete the multi-turn interaction.)*

***

## 6️⃣ Menu / Command Syntax

```python
menu_prompt = """
ACTION = Translate; FORMAT = Numbered; LANGUAGE = Spanish; TEXT = "Machine learning improves over time."
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": menu_prompt}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

*(Try different commands as further input, e.g., Summarize in Bullets, Explain in English, etc.)*

***

## 7️⃣ Output Length and Style

**a. Tweet**

```python
prompt = "Describe 'Artificial Intelligence' in a tweet, maximum 280 characters."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**b. Academic Abstract**

```python
prompt = "Write an academic-style abstract (~150 words) explaining Artificial Intelligence."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200
)
print(response.choices[0].message.content)
```

**c. Children’s Story**

```python
prompt = (
    "Explain Artificial Intelligence as a simple story for children aged 7, using simple language."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=120
)
print(response.choices[0].message.content)
```


***

## 8️⃣ Multi-turn Conversation Simulation

```python
prompt = (
    "Simulate a customer support conversation for a broken printer. Ask me clarifying questions before offering a solution, and confirm my answers before proceeding."
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=180
)
print(response.choices[0].message.content)
```

*(Continue the dialogue in new cells by entering user answers and resubmitting, keeping the message history.)*

***

**Tip:**
For multi-turn situations (like Exercises 5 and 8), add each new question/answer as a new dict in `messages` to preserve context, i.e.:

```python
messages = [
    {"role": "user", "content": initial_prompt},
    {"role": "assistant", "content": "<model's first question>"},
    {"role": "user", "content": "<your answer>"},
    # and so on
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=80
)
```


***