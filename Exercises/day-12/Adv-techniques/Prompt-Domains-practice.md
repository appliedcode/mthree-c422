## **Self‑Practice Problem Statements – Prompt Engineering for Specific Domains**


***

### **Section 1 – Chatbots**

**Problem 1 – Personality \& Tone Control**

- Create a chatbot that acts as a **travel advisor** for budget backpackers.
- Requirements:
    - Friendly tone
    - Give concise travel tips
    - Offer suggestions for cheap accommodations in any given country
- Test it by asking about **3 different destinations** and see if the tone remains consistent.

**Problem 2 – Few‑Shot Chatbot Training**

- Build a customer support chatbot for an **online bookstore** using **few-shot examples**.
- Include at least **3 Q\&A examples** before the real question.
- Test it with a query not present in your examples to see if it responds in the correct style.

***

### **Section 2 – Summarization**

**Problem 3 – Multi-Format Summaries**

- Take a news article or Wikipedia entry.
- First, create:

1. A **2-sentence concise summary**.
2. A **5-bullet key facts summary**.
3. A **child-friendly 3-sentence summary**.
- Observe how the same source produces different outputs based on prompt instructions.

**Problem 4 – Focused Summarization**

- Choose a long technical blog post.
- Produce a summary that:
    - Omits introduction and unrelated details
    - Focuses **only** on "challenges" discussed in the article.

***

### **Section 3 – Coding**

**Problem 5 – Multi-Language Code Generation**

- Prompt the AI to create a **Fibonacci sequence generator**:
    - In **Python**
    - In **JavaScript**
    - In **Java**
- For each, instruct the AI to **add code comments** and **a sample run**.

**Problem 6 – Algorithm with Performance Requirement**

- Write a prompt that asks the AI to implement **binary search**:
    - In Python
    - Must handle both sorted ascending and descending arrays
    - Include **unit test code**.

***

### **Section 4 – Data Extraction**

**Problem 7 – JSON Output Extraction**

- Give the AI a paragraph describing multiple people with names, job titles, and emails.
- Prompt it to return a **JSON list of objects** with `name`, `title`, and `email` keys.
- Ensure the output is **pure JSON** with no extra text.

**Problem 8 – Sentiment \& Keyword Extraction**

- Take a set of **3 user product reviews** (mix of positive, neutral, negative).
- Prompt the AI to return:
    - Reviewer name
    - Sentiment (positive/neutral/negative)
    - 5 keywords describing the review content
- Output format: **CSV or JSON**.

***

### **Student Instructions**

For each problem:

1. **Start with a baseline prompt** and run it.
2. If the result is incomplete or off-target, **refine** by:
    - Adding role instructions
    - Specifying output format
    - Giving examples (few-shot prompting)
3. Compare initial and refined outputs — write **2–3 sentence observations** for each.

***

