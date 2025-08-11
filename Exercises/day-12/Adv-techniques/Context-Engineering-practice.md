## **Self-Practice Problems – Context Management \& Prompt Optimization**

### **Part A – Context Management**

These tasks will test your ability to pass context across multiple AI calls, handle continuity, and see how truncating context impacts responses.

***

**Problem 1 – Story Continuation Without Losing Details**

- Start a short story (2–3 lines) with the AI.
- In the next 3–4 prompts, ask the AI to **add more events**, **introduce a new character**, and **create a twist ending**.
- Compare two runs:

1. Without passing any prior context.
2. With a conversation log that includes all previous responses.
- **Question:** How different are the coherence and continuity between the two runs?

***

**Problem 2 – FAQ Bot Simulation**

- Choose a topic (e.g., "Python Programming Basics").
- Ask a sequence of **related questions** as if you are building a FAQ chatbot:

1. "What is Python used for?"
2. "What are data types in Python?"
3. "Explain Python lists in one sentence."
- Test answering with and without including **full previous Q\&A** in the prompt.
- **Question:** What information does the AI miss without context?

***

**Problem 3 – Context Truncation Challenge** *(Advanced)*

- Build a loop that stores conversation history but **truncates older turns** if it exceeds a certain number of characters/tokens.
- Simulate a 10-turn conversation and see if truncation breaks continuity.
- **Question:** Which parts of the story/answers start to drop off first?

***

### **Part B – Prompt Optimization**

Students will refine vague prompts into clear, specific instructions and analyze the difference in AI responses.

***

**Problem 4 – Recipe Request Optimization**

- Vague prompt: `"Give me a recipe for pasta"`.
- Optimize it by specifying:
    - Audience (beginner cook)
    - Ingredients to include/exclude
    - Portion size
    - Output format (ingredients list + numbered steps)
- Compare the vague and optimized outputs.
- **Question:** What improvements did you notice?

***

**Problem 5 – Study Guide Creation**

- Choose a subject (Math, History, Physics, etc.).
- First, ask: `"Tell me about this subject"`.
- Then, design a prompt that produces:
    - 5 bullet points summarizing key concepts
    - A suggested daily study plan
    - Easy examples for each concept
- Compare both outputs for usefulness and structure.

***

**Problem 6 – Summarization with Constraints**

- Start with a paragraph from Wikipedia or any online source.
- First prompt: `"Summarize this paragraph"`.
- Optimized prompt: `"Summarize this paragraph in exactly 3 bullet points, each under 12 words, keep only key facts"`.
- Compare results for precision and conciseness.

***

### **Part C – Combined Challenge**

**Problem 7 – Multi-Turn Optimized Interview**

- Simulate a job interview with the AI for a "Machine Learning Engineer" role:

1. Ask an AI-generated job description.
2. Ask it to generate 3 common interview questions.
3. Ask follow-up questions based on its previous answers.
- **Rules:**
    - Always keep context from all previous turns.
    - Optimize every prompt to clearly state the desired format.
- **Question:** How does the AI’s interview flow change without context or with vague prompts?

***

💡 **Tip:** For each problem, students should:

1. Run first with vague or no-context setups.
2. Re-run with optimized prompts and/or full context.
3. Write 2–3 sentences on observed differences.

***

If you want, I can **package these problems into a ready-to-run Colab notebook** where placeholders are left for students to fill in code — so they learn by actually typing rather than just reading.

Do you want me to prepare that interactive **practice notebook version**?

