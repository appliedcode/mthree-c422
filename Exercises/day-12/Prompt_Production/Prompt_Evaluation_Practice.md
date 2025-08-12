## **Self‑Practice Problem Set – Prompt Integration \& Performance Evaluation**

***

### **Section 1 – Integrating Prompts in Applications and Pipelines**

**Problem 1 – Multi‑Stage Prompt Pipeline**

- Design a two‑step pipeline where:

1. The first prompt summarizes an article.
2. The second prompt takes that summary and generates 3 quiz questions.
- Implement a function that automatically feeds the output of step 1 as the input to step 2.
- Test with at least two different articles to ensure reusability.

***

**Problem 2 – Role‑Switching in Prompt Pipelines**

- Create a small application where the AI can operate in two distinct roles:

1. As a **technical explainer** (breaking a topic into bullet points).
2. As a **motivational mentor** (giving encouragement on the same topic).
- Implement a role selector in code so the same base question can be processed in either role.

***

**Problem 3 – Prompt Version Control Simulation**

- Store three versions of a prompt for a travel chatbot:
    - Version 1: Minimal instructions
    - Version 2: More detailed instructions
    - Version 3: Includes examples (few‑shot)
- Write code to allow “switching” versions dynamically via a variable or function parameter.
- Compare outputs for the same test queries.

***

### **Section 2 – Prompt Performance Measurement and Evaluation**

**Problem 4 – Keyword‑Based Scoring**

- Choose a test prompt (e.g., “Explain how photosynthesis works”).
- Define a set of 5 keywords you expect in the response.
- Write a function to:
    - Run the prompt.
    - Score the output based on how many expected keywords appear.
- Display a “keyword match score” for each run.

***

**Problem 5 – Response Length Evaluation**

- For a given prompt, run it at least 3 times.
- Record the **word count** for each output.
- Calculate:
    - Average length.
    - Minimum and maximum length.
- Reflect on whether differences in length affect clarity.

***

**Problem 6 – Consistency Measurement**

- Run the same prompt 5 times with the same parameters.
- Compare each output’s similarity using a text‑matching approach (e.g., `difflib.SequenceMatcher`).
- Calculate an average **consistency score**.
- Decide if the current prompt is “stable” or needs refinement.

***

**Problem 7 – Simulated User Feedback Loop**

- Create a simple rating system where after each AI response, a “user” assigns a rating from 1–5 (manually or randomly).
- Store prompt, response excerpt, and rating in a log.
- At the end, output the **average rating per prompt version** to identify better‑performing prompts.

***

**Problem 8 – Latency Tracking for Prompt Performance**

- Time how long the API takes to return a response for each prompt call.
- Record:
    - Prompt text
    - Response time
    - Word count of output
- Plot or print the trade‑off between latency and length, noting any patterns.

***

### **📌 Instructions for Students**

For each problem:

1. Implement the code in Google Colab.
2. Try with at least **two different prompts or datasets**.
3. For performance evaluation tasks, **capture metrics** (keywords, similarity, latency) and store results in a dictionary or DataFrame.
4. At the end, write a short reflection on:
    - What prompt style worked best?
    - Which metrics were most useful for evaluation?
    - What trade‑offs you noticed between detail, speed, and consistency.

***

