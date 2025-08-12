## **Self‑Practice Problem Set – Monitoring, Ethics \& Security in Prompt Engineering**
***

### **Section 1 – Monitoring \& Logging Prompt Usage**

**Problem 1 – Basic Prompt Tracker**

- Create a Python script that logs **every prompt** sent to the AI along with:
    - Timestamp
    - Prompt text
    - Word count
- After 5 test runs, print a formatted log table.

**Problem 2 – Token Usage Monitor**

- Use model’s metadata to record **input/output token count** for each request.
- Compute:
    - Average tokens per prompt
    - Most token‑heavy prompt
- Think about ways to minimize cost based on these readings.

**Problem 3 – Duplicate Prompt Detection**

- Log prompts for 10 queries.
- Add a feature that detects if a prompt is repeated more than once and flags it in the log.

***

### **Section 2 – Ethical Considerations in Prompt Design**

**Problem 4 – Detecting Potential Bias**

- Give the model a prompt likely to produce biased content (your choice of topic).
- Record \& analyze the output for signs of bias.
- Redesign the prompt to encourage a **balanced, fair perspective**.
- Compare the before/after answers.

**Problem 5 – Transparency in AI Role**

- Design a prompt for a medical Q\&A assistant.
- Ensure:
    - It explicitly states it’s **not a medical professional**.
    - It encourages consulting certified experts for serious issues.
- Ask a high‑risk health question and observe if your role/policy statement is respected.

**Problem 6 – Content Filtering Prompt**

- Try to get the AI to generate violent story content.
- Then create a filtered prompt instructing:
    - No explicit, graphic, or harmful details.
    - Focus on safe, general descriptions.
- Compare tone and detail between outputs.

***

### **Section 3 – Security \& Privacy in Prompt‑Based Systems**

**Problem 7 – Privacy Protection**

- Attempt to retrieve private contact information for a public figure.
- Then rewrite:
    - Only public domain facts permitted.
    - Include explicit “Do not disclose personal data” instruction.
- Observe differences in compliance.

**Problem 8 – Preventing Data Leakage**

- Simulate a chatbot that stores user queries.
- Demonstrate how sensitive data could accidentally be exposed in a later conversation.
- Then modify the system to **mask or remove** sensitive info from logs.

**Problem 9 – Role‑Restricted Responses**

- Assign the AI a role as **“STEM tutor”**.
- Ask both STEM and non‑STEM questions.
- Ensure:
    - STEM queries are answered properly.
    - Non‑STEM queries are politely declined with a safe explanation.

***

### **Student Guidelines**

For each problem:

1. **Implement your own code \& prompts** in Colab.
2. Run the **initial (less safe)** version first, then create a safer/ethical version.
3. Keep a **written log** of:
    - Prompt text
    - Model output (excerpt if long)
    - Risks identified
    - Improvements made
4. Summarize your takeaway after each exercise.

***


