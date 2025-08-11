## 📝 **Advanced Prompt Design – Problem Statement Set**

**Objective:**
Individually explore and practice five advanced prompt design techniques with a conversational AI model (e.g., GPT‑4o‑mini) using Google Colab.
Learn how prompt structuring affects content depth, organization, and context management.

***

### **1️⃣ Interview Pattern**

**Scenario:**
You are creating a magazine article on *Space Tourism in 2045*.

- Design a prompt that simulates a Q\&A format between a journalist and a space tourism CEO.
- Ensure the interviewer asks at least 5 questions covering history, technology, safety, cost, and future plans.
- Compare this structured interview output to an unstructured “write about space tourism” prompt.

***

### **2️⃣ Tree‑of‑Thoughts**

**Scenario:**
You want to choose the best **startup idea** to pitch to investors.

- Prompt the model to generate **3 different business ideas**.
- For each idea, list **2 pros** and **2 cons**.
- Ask the model to reason about each idea and then recommend the best one with justification.
- Compare outcomes when you **do** and **don’t** explicitly request the pros/cons tree format.

***

### **3️⃣ Step‑by‑Step Reasoning \& Decomposition**

**Scenario:**
You are given a problem: “A company produces 500 gadgets. Each gadget is sold for \$40. The production cost per gadget is \$25. Calculate the total revenue, total cost, and total profit.”

- First, try prompting the model directly for the final answer.
- Then, rewrite the prompt to guide the model step‑by‑step through calculations for revenue, cost, and profit.
- Compare clarity, correctness, and transparency.

***

### **4️⃣ Combining Prompts \& Managing Context**

**Scenario:**
You are acting as a **climate change consultant** preparing a quick report for city planners.

- In the **same prompt**, ask the model to:

1. Summarize 3 benefits of urban green roofs.
2. List 3 key challenges to implementation.
3. Suggest 3 evidence‑based solutions.
- Experiment with merging all sub‑tasks into one prompt vs. running them in separate prompts and combining outputs yourself.

***

### **5️⃣ Prompt Chaining \& Multi‑Turn Conversations**

**Scenario:**
Role-play as a business analyst exploring e‑commerce trends.

- **Turn 1:** Ask the model to explain what e‑commerce is and its current global scope.
- **Turn 2:** Using the above answer, ask how AI can improve e‑commerce operations.
- **Turn 3:** Using both previous answers, request a strategy outline for a small online retailer to implement AI for better customer experience.
- Pay attention to how context carries over between turns.

***

## **Deliverables for Each Technique**

- Your crafted prompt(s)
- Model output(s) from Colab
- A short reflection on:
    - What worked well in your prompt
    - One change you would try next time
    - How the advanced technique affected the depth, clarity, or relevance of the AI’s output

***

## **Learning Goals**

By completing this set, you should be able to:

- Structure prompts for richer, more tailored AI responses
- Maintain and control context across multiple turns
- Break complex tasks into manageable, logical stages
- Guide AI reasoning and decision-making using structured techniques

***
