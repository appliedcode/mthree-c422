<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Generate problem statement based on above topic for student to try on its own

Got it ✅ — you want a **fresh set of self‑practice problem statements** for students to work on **Handling Ambiguity, Troubleshooting, and Prompt Failures** (based on the last Colab lab exercises we created).

Here’s a **practice set** they can attempt independently — each is designed so students can first try a “bad prompt” or “failure scenario,” then fix it by applying techniques like clarification, refinement, splitting tasks, and anchoring facts.

***

## **Self‑Practice Problems – Handling Ambiguity, Troubleshooting \& Prompt Failures**


***

### **Problem 1 – Ambiguous Prompt Clarification**

You are tasked with asking the AI to explain a “process,” but the first try should yield vague or irrelevant results.

- Start with an ambiguous prompt like `"Describe the process."`
- Then refine it step-by-step until the AI produces a detailed explanation matching your intended topic (e.g., *water purification*, *cell division*).
- Write down how the meaning changed with each refinement.

***

### **Problem 2 – Avoiding Hallucinations**

You ask the AI: `"Tell me about the most recent discovery in black hole research."`

- First, run this as-is and note any unverifiable or speculative details.
- Then reframe the prompt to limit the AI to **confirmed facts before a fixed date** and to explicitly say `"I don't know"` if unsure.
- Compare the factual accuracy across runs.

***

### **Problem 3 – Troubleshooting Irrelevant Output**

You prompt the AI: `"Write something interesting about the ocean."`

- If the first result is too generic, try adding constraints such as:
    - audience level (e.g., *10-year-old student*)
    - format (bullet list vs. paragraph)
    - focus area (e.g., *deep-sea creatures only*).
- Document how each added detail changes the output focus.

***

### **Problem 4 – Overloaded Task Decomposition**

Create a single overloaded prompt that asks the AI to:

1. Summarize a historical event
2. Provide causes, effects, significance, and modern impact
3. Suggest three fictional “what-if” scenarios

- Notice if the result is incomplete or jumbled.
- Then split the request into **three separate prompts** and stitch the results together manually.
- Compare clarity and completeness.

***

### **Problem 5 – Using Meta-Prompting to Clarify Requirements**

Ask the AI: `"Create an efficient search algorithm."`

- Before letting it produce an answer, add a **meta-prompt** asking it to first state what “efficient” might mean in this context.
- Based on its interpretation, rewrite your original request with specific complexity targets, language, and constraints.
- Compare the unrefined code vs. the refined one.

***

### **Problem 6 – Prompt Failure Recovery**

Craft a deliberately vague request (e.g., `"Create a game"`) that produces a result not meeting your expectations.

- Diagnose why it failed:
    - Was it too broad?
    - Missing constraints?
    - Wrong assumed format?
- Then iteratively upgrade the prompt until it produces a complete, functional response matching your intent.

***

### **Problem 7 – Length \& Truncation Management**

Give the AI a **very long combined prompt** with multiple unrelated questions.

- Observe if the output is cut off or if some parts are skipped.
- Modify your approach to:
    - Remove less relevant details
    - Ask questions in separate turns while passing minimal context
- Record the changes in answer quality.

***

💡 **Instructions for Students**
For each problem:

1. Run the **vague/ambiguous version** first.
2. Apply troubleshooting/refinement strategies:
    - Clarifying the request
    - Adding constraints \& format requirements
    - Anchoring in verified facts
    - Splitting large tasks into smaller ones
3. Compare results and write **2–3 observations** about output quality improvement.

***

If you want, I can now **package these into a runnable Colab workbook** where the “bad” prompts are pre-filled and students fill in their “fixed” prompts — with cells to run and compare outputs.

Do you want me to prepare that interactive **student practice notebook**?

