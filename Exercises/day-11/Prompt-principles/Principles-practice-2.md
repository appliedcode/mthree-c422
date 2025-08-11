## 📝 Problem Statement Set \#2: Prompt Engineering Principles Lab

**Objective:**
Further practice and reinforce mastery of the 5 principles of prompt engineering:

1. Give Clear Direction
2. Specify Format
3. Provide Examples
4. Evaluate Quality
5. Divide Labor

For each, use the given dataset, follow the instructions for prompt design and comparison, and reflect on the impact.

***

### 1. Give Clear Direction

**Dataset:**
Short bio:
"Marie Curie was a physicist and chemist who discovered two new elements and was the first woman to win a Nobel Prize."

- Write a general prompt (e.g., "Tell me about Marie Curie.").
- Rewrite your prompt to clarify the intended audience (e.g., third graders), style (excited/storytelling), and target length (under 50 words).
- Compare and analyze the outputs.

***

### 2. Specify Format

**Dataset:**
Vacation destination info:

- Tokyo: Modern city, cherry blossoms, sushi, bullet trains
- Paris: Art museums, Eiffel Tower, pastries, romantic walks
- Cape Town: Mountains, beaches, wildlife, multicultural
- First, ask for a list of the best features for each city, without format guidance.
- Then, request the same information in a JSON array, where each city is an object with "name" and "features" fields.
- Which output is more convenient to use programmatically/in a web app?

***

### 3. Provide Examples (Few-Shot)

**Dataset:**
Social media post comments:

- "Had a great time at the concert!"
- "The app keeps crashing every time I use it."
- "Thanks for the quick delivery!"
- Give 2–3 examples labeling comments as 'Praise', 'Complaint', or 'Thank You'.
- Add a new, unlabeled comment: "My package arrived damaged." and ask the model to classify it.

***

### 4. Evaluate Quality

**Dataset:**
Job advertisement:
"We are seeking a software developer with experience in Python, JavaScript, and cloud platforms. Responsibilities include building web applications, collaborating with designers, and deploying solutions to cloud infrastructure."

- Start with a simple prompt: "Summarize the job ad."
- Evaluate if the summary includes all key requirements.
- Refine your prompt: "Summarize the job ad in 2 sentences, including required skills and main responsibilities."
- Compare how much better the refined summary fits your needs.

***

### 5. Divide Labor (Break Complex Tasks)

**Dataset:**
Event planning for a university science fair:

- Step 1: Identify the types of exhibits and activities suitable for middle school visitors.
- Step 2: Propose a timeline for the event, from setup to conclusion.
- Step 3: Draft a list of safety instructions for participants and guests.
- Write and run a separate prompt for each task.
- Combine the results into a structured event proposal.

***

## Deliverables

For every principle above:

- Your original prompt(s)
- Model output(s)
- Your observations about the impact of your prompt choices

***

## Learning Goal

Deepen understanding of how changing clarity, format, examples, evaluation, and decomposition in your prompts affects the usefulness and quality of AI output.

***

**Tip:**
Try adding your own creative twist by changing the audience (e.g., 8-year-old, business executive) or output style (e.g., poem, checklist, graph description) for even more practice!

***


