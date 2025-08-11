## 🎯 Self-Study Exercises: Exploring Prompt Engineering Patterns

### Exercise 1: Instruction Clarity and Detail

- Write a prompt asking an AI to explain how photosynthesis works.
- Try two versions:
a) Give a brief explanation in 3 sentences.
b) Give a detailed explanation with examples, formatted in bullet points.
- Compare the length, clarity, and completeness of the outputs.

***

### Exercise 2: Few-Shot Learning

- Create a few-shot prompt that shows how to politely decline an invitation in English and French.
- Provide 2 examples (English phrase + French translation).
- Add a new English phrase and ask the model to provide the French translation.
- Experiment by changing the number of examples and observe changes.

***

### Exercise 3: Persona Roleplay

- Write a prompt instructing the model to behave as a friendly travel guide for Tokyo.
- Ask for recommendations on places to visit and food to try.
- Rewrite the prompt to roleplay as a strict tour manager focusing on timing and schedules.
- Note the differences in tone and content.

***

### Exercise 4: Recipe and Alternatives

- Request a recipe for improving daily productivity.
- Ask the model to suggest two alternative approaches, each with advantages and disadvantages.
- Modify the prompt to ask for a creative approach and a scientific approach instead.
- See how the style and content vary.

***

### Exercise 5: Flipped Interaction / Ask-for-Input

- Write an initial prompt that instructs the AI to interview you with 3 questions to understand your favorite hobbies.
- Use the model’s questions to build a personalized weekend plan based on your answers.
- Try making the AI summarize your answers before giving the plan.
- Test how the interaction feels when you answer in multiple turns.

***

### Exercise 6: Menu / Command Syntax

- Define a small command menu using this pattern:

```
ACTION = Translate, Summarize, Explain  
FORMAT = Plain, Bullets, Numbered  
LANGUAGE = English, French, Spanish  

Command: ACTION=Translate; FORMAT=Numbered; LANGUAGE=Spanish; TEXT="Machine learning improves over time."
```

- Run this prompt and observe output.
- Change commands to translate into French, summarize in bullets, and explain in plain English.
- Expand the command with additional options like STYLE = Formal, Casual and test.

***

### Exercise 7: Controlling Output Length and Style

- Ask the model to describe “Artificial Intelligence” in:
a) a tweet (280 characters max)
b) a formal academic abstract (~150 words)
c) a children’s story (simple language)
- Observe how the style and complexity change with length constraints.

***

### Exercise 8: Multi-turn Conversation Simulation

- Simulate a customer support dialogue where the AI asks clarifying questions before providing a solution to a printer issue.
- Write prompt instructions to make the AI confirm your answers before continuing.
- Try chaining multiple prompts and see how memory of conversation context influences coherence.

***

### Reflection Questions for Self-Review

- Which prompts resulted in the most relevant and coherent answers?
- How did instructions and personas influence tone and style?
- Did few-shot examples improve output quality or consistency?
- How effective was the menu/command syntax in structuring the response?
- How did controlling verbosity and formatting shape the clarity of information?
- What challenges did you face in multi-turn dialogue simulation?

***
