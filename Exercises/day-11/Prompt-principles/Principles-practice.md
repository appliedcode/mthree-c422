## 📝 Problem Statement: Prompt Engineering Practice Lab with Diverse Datasets

**Objective:**
Apply and experiment with the **5 principles of prompt engineering** —

1. Give Clear Direction
2. Specify Format
3. Provide Examples
4. Evaluate Quality
5. Divide Labor

**Task:**
Using the OpenAI API in Google Colab with the `gpt-4o-mini` model:

***

### 1. Give Clear Direction

**Dataset:** Article excerpt on renewable energy
**Data snippet:**
"Renewable energy sources such as solar, wind, and hydropower are increasingly cost-effective alternatives to fossil fuels. They reduce greenhouse gas emissions and offer sustainable energy solutions worldwide."

- Start with a vague prompt (e.g., "Write about energy.")
- Rewrite it with clear direction including audience (e.g., teenagers), tone (informative), and length (100 words).
- Compare and analyze the outputs.

***

### 2. Specify Format

**Dataset:** Product descriptions of three smartphones
**Data snippet:**

- Phone A: OLED display, 12MP camera, 4000mAh battery.
- Phone B: AMOLED display, 108MP camera, 4500mAh battery.
- Phone C: LCD display, 48MP camera, 3500mAh battery.
- Ask a comparison question without specifying format.
- Repeat with instructions to produce a markdown table listing features and specs.
- Compare the clarity and usefulness of the outputs.

***

### 3. Provide Examples (Few-Shot)

**Dataset:** Customer reviews for a coffee shop
**Reviews:**

- "Loved the espresso, very rich flavor."
- "Service was slow but the ambiance was nice."
- "The croissant was stale and dry."
- Provide 2-3 labeled examples classifying reviews as Positive, Neutral, or Negative.
- Add a new review for the model to classify.

***

### 4. Evaluate Quality

**Dataset:** Short news article on urban gardening
**Text:**
"Community gardens are flourishing in cities worldwide, providing fresh produce and green spaces for residents. They improve air quality, foster social connections, and promote sustainability."

- Summarize this text initially with a generic prompt.
- Evaluate the output for completeness and key points.
- Refine the prompt to ask for a 3-bullet summary mentioning benefits, social impact, and environmental effects.
- Compare before and after outputs.

***

### 5. Divide Labor (Break Complex Tasks)

**Dataset:** Business project - launching an online bookstore

- Step 1 prompt: Define and describe the target customer base.
- Step 2 prompt: Develop a marketing strategy tailored for the audience.
- Step 3 prompt: Draft a content calendar for social media launch posts.
- Run each prompt sequentially in Colab.
- Combine the outputs into a cohesive overall launch plan.

***

## Deliverables

For each principle:

- Your original prompt(s)
- Model output(s)
- Observations on prompt design impact

***

**Goal:**
Experience how tailoring prompts with clear direction, format, examples, evaluation, and task decomposition improves AI-generated content quality and usability.

***

If you would like, I can prepare this as a **ready-to-use Colab notebook template** with code cells pre-set for the datasets and exercises, so students can focus just on crafting prompts and observing results. Let me know if you want me to do that!

