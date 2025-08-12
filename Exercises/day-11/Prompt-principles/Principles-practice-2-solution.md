# 💡 **Solutions: Prompt Engineering Principles – Practice Set \#2**


***

## **1. Give Clear Direction**

**Dataset:** Marie Curie bio

**Vague Prompt:**

```
Tell me about Marie Curie.
```

**Sample Output:**
> Marie Curie was a physicist and chemist known for her research on radioactivity. She discovered polonium and radium and won two Nobel Prizes.

**Improved Prompt (audience + tone + length):**

```
Explain who Marie Curie was to a group of 8-year-old students in a fun, storytelling style, in under 50 words.
```

**Sample Output:**
> Marie Curie was a brilliant scientist who discovered hidden treasures in nature—two new elements! She became the first woman to win a Nobel Prize, proving curiosity can change the world.

**Observation:** The improved prompt adjusts tone and detail to suit young learners, staying brief and engaging.

***

## **2. Specify Format**

**Dataset:** Vacation destination info

**No format specified Prompt:**

```
Describe the key attractions of Tokyo, Paris, and Cape Town.
```

**Sample Output:**
> Tokyo has cherry blossoms, modern tech, sushi, and bullet trains. Paris offers the Eiffel Tower, art museums, pastries, and romantic walks. Cape Town features mountains, beaches, wildlife, and a multicultural city vibe.

**Format specified Prompt (JSON):**

```
List the best features of Tokyo, Paris, and Cape Town in JSON format. Each city should be an object with "name" and "features" as an array of strings.
```

**Sample Output:**

```json
[
  {
    "name": "Tokyo",
    "features": ["Modern city", "Cherry blossoms", "Sushi", "Bullet trains"]
  },
  {
    "name": "Paris",
    "features": ["Art museums", "Eiffel Tower", "Pastries", "Romantic walks"]
  },
  {
    "name": "Cape Town",
    "features": ["Mountains", "Beaches", "Wildlife", "Multicultural"]
  }
]
```

**Observation:** JSON makes data ready for programmatic use, unlike plain paragraph form.

***

## **3. Provide Examples (Few‑Shot Prompting)**

**Dataset:** Social media comments

**Prompt:**

```
Label each comment as 'Praise', 'Complaint', or 'Thank You'.

Comment: "Had a great time at the concert!"
Label: Praise

Comment: "The app keeps crashing every time I use it."
Label: Complaint

Comment: "Thanks for the quick delivery!"
Label: Thank You

Comment: "My package arrived damaged."
Label:
```

**Sample Output:**
> Complaint

**Observation:** Providing examples allowed the model to correctly infer that “damaged package” is a complaint without confusion.

***

## **4. Evaluate Quality**

**Dataset:** Job advertisement

**Vague Prompt:**

```
Summarize this job ad: We are seeking a software developer...
```

**Sample Output:**
> Looking for a software developer with relevant skills.

**Refined Prompt:**

```
Summarize this job ad in 2 sentences, including required skills and main responsibilities.

Job ad: We are seeking a software developer with experience in Python, JavaScript, and cloud platforms. Responsibilities include building web applications, collaborating with designers, and deploying solutions to cloud infrastructure.
```

**Sample Output:**
> The company seeks a software developer skilled in Python, JavaScript, and cloud platforms. Key duties include building web apps, collaborating with designers, and deploying to cloud infrastructure.

**Observation:** The refined prompt ensured both skills and responsibilities appeared clearly.

***

## **5. Divide Labor (Break Complex Tasks)**

**Dataset:** University science fair

**Step 1 Prompt:**

```
List 5 types of exhibits and activities for middle school visitors at a science fair.
```

**Sample Output:**

- Interactive physics experiments
- Simple robotics demos
- Fun chemistry reaction shows
- Space exploration VR booth
- Renewable energy model displays

**Step 2 Prompt:**

```
Propose a timeline for a one-day science fair with setup, activities, breaks, and closing ceremony.
```

**Sample Output:**
> 8:00–9:00 AM: Setup
> 9:00–9:30 AM: Opening remarks
> 9:30–12:00 PM: Morning exhibits and demos
> 12:00–1:00 PM: Lunch break
> 1:00–3:30 PM: Afternoon activities/workshops
> 3:30–4:00 PM: Awards and closing
> 4:00–5:00 PM: Pack-up

**Step 3 Prompt:**

```
Draft 5 safety rules for participants and guests at the science fair.
```

**Sample Output:**

- No running near exhibits
- Supervise all chemical experiments
- Wear safety goggles during lab demos
- Keep food/drinks away from displays
- Follow event staff instructions at all times

**Combined Final Proposal:** Merge the three steps into sections: *Activities*, *Timeline*, *Safety Rules*.

**Observation:** Decomposing into steps created richer, more organized results.

***

✅ **Ready for Colab:**
Each solution can be run in a Colab code cell using your `ask_gpt(prompt)` helper to compare model outputs against these examples.

***

