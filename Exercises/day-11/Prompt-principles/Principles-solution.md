## **💡 Solutions: Prompt Engineering Principles with Sample Datasets**


***

### **1. Give Clear Direction**

**Dataset:** Renewable energy article snippet

**Vague Prompt:**

``` 
Write about energy.
```

**Sample Output:**
> Energy is important because it powers our homes, cars, and industries. There are many types of energy like electricity, heat, and solar energy…

**Clear Direction Prompt:**

```
Explain renewable energy sources (solar, wind, hydropower) to high school students in an informative but simple tone, 100 words or less.
```

**Sample Output:**
> Renewable energy comes from sources that won’t run out, such as sunlight, wind, and flowing water. Solar panels turn sunlight into electricity. Wind turbines use moving air to spin blades that generate power. Hydropower comes from dams that use moving water to produce energy. These methods reduce pollution and help fight climate change. They are sustainable, meaning they can provide power for generations without depleting natural resources.

**Observation:**
Clear direction improved audience focus, kept language simple, and stayed within word count.

***

### **2. Specify Format**

**Dataset:** Smartphone specs

**Prompt without format:**

```
Compare Phone A, Phone B, and Phone C.
```

**Sample Output:**
> Phone A has an OLED display, 12MP camera, and 4000mAh battery. Phone B features AMOLED, 108MP, and 4500mAh battery. Phone C uses LCD display…

**Prompt with format specified:**

```
Compare Phone A, Phone B, and Phone C in a markdown table with columns: Phone, Display, Camera, Battery.
```

**Sample Output:**


| Phone | Display | Camera | Battery |
| :-- | :-- | :-- | :-- |
| Phone A | OLED | 12 MP | 4000 mAh |
| Phone B | AMOLED | 108 MP | 4500 mAh |
| Phone C | LCD | 48 MP | 3500 mAh |

**Observation:**
Formatting made comparison faster and easier to read.

***

### **3. Provide Examples (Few‑Shot Prompting)**

**Dataset:** Coffee shop reviews

**Few‑Shot Prompt:**

```
Classify each review as Positive, Negative, or Neutral:

Review: "Loved the espresso, very rich flavor."
Sentiment: Positive

Review: "Service was slow but the ambiance was nice."
Sentiment: Neutral

Review: "The croissant was stale and dry."
Sentiment: Negative

Review: "Great coffee but a bit noisy during peak hours."
Sentiment:
```

**Sample Output:**
> Neutral

**Observation:**
The model followed example pattern — identified a mixed review (positive \& negative) as Neutral, consistent with prior examples.

***

### **4. Evaluate Quality**

**Dataset:** Urban gardening article

**Vague Prompt:**

```
Summarize this text: "Community gardens are flourishing…"
```

**Sample Vague Output:**
> Community gardens are becoming more common and help cities by providing fresh produce and improving the environment.

**Refined Prompt:**

```
Summarize this article in 3 bullet points each covering:
1) Main benefits
2) Social impact
3) Environmental effects

Text: Community gardens are flourishing in cities worldwide, providing fresh produce and green spaces for residents. They improve air quality, foster social connections, and promote sustainability.
```

**Sample Output:**

- **Benefits:** Fresh produce and green spaces for residents
- **Social Impact:** Strengthens community bonds and encourages collaboration
- **Environmental Effects:** Improves air quality and supports urban sustainability

**Observation:**
Refinement ensured all key aspects were covered with clear structure.

***

### **5. Divide Labor (Break Complex Tasks)**

**Dataset:** Online bookstore launch plan

**Step 1 Prompt:**

```
Describe the target audience for an online bookstore focused on rare and antique books.
```

**Sample Output:**
> - Book collectors aged 30–65
> - Academic researchers seeking historical texts
> - Literary enthusiasts with niche interests
> - Higher income group with disposable income for collectibles

**Step 2 Prompt:**

```
Based on the above audience, create a marketing strategy with 5 bullet points.
```

**Sample Output:**
> - Target niche Facebook and Instagram groups for book lovers
> - Collaborate with libraries and literary societies
> - Send monthly email newsletters with featured rare books
> - Host virtual book appreciation webinars
> - Offer limited‑time discounts on special collections

**Step 3 Prompt:**

```
Propose a 1-month social media content calendar in a table with Date, Post Idea, and Caption.
```

**Sample Output:**


| Date | Post Idea | Caption |
| :-- | :-- | :-- |
| Mar 1 | Featured Book of the Month | "Discover the history behind this rare gem 📖" |
| Mar 5 | Behind-the-scenes: sourcing | "How we find treasures for true book lovers" |
| Mar 10 | Customer spotlight | "Meet Jane, a proud owner of 1st edition Austen" |
| … | … | … |

**Final Plan:** Combined insights gave a clear audience definition, targeted strategies, and an actionable posting schedule.

**Observation:**
Breaking the big request into precise steps made outputs richer and more usable.

***

✅ **Tip for Students:**
When working in Colab, place each principle in a separate code cell, run it, and **document how the results improved** after prompt adjustments.


