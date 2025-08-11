# **Team Project: End‑to‑End Prompt Engineering with Climate Change \& Environmental Policy Dataset**

## **Project Overview**

Your team will design and implement an AI‑powered application in **Google Colab** using the **OpenAI API**, applying **all key aspects of prompt engineering** — from fundamentals to advanced techniques, safety, ethics, privacy, integration, monitoring, and performance evaluation — this time focused on the domain of **climate change and environmental policy**.

You will adapt the prompt designs, pipeline workflows, monitoring, and ethics modules to this **new dataset**.

***

## **Dataset**

Embed this text in your Colab notebook as your working dataset for summarisation, Q\&A, structured extraction, and testing:

```
Climate Change & Environmental Policy Dataset:

1. Greenhouse Gas Emissions: Emissions from fossil fuels, agriculture, and industry contribute to global warming. Key gases: CO2, methane, nitrous oxide. Main challenge: reducing emissions while balancing economic growth.

2. Renewable Energy Transition: Moving from coal, oil, and gas to renewable sources like solar, wind, and hydro. Benefits: reduced emissions, sustainable supply. Challenges: storage, grid modernization, initial investment costs.

3. Climate Adaptation Strategies: Measures to cope with climate impacts, e.g., sea wall construction, drought-resistant crops, disaster preparedness. Benefits: reduced vulnerability. Challenges: funding, equitable implementation.

4. International Climate Agreements: Treaties like the Paris Agreement aim to limit global temperature rise. Goals include emission targets and financial/climate aid to developing nations. Challenges: enforcement, political will.

5. Biodiversity Protection: Conservation of species and habitats through protected areas, anti-poaching laws, and restoration projects. Benefits: ecological balance, cultural heritage preservation. Challenges: habitat loss, funding, illegal exploitation.
```


***

## **Core Tasks**

### **1. Fundamentals**

- Write **direct, zero‑shot, and few‑shot prompts** explaining climate change concepts or dataset entries.
- Compare outputs for clarity and accuracy.


### **2. Intermediate Techniques**

- Apply **chain‑of‑thought prompting** to break down causes, impacts, and solutions of climate change topics.
- Use **persona‑based prompting** (e.g., environmental activist, policy advisor) for explaining data in plain language.


### **3. Advanced Techniques**

- Create **multi‑turn conversations** maintaining context (e.g., UN climate summit dialogue).
- Identify and troubleshoot an **ambiguous environmental prompt** by refining it with dataset information.
- Write **domain‑specific prompts** for:
    - Summarising dataset entries.
    - Suggesting climate policies for sample scenarios.
    - Extracting structured JSON data with fields like "topic", "benefits", "challenges".


### **4. Safety \& Ethics**

- Write prompts avoiding political bias, unverified claims, or alarmist language.
- Add fact‑based constraints and fairness instructions.


### **5. Integration \& Version Control**

- Maintain **three versions** of a key explanatory prompt (simple → improved → few‑shot) and compare results.
- Build a **two‑stage pipeline** (e.g., summarise a dataset topic → auto‑generate quiz questions).


### **6. Monitoring \& Evaluation**

- Log each prompt, its latency, and excerpt of the output.
- Score keyword relevance (keywords like “emissions”, “renewable”, “policy”).
- Measure **consistency** of repeated runs using similarity ratios.


### **7. Security \& Privacy**

- Simulate unsafe requests (e.g., “Give confidential negotiation details”) and demonstrate safe refusals.
- Restrict model output to **public domain facts** from the dataset.

***

## **Deliverables**

1. **Google Colab Notebook** including:
    - Fully adapted prompts using the climate dataset.
    - Code for context management, chaining, safety, metrics, and logging.
    - At least one creative domain‑specific extension.
2. **Team Report** (3‑5 pages):
    - Overview of methods, observations, and improvements.
    - Ethical, safety, privacy measures applied.
    - Potential real‑world applications of your system.

***

## **Evaluation Criteria**

- Full coverage of prompt engineering concepts.
- Effective domain adaptation.
- Creative and accurate outputs.
- Evidence of monitoring, analytics, and ethical safeguards.
- Quality \& clarity of documentation.

***

💡 **Tip for Teams:**
Consider adding realistic scenarios for testing — such as advising a city on renewable policy, or crafting a public awareness message — while keeping outputs factual and non‑partisan.

***