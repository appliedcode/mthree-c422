## Exercise Set: Transformer-Based Encoder-Decoder Practice

### Exercise 1: **Parameter Experimentation**

**Task:**
Modify your existing summarization code to experiment with different parameters. Try these variations:

- Change `max_length` to 30, 60, and 100
- Change `min_length` to 10, 25, and 40
- Set `do_sample=True` and add `temperature=0.7`

**Expected Output:** Compare the summaries and observe how parameter changes affect output quality and length.

**Hint:**
Think about how `max_length` controls summary length and how `do_sample=True` with temperature affects creativity vs. consistency. Document which settings work best for your text.

### Exercise 2: **Different Text Types**

**Task:**
Test your summarizer on different types of text:

```python
# Add these text samples to your code
news_article = "Scientists at MIT have developed a new battery technology that could revolutionize electric vehicles. The lithium-metal battery can store twice as much energy as current batteries and charge 50% faster. The research team, led by Dr. Sarah Johnson, spent three years developing this breakthrough. The technology could be commercially available within five years, potentially making electric cars more affordable and practical for everyday use."

recipe_text = "To make chocolate chip cookies, you'll need flour, sugar, butter, eggs, vanilla, and chocolate chips. First, preheat your oven to 375°F. Mix the dry ingredients in one bowl and wet ingredients in another. Combine them slowly, then fold in chocolate chips. Drop spoonfuls of dough on a baking sheet and bake for 10-12 minutes until golden brown."

email_text = "Hi everyone, I wanted to update you on our quarterly sales results. We exceeded our target by 15% this quarter, thanks to strong performance in the mobile app division. The marketing campaign we launched in July was particularly successful, generating 200 new leads. Our customer satisfaction scores also improved by 8%. Great work team, and let's keep the momentum going into Q4!"
```

**Hint:**
Notice how the model handles different writing styles. Does it work better for formal vs. informal text? Technical vs. conversational content?

### Exercise 3: **Model Comparison**

**Task:**
Compare different pre-trained models by replacing `"facebook/bart-large-cnn"` with these alternatives:

- `"google/t5-small"` (T5 model)
- `"facebook/bart-base"` (smaller BART)
- `"sshleifer/distilbart-cnn-12-6"` (lightweight BART)

**Hint:**
You'll need to add a prefix for T5: `"summarize: " + text`. Compare speed, output quality, and resource usage. Which model works best for your needs?

### Exercise 4: **Custom Input Processing**

**Task:**
Create a function that automatically splits long texts and summarizes each part:

```python
def smart_summarize(text, chunk_size=500, overlap=50):
    # Split text into overlapping chunks
    # Summarize each chunk
    # Combine summaries
    # Return final result
    pass
```

**Hint:**
Think about how to handle text longer than the model's token limit. Consider sentence boundaries when splitting and how to merge chunk summaries coherently.

### Exercise 5: **Interactive Summarizer**

**Task:**
Build a simple interactive program:

```python
def interactive_summarizer():
    print("📝 Text Summarizer")
    print("Enter 'quit' to exit")
    
    while True:
        user_input = input("\nEnter text to summarize: ")
        if user_input.lower() == 'quit':
            break
        
        # Add your summarization code here
        # Handle empty input and very short text
        
    print("Thanks for using the summarizer!")
```

**Hint:**
Add error handling for empty inputs, very short texts (less than min_length), and potential model errors. Make the interface user-friendly.

### Exercise 6: **Evaluation Metrics**

**Task:**
Create a simple evaluation system:

```python
def evaluate_summary(original_text, summary):
    # Calculate compression ratio
    compression_ratio = len(summary) / len(original_text)
    
    # Count sentences in original vs summary  
    original_sentences = original_text.count('.') + original_text.count('!') + original_text.count('?')
    summary_sentences = summary.count('.') + summary.count('!') + summary.count('?')
    
    # Return metrics
    return {
        'compression_ratio': compression_ratio,
        'original_sentences': original_sentences,
        'summary_sentences': summary_sentences
    }
```

**Hint:**
Think about what makes a good summary. Consider length reduction, information retention, and readability. Add more sophisticated metrics if you want a challenge.

### Bonus Challenge: **Multi-Task Pipeline**

**Task:**
Create a pipeline that can handle multiple NLP tasks:

```python
# Initialize different pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("sentiment-analysis")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

def multi_task_processor(text, tasks=['summarize', 'sentiment', 'translate']):
    results = {}
    # Process text through selected tasks
    # Return combined results
    pass
```

**Hint:**
Consider the order of operations (e.g., should you translate before or after summarizing?). Handle different input/output formats for each task.

## **Getting Started Tips:**

1. Start with Exercise 1 - it's the simplest modification of your existing code
2. Copy your current code and modify it step by step
3. Test each change with the same input text to see differences
4. Print intermediate results to understand what's happening
5. Don't worry if some experiments don't work perfectly - learning from failures is valuable!

**Remember:** The goal is to understand how Transformer encoder-decoder models work in practice, not to create perfect solutions!

<div style="text-align: center">⁂</div>

[^1]: https://www.cs.upc.edu/~padro/ahlt/exercises/08-Exercises-Transformers-AHLT-SOLVED.pdf

[^2]: https://www.width.ai/post/4-long-text-summarization-methods

[^3]: https://discuss.huggingface.co/t/t5-outperforms-bart-when-fine-tuned-for-summarization-task/20009

[^4]: https://www.youtube.com/watch?v=X_lyR0ZPQvA

[^5]: https://www.geeksforgeeks.org/text-summarization-techniques/

[^6]: https://www.cs.utexas.edu/~gdurrett/courses/fa2021/lectures/lec24-4pp.pdf

[^7]: https://machinelearningmastery.com/encoders-and-decoders-in-transformer-models/

[^8]: https://www.gurully.com/pte-writing-practice/summarize-written-text

[^9]: https://www.cs.utexas.edu/~gdurrett/courses/fa2022/lectures/lec21-4pp.pdf

[^10]: https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/

[^11]: https://goarno.io/blog/summarize-written-text-examples-and-answers-pearson-test-of-english/

[^12]: https://svivek.com/teaching/machine-learning/lectures/slides/t5/t5.pdf

[^13]: https://huggingface.co/learn/llm-course/en/chapter1/5

[^14]: https://www.ereadingworksheets.com/free-reading-worksheets/reading-comprehension-worksheets/summarizing-worksheets-and-activities/

[^15]: https://pub.aimind.so/fine-tuning-t5-for-summarization-a-beginners-guide-1d0fce60f680

[^16]: https://www.kaggle.com/code/nadaahassan/transformer-network

[^17]: https://quillbot.com/courses/english-literacy-and-composition-b/chapter/text-summary-writing/

[^18]: https://huggingface.co/docs/transformers/en/model_doc/t5

[^19]: https://e2eml.school/transformers.html

[^20]: https://owl.purdue.edu/owl_exercises/multilingual_exercises/paraphrase_and_summary_exercises/basic_level_paraphrase_and_summary_writing.html

