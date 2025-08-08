# Advanced Transformer-Based Encoder-Decoder Practice Exercises

Building on your basic summarization program, here are more comprehensive exercises to deepen your understanding of Transformer-based Encoder-Decoder models:

## Exercise 7: **Model Architecture Exploration**

**Task:**
Compare different encoder-decoder architectures by implementing the same summarization task using:

- BART variants: `facebook/bart-base`, `facebook/bart-large`, `facebook/bart-large-cnn`
- T5 variants: `t5-small`, `t5-base`, `google/t5-efficient-base`
- Other models: `google/pegasus-xsum`, `sshleifer/distilbart-cnn-12-6`

Create a comparison table showing:

- Model size (parameters)
- Inference time
- Memory usage
- Summary quality (subjective scoring)

**Hint:**
Use `model.num_parameters()` to get parameter counts. Time your inference with `time.time()`. For T5 models, remember to prefix with `"summarize: "`.

## Exercise 8: **Custom Dataset Fine-tuning**

**Task:**
Create a small custom dataset and fine-tune a pre-trained model:

```python
# Create your dataset
custom_data = [
    {"article": "Your long text here...", "summary": "Your summary here..."},
    # Add 20-50 examples
]

# Fine-tune BART or T5 on this data
# Evaluate before and after fine-tuning
```

**Hint:**
Use Hugging Face's `Trainer` class. Start with a very small learning rate (1e-5). Create train/validation splits. Compare performance on your custom domain vs. general summarization.

## Exercise 9: **Attention Visualization**

**Task:**
Extract and visualize attention patterns from your summarization model:

```python
def visualize_attention(model, tokenizer, text):
    # Tokenize input
    # Get model outputs with attention weights
    # Create heatmap showing which input tokens 
    # the model focuses on for each output token
    pass
```

**Expected Output:** Heatmap showing attention patterns during summarization.

**Hint:**
Set `output_attentions=True` when calling the model. Use `matplotlib` or `seaborn` for visualization. Focus on cross-attention between encoder and decoder.

## Exercise 10: **Multi-Task Learning**

**Task:**
Modify your pipeline to handle multiple tasks with the same T5 model:

```python
def multi_task_processor(text, task_type):
    tasks = {
        'summarize': 'summarize: ',
        'translate_to_french': 'translate English to French: ',
        'answer_question': 'question: {} context: {}',
        'paraphrase': 'paraphrase: '
    }
    # Implement task routing
    pass
```

**Hint:**
T5 was designed for multi-task learning. Use appropriate task prefixes. Test on different types of text to see how well the model generalizes.

## Exercise 11: **Beam Search vs. Sampling Comparison**

**Task:**
Implement and compare different text generation strategies:

```python
def compare_generation_methods(text):
    methods = {
        'greedy': {'do_sample': False},
        'beam_search': {'do_sample': False, 'num_beams': 4},
        'nucleus_sampling': {'do_sample': True, 'top_p': 0.9, 'temperature': 0.8},
        'top_k_sampling': {'do_sample': True, 'top_k': 50, 'temperature': 0.7}
    }
    
    results = {}
    for method, params in methods.items():
        # Generate summary with each method
        # Store and compare results
        pass
    return results
```

**Hint:**
Notice differences in creativity vs. consistency. Beam search tends to be more deterministic, while sampling methods produce more varied outputs.

## Exercise 12: **Domain Adaptation**

**Task:**
Test your model's performance across different domains:

```python
domains = {
    'scientific': "Recent studies in quantum computing show...",
    'legal': "The plaintiff argues that the defendant...", 
    'medical': "The patient presented with symptoms of...",
    'technical': "The API endpoint accepts POST requests...",
    'news': "Breaking news: Local elections results..."
}

# Test summarization quality across domains
# Identify which domains work best/worst
# Suggest improvements for poor domains
```

**Hint:**
Some models are trained on specific domains (like `facebook/bart-large-cnn` for news). Consider domain-specific preprocessing or model selection.

## Exercise 13: **Evaluation Metrics Implementation**

**Task:**
Implement automatic evaluation metrics for summarization:

```python
def evaluate_summaries(original_texts, generated_summaries, reference_summaries):
    metrics = {}
    
    # ROUGE scores (if you have reference summaries)
    # BLEU scores  
    # BERTScore for semantic similarity
    # Custom metrics: compression ratio, readability
    
    return metrics
```

**Hint:**
Install `rouge-score`, `nltk`, and `bert-score` packages. ROUGE measures n-gram overlap. BERTScore uses contextual embeddings for semantic similarity.

## Exercise 14: **Real-time Streaming Summarization**

**Task:**
Build a system that can summarize streaming text:

```python
class StreamingSummarizer:
    def __init__(self, model_name, window_size=1000):
        self.buffer = ""
        self.window_size = window_size
        # Initialize model
    
    def add_text(self, new_text):
        # Add to buffer
        # If buffer exceeds window size, summarize and reset
        pass
    
    def get_current_summary(self):
        # Return summary of current buffer
        pass
```

**Hint:**
Think about how to handle sentence boundaries when the buffer gets full. Consider overlapping windows to maintain context.

## Exercise 15: **Multilingual Summarization**

**Task:**
Experiment with multilingual models:

```python
multilingual_models = [
    "facebook/mbart-large-50-many-to-many-mmt",
    "google/mt5-small"
]

# Test summarization in different languages
languages = ['en', 'es', 'fr', 'de', 'it']

# Compare performance across languages
```

**Hint:**
Some models require language codes as input. mBART and mT5 support multiple languages. Test both input-output language matching and cross-lingual summarization.

## Exercise 16: **Model Distillation**

**Task:**
Create a smaller, faster model by distilling knowledge from a larger one:

```python
def knowledge_distillation(teacher_model, student_model, training_data):
    # Use teacher model outputs as soft targets
    # Train student model to mimic teacher's behavior
    # Compare performance vs. speed trade-offs
    pass
```

**Hint:**
Use the teacher model's probability distributions as targets, not just the final predictions. This is more advanced but very practical for deployment.

## Bonus Challenge: **End-to-End Application**

**Task:**
Build a complete summarization web application:

```python
# Features to implement:
# - File upload (PDF, TXT, DOCX)
# - Multiple summarization models
# - Adjustable summary length
# - Export options
# - Performance metrics display
# - Batch processing
# - API endpoint for integration
```

**Hint:**
Use Streamlit or Flask for the web interface. Consider using Celery for background task processing. Add error handling and input validation.

## **Learning Objectives:**

After completing these exercises, students should understand:

1. **Architecture differences** between BART, T5, and other encoder-decoder models
2. **Attention mechanisms** and how they work in practice
3. **Fine-tuning strategies** for domain-specific tasks
4. **Generation methods** and their trade-offs
5. **Evaluation metrics** for sequence-to-sequence tasks
6. **Practical deployment** considerations
7. **Multilingual capabilities** of modern transformers
8. **Performance optimization** techniques

## **Getting Started Tips:**

1. Begin with exercises 7-8 as they build directly on your existing code
2. For advanced exercises (13-16), research the concepts first
3. Use smaller models (base/small) for experimentation to save time
4. Keep detailed notes on performance differences
5. Create reusable utility functions for common operations

**Remember:** These exercises are designed to be challenging and may require additional research. The goal is to deeply understand how transformer-based encoder-decoder models work in practice!

<div style="text-align: center">⁂</div>

[^1]: https://www.cs.upc.edu/~padro/ahlt/exercises/08-Exercises-Transformers-AHLT-SOLVED.pdf

[^2]: https://smashinggradient.com/2023/05/11/tinkering-with-peft-bart-t5-flan-for-spell-correction/

[^3]: https://arxiv.org/html/2502.19597v1

[^4]: https://www.youtube.com/watch?v=X_lyR0ZPQvA

[^5]: https://www.cs.utexas.edu/~gdurrett/courses/fa2021/lectures/lec24-4pp.pdf

[^6]: https://www.ultralytics.com/glossary/sequence-to-sequence-models

[^7]: https://www.geeksforgeeks.org/deep-learning/working-of-encoders-in-transformers/

[^8]: https://huggingface.co/blog/sagemaker-distributed-training-seq2seq

[^9]: https://www.geeksforgeeks.org/seq2seq-model-in-machine-learning/

[^10]: https://machinelearningmastery.com/encoders-and-decoders-in-transformer-models/

[^11]: https://www.youtube.com/watch?v=HDSNjrxSwqw

[^12]: http://lena-voita.github.io/nlp_course/seq2seq_and_attention.html

[^13]: https://www.kaggle.com/code/nadaahassan/transformer-network

[^14]: https://www.iieta.org/download/file/fid/132319

[^15]: https://hannibunny.github.io/mlbook/transformer/attention.html

[^16]: https://huggingface.co/blog/encoder-decoder

[^17]: https://projectai.in/projects/e19e2cad-9e90-4283-a7b4-095f1267bc19/tasks/5fcd07ca-8aa1-4002-bbf4-24a42009458a

[^18]: https://en.wikipedia.org/wiki/Seq2seq

[^19]: https://www.ibiblio.org/kuphaldt/socratic/output/ELTR145_sec3.pdf

[^20]: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Encoder_Decoder_Model.ipynb

