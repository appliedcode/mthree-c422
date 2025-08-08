## Solution: Comparative Analysis of Transformer Components Using Movie Plot Summaries

### Prerequisites

- Python environment (Google Colab recommended)
- Libraries: PyTorch, NumPy, Matplotlib/Seaborn for plotting
- Dataset: Download either CMU Movie Summary Corpus or IMDb Movie Plot Summaries (select and download manually)


### Task 1: Data Preparation

1. **Download and select a subset (10–20 summaries)**
    - Load sample movie plot summaries into a list.
2. **Text cleaning and tokenization**

```python
import re
from transformers import BertTokenizer

# Example: clean and tokenize sample plots
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

sample_texts = [clean_text(plot) for plot in selected_plots]

# Tokenize
tokenized = [tokenizer.tokenize(text) for text in sample_texts]
token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized]
```

3. **Convert tokens to embeddings**
Use pretrained or random embeddings (for demonstration, random embeddings can suffice)

```python
import torch
import torch.nn as nn

embedding_dim = 64
vocab_size = tokenizer.vocab_size

# Random initialization for demonstration
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Example: Get embeddings for first sentence tokens
inputs = torch.tensor(token_ids[0]).unsqueeze(0)  # batch_size = 1
embeddings = embedding_layer(inputs)  # shape: [1, seq_len, embedding_dim]
```


### Task 2: Self-Attention Implementation and Visualization

1. **Implement scaled dot-product self-attention**

```python
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

2. **Apply self-attention on embeddings**
Use `Q = K = V = embeddings` to implement self-attention on the input token embeddings.
3. **Visualize attention weights as heatmaps**
Using `matplotlib` or `seaborn`:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attn_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.squeeze().detach().cpu().numpy(), xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title('Self-Attention Weight Matrix')
    plt.show()

# Example after running self-attention
plot_attention(attn_weights, tokenized[0])
```

4. **Interpret attention**
Observe which query token attends to which key tokens and relate it to semantic importance or syntactic relations.

### Task 3: Multi-Head Attention Implementation and Visualization

1. **Implement Multi-Head Attention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = []
        attn_weights_all = []
        for i in range(self.num_heads):
            out_i, attn_i = scaled_dot_product_attention(Q[:, i], K[:, i], V[:, i])
            out.append(out_i)
            attn_weights_all.append(attn_i)

        concatenated = torch.cat(out, dim=-1)
        output = self.out_linear(concatenated)
        return output, attn_weights_all
```

2. **Apply on embeddings and visualize each head's attention heatmap**
Plot each head’s attention matrix similarly to self-attention visualization.
3. **Comparison insights**
Note how each head focuses on different relational aspects of the tokens, showing diverse specialization.

### Task 4: Feedforward Neural Network (FFN) Application and Comparison

1. **Implement position-wise FFN**

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

2. **Apply FFN to multi-head attention output**

```python
ff = PositionwiseFeedForward(embed_dim=embedding_dim, ff_dim=256)
ffn_output = ff(output)  # output from Multi-Head Attention
```

3. **Compare token embeddings before and after FFN**
Use dimensionality reduction (e.g., PCA or t-SNE) or cosine similarity to see how FFN refines embeddings.

### Task 5: Comparative Analysis Report

- Compile heatmaps of attention weights for single-head and multi-head attention.
- Include visualizations/examples of token embeddings pre- and post-FFN.
- Discuss:
    - Self-Attention captures token dependencies using a single attention distribution.
    - Multi-Head Attention expands this by attending to multiple different subspaces/features simultaneously.
    - FFNs provide nonlinear transformations that enhance and refine token representations after attention.
- Reflect on how these components contribute to transformers' ability to model complex language phenomena.


### Optional Extensions

- Add positional encodings to embeddings before attention.
- Stack multiple transformer layers.
- Use pretrained embeddings (e.g., BERT or GloVe) for richer initial token representations.

If you want, I can provide you with a ready-to-run Colab notebook script implementing these components with detailed visualization and commentary. Just let me know!

