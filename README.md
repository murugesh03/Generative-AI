# 🤖 Generative AI — The Complete Guide
### From Fundamentals to Cutting-Edge Concepts

> A comprehensive reference covering every Generative AI concept, tool, technique, and framework with detailed explanations and practical code samples.

---

## 📚 Table of Contents

1. [Introduction to Generative AI](#1-introduction-to-generative-ai)
2. [Foundations of Machine Learning](#2-foundations-of-machine-learning)
3. [Neural Networks Deep Dive](#3-neural-networks-deep-dive)
4. [Transformers Architecture](#4-transformers-architecture)
5. [Large Language Models (LLMs)](#5-large-language-models-llms)
6. [Prompt Engineering](#6-prompt-engineering)
7. [Fine-Tuning & Training](#7-fine-tuning--training)
8. [Retrieval-Augmented Generation (RAG)](#8-retrieval-augmented-generation-rag)
9. [Vector Databases & Embeddings](#9-vector-databases--embeddings)
10. [AI Agents & Agentic Systems](#10-ai-agents--agentic-systems)
11. [Image Generation Models](#11-image-generation-models)
12. [Audio & Speech Generation](#12-audio--speech-generation)
13. [Video Generation](#13-video-generation)
14. [Multimodal AI](#14-multimodal-ai)
15. [Model Evaluation & Benchmarks](#15-model-evaluation--benchmarks)
16. [AI Safety & Alignment](#16-ai-safety--alignment)
17. [Inference Optimization](#17-inference-optimization)
18. [LangChain Framework](#18-langchain-framework)
19. [LlamaIndex Framework](#19-llamaindex-framework)
20. [OpenAI API Deep Dive](#20-openai-api-deep-dive)
21. [Anthropic Claude API](#21-anthropic-claude-api)
22. [Hugging Face Ecosystem](#22-hugging-face-ecosystem)
23. [Local LLMs & Ollama](#23-local-llms--ollama)
24. [Model Context Protocol (MCP)](#24-model-context-protocol-mcp)
25. [AI Memory Systems](#25-ai-memory-systems)
26. [Structured Output & Function Calling](#26-structured-output--function-calling)
27. [AI Observability & Monitoring](#27-ai-observability--monitoring)
28. [Responsible AI & Ethics](#28-responsible-ai--ethics)
29. [Production AI Systems](#29-production-ai-systems)
30. [Future of Generative AI](#30-future-of-generative-ai)

---

## 1. Introduction to Generative AI

### What is Generative AI?

Generative AI refers to artificial intelligence systems that can **create new content** — text, images, audio, video, code, and more — that resembles human-created output. Unlike discriminative AI (which classifies or predicts), generative models learn the underlying distribution of training data and can sample from that distribution to produce novel outputs.

```
Discriminative AI:  Input → "Is this a cat or dog?"
Generative AI:      Prompt → "Generate an image of a cat"
```

### The Generative AI Landscape

```
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATIVE AI                               │
├─────────────┬──────────────┬─────────────┬──────────────────────┤
│    Text     │    Images    │    Audio    │       Video          │
│             │              │             │                      │
│ GPT-4       │ Stable       │ ElevenLabs  │ Sora                 │
│ Claude 3    │ Diffusion    │ Whisper     │ Runway               │
│ Gemini      │ DALL-E 3     │ MusicGen    │ Pika                 │
│ Llama 3     │ Midjourney   │ Bark        │ Gen-3                │
│ Mistral     │ Flux         │ Suno        │ Kling                │
└─────────────┴──────────────┴─────────────┴──────────────────────┘
```

### History & Timeline

| Year | Milestone |
|------|-----------|
| 2014 | GANs introduced by Ian Goodfellow |
| 2017 | "Attention Is All You Need" — Transformer paper |
| 2018 | GPT-1, BERT released |
| 2019 | GPT-2 (1.5B params) — considered "too dangerous" |
| 2020 | GPT-3 (175B params), scaling laws paper |
| 2021 | DALL-E, Codex, GitHub Copilot |
| 2022 | ChatGPT launch (Nov), Stable Diffusion open-source |
| 2023 | GPT-4, Claude, Llama, Gemini, Midjourney v5 |
| 2024 | GPT-4o, Claude 3, Gemini 1.5, Llama 3, Sora |
| 2025 | Reasoning models, Agentic AI, Multimodal everywhere |

### Core Capabilities

```python
# What Generative AI can do:

capabilities = {
    "text": [
        "Question answering", "Summarization", "Translation",
        "Creative writing", "Code generation", "Reasoning",
        "Sentiment analysis", "Chatbots", "Document analysis"
    ],
    "images": [
        "Text-to-image", "Image editing", "Style transfer",
        "Image restoration", "Object removal", "Background generation"
    ],
    "audio": [
        "Text-to-speech", "Voice cloning", "Music generation",
        "Audio transcription", "Sound effects generation"
    ],
    "video": [
        "Text-to-video", "Video editing", "Motion generation",
        "Video-to-video translation", "Avatar animation"
    ],
    "code": [
        "Code completion", "Code explanation", "Bug fixing",
        "Test generation", "Documentation", "Refactoring"
    ]
}
```

---

## 2. Foundations of Machine Learning

### Types of Machine Learning

```python
# Supervised Learning — labeled data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)   # learn from (input, expected_output) pairs

# Unsupervised Learning — find patterns in unlabeled data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)   # discover structure without labels

# Reinforcement Learning — learn from rewards/penalties
# Agent → takes Action → Environment → State + Reward → Agent (loop)
# Used in: RLHF (Reinforcement Learning from Human Feedback)

# Self-Supervised Learning — create labels from data itself
# Example: Predict next word from previous words (GPT pretraining)
# "The cat sat on the ___" → "mat"
```

### Key Mathematical Concepts

```python
import numpy as np

# ─── Vectors & Embeddings ────────────────────────────────────────────
# Words/tokens are represented as dense vectors in high-dimensional space
word_embedding = np.array([0.25, -0.71, 0.33, 0.89, -0.12])  # 5-dim example
# Real embeddings: 768, 1536, 3072 dimensions

# Cosine Similarity — how similar are two vectors?
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

king   = np.array([0.9, 0.1, 0.8, 0.9])
queen  = np.array([0.8, 0.9, 0.8, 0.9])
man    = np.array([0.9, 0.1, 0.2, 0.1])
woman  = np.array([0.8, 0.9, 0.2, 0.1])

# Famous: king - man + woman ≈ queen
result = king - man + woman
print(cosine_similarity(result, queen))   # high similarity!

# ─── Softmax — convert logits to probabilities ───────────────────────
def softmax(x):
    e_x = np.exp(x - np.max(x))   # numerical stability
    return e_x / e_x.sum()

logits = np.array([2.0, 1.0, 0.1, 3.5, -0.5])
probs = softmax(logits)
print(probs)   # [0.09, 0.03, 0.01, 0.83, 0.004] — sums to 1.0

# ─── Cross-Entropy Loss — how wrong is our prediction? ───────────────
def cross_entropy(predicted_probs, true_label_idx):
    return -np.log(predicted_probs[true_label_idx])

# Lower loss = better prediction

# ─── Gradient Descent — how models learn ─────────────────────────────
# Loss function measures error
# Gradient = direction of steepest ascent
# We move OPPOSITE to gradient to minimize loss

learning_rate = 0.01
weights = np.random.randn(10)

def training_step(weights, loss_gradient):
    return weights - learning_rate * loss_gradient   # parameter update

# Variants:
# SGD:     update per single example (noisy but fast)
# Mini-batch: update per batch of examples (balance)
# Adam:    adaptive learning rates per parameter (most popular)
```

### Tokenization

```python
# Tokenization: converting text to numbers the model can process
# Tokens ≈ ~0.75 words on average for English

# ─── Using tiktoken (OpenAI's tokenizer) ─────────────────────────────
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 encoding

text = "Hello, world! Generative AI is fascinating."
tokens = enc.encode(text)
print(tokens)        # [9906, 11, 1917, 0, 1183, 1413, 15592, 374, 27387, 13]
print(len(tokens))   # 10 tokens

# Decode back to text
decoded = enc.decode(tokens)
print(decoded)       # "Hello, world! Generative AI is fascinating."

# Token counts matter for:
# - Context window limits
# - API costs (charged per token)
# - Response generation speed

# Different tokenizers for different models:
# GPT-4:       cl100k_base  (~100K vocab)
# GPT-3:       p50k_base    (~50K vocab)
# Llama:       SentencePiece BPE
# Claude:      Custom BPE tokenizer

# ─── Byte-Pair Encoding (BPE) — how tokenizers are built ─────────────
# Start with characters, iteratively merge most frequent pairs
# "l o w e r" → "low er" → "lower"
# Balances vocabulary size with coverage
```

### Scaling Laws

```python
# Chinchilla Scaling Laws (Hoffmann et al., 2022)
# Optimal training: tokens ≈ 20× model parameters

# Examples:
# 7B  params  → train on ~140B tokens
# 70B params  → train on ~1.4T tokens
# 405B params → train on ~8T tokens

# Key insight: smaller models trained longer often beat
# larger models trained less — compute-optimal training

scaling_law = {
    "model_size":    "N parameters",
    "dataset_size":  "D tokens",
    "compute":       "C = 6ND FLOPs",
    "optimal_ratio": "D/N ≈ 20",
    "loss":          "L(N, D) = A/N^α + B/D^β + L_∞"
}
```

---

## 3. Neural Networks Deep Dive

### Basic Neural Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Simple Feedforward Network ──────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))     # activation function
        x = self.norm(x)               # normalize
        x = self.dropout(x)            # regularize
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNN(input_dim=768, hidden_dim=2048, output_dim=10)

# ─── Activation Functions ────────────────────────────────────────────
x = torch.randn(10)

relu = F.relu(x)           # max(0, x) — most common
gelu = F.gelu(x)           # smooth ReLU — used in transformers
silu = F.silu(x)           # sigmoid-weighted linear — used in Llama
sigmoid = torch.sigmoid(x) # (0,1) — for binary classification
tanh = torch.tanh(x)       # (-1,1) — RNN gates

# ─── Normalization ────────────────────────────────────────────────────
batch_norm  = nn.BatchNorm1d(hidden_dim)   # normalize over batch
layer_norm  = nn.LayerNorm(hidden_dim)     # normalize over features (used in LLMs)
rms_norm    = nn.RMSNorm(hidden_dim)       # simplified LayerNorm (Llama, Mistral)
group_norm  = nn.GroupNorm(8, hidden_dim)  # between batch and layer norm

# ─── Training Loop ────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()          # clear gradients
        predictions = model(batch_x)   # forward pass
        loss = loss_fn(predictions, batch_y)  # compute loss
        loss.backward()                # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()               # update weights
        scheduler.step()               # update learning rate
```

### Attention Mechanism — The Core of Transformers

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Q = Query  — "What am I looking for?"
    K = Key    — "What do I contain?"
    V = Value  — "What information do I provide?"
    """
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)   # Query projection
        self.W_k = nn.Linear(d_model, d_model)   # Key projection
        self.W_v = nn.Linear(d_model, d_model)   # Value projection
        self.W_o = nn.Linear(d_model, d_model)   # Output projection

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, heads, seq_len, d_k)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch, heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Project and split into heads
        Q = self.split_heads(self.W_q(Q), batch_size)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)

        # Attention
        attn_output, weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)
```

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Since transformers have no inherent order, we add positional information.
    Two types:
    1. Sinusoidal (original)       — absolute position
    2. RoPE (Rotary)               — relative position (Llama, Mistral)
    3. ALiBi                       — attention bias (Falcon)
    4. Learned embeddings          — GPT-style
    """
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─── RoPE (Rotary Position Embedding) — state of the art ─────────────
def apply_rotary_embeddings(q, k, cos, sin):
    """Used by Llama, Mistral, Qwen, and most modern LLMs"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

---

## 4. Transformers Architecture

### Full Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    Each transformer layer consists of:
    1. Multi-Head Self-Attention
    2. Feed-Forward Network (MLP)
    Both wrapped with residual connections and layer normalization

    Pre-norm architecture (modern LLMs):
    output = x + sublayer(LayerNorm(x))
    """
    def __init__(self, d_model=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection (pre-norm)
        x = x + self.dropout(self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Feed-forward with residual connection
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """Decoder-only transformer (GPT-style) — used for text generation"""
    def __init__(self, vocab_size=50257, d_model=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — token embedding and output head share weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        # Causal mask (can't attend to future tokens)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
```

### Transformer Variants

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURES                         │
├─────────────────┬───────────────────┬───────────────────────────────┤
│  Encoder-Only   │  Decoder-Only     │  Encoder-Decoder              │
│                 │                   │                               │
│  BERT, RoBERTa  │  GPT, Claude,     │  T5, BART, mT5               │
│  DistilBERT     │  Llama, Mistral   │  MarianMT                    │
│  ELECTRA        │  Falcon, Gemini   │  FLAN-T5                     │
│                 │                   │                               │
│  Use: Classif.  │  Use: Generation  │  Use: Translation,           │
│  Embedding      │  Completion       │  Summarization               │
│  Understanding  │  Chat             │  QA                          │
│                 │                   │                               │
│  Bidirectional  │  Unidirectional   │  Full attention              │
│  attention      │  (causal) attn    │  cross-attention             │
└─────────────────┴───────────────────┴───────────────────────────────┘
```

### Attention Variants

```python
# ─── Grouped Query Attention (GQA) — used in Llama 2/3, Mistral ──────
# Shares key/value heads across multiple query heads
# Reduces memory bandwidth without sacrificing much quality
# n_kv_heads < n_q_heads (e.g., 8 KV heads, 32 Q heads)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads=32, n_kv_heads=8):
        super().__init__()
        self.n_q_heads  = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads   # repetition factor = 4
        self.d_q = d_model // n_q_heads
        self.d_kv = d_model // n_kv_heads

        self.W_q = nn.Linear(d_model, n_q_heads  * self.d_q)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_kv)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_kv)
        self.W_o = nn.Linear(d_model, d_model)

# ─── Flash Attention — memory-efficient attention ─────────────────────
# Standard attention: O(n²) memory (stores full attention matrix)
# Flash Attention: O(n) memory (computes in tiles, never materializes full matrix)
# 2-4× speedup on A100 GPUs

# Using Flash Attention in PyTorch (2.0+)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)

# ─── Sparse Attention — attend to subset of tokens ───────────────────
# Local attention:   attend to nearby tokens only (window size w)
# Strided attention: attend to every k-th token
# Global tokens:     some tokens attend to all (like [CLS] in Longformer)
# Complexity: O(n*w) instead of O(n²)

# ─── Sliding Window Attention (Mistral) ──────────────────────────────
# Each token attends to W past tokens only
# But with multiple layers, effective receptive field grows
# Mistral 7B: window_size=4096

# ─── Multi-Query Attention (MQA) — used in Falcon, PaLM ──────────────
# Single KV head shared by all query heads
# Most memory efficient, some quality loss vs GQA
```

### KV Cache

```python
# KV Cache — crucial for efficient autoregressive generation
# During generation, we recompute the same K,V for previous tokens repeatedly
# KV Cache stores them so we only compute for the NEW token each step

class KVCache:
    """
    Without KV Cache: O(n²) operations for generating n tokens
    With KV Cache:    O(n) operations — each step only processes 1 new token
    """
    def __init__(self):
        self.k_cache = {}   # layer_id → cached keys
        self.v_cache = {}   # layer_id → cached values

    def update(self, layer_id, new_k, new_v):
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = new_k
            self.v_cache[layer_id] = new_v
        else:
            # Append new K,V to existing cache
            self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], new_k], dim=2)
            self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], new_v], dim=2)
        return self.k_cache[layer_id], self.v_cache[layer_id]

# KV Cache memory formula:
# memory = 2 × n_layers × n_kv_heads × d_head × seq_len × dtype_bytes
# Llama 3 70B at 4096 tokens: ~8GB KV cache!
```

---

## 5. Large Language Models (LLMs)

### How LLMs Generate Text

```python
# LLMs are next-token predictors trained on massive text corpora
# Generation is autoregressive: predict token → append → predict next token

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]  # last position

        # ── Temperature scaling ─────────────────────────────────────
        # temperature > 1: more random (creative)
        # temperature < 1: more deterministic (focused)
        # temperature = 0: greedy (always pick max)
        next_token_logits = next_token_logits / temperature

        # ── Top-K sampling ───────────────────────────────────────────
        # Keep only top K most likely tokens
        if top_k > 0:
            top_k_values = torch.topk(next_token_logits, top_k)[0]
            next_token_logits[next_token_logits < top_k_values[:, -1:]] = -float('inf')

        # ── Top-P (Nucleus) sampling ──────────────────────────────────
        # Keep smallest set of tokens whose cumulative prob ≥ p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = -float('inf')
            next_token_logits.scatter_(1, sorted_indices, sorted_logits)

        # Sample from distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Check for end-of-sequence
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

### Sampling Strategies Explained

```python
sampling_strategies = {

    "greedy": {
        "description": "Always pick highest probability token",
        "temperature": 0,
        "pros": "Deterministic, fast",
        "cons": "Repetitive, gets stuck in loops",
        "use_case": "When you need exact reproducibility"
    },

    "temperature": {
        "description": "Scale logits before softmax",
        "temp_0.1": "Near-deterministic, very focused",
        "temp_0.7": "Balanced creativity (most common)",
        "temp_1.0": "Sample from model distribution directly",
        "temp_1.5": "Very creative, sometimes incoherent",
        "use_case": "Creative writing vs factual Q&A"
    },

    "top_k": {
        "description": "Sample only from top K tokens",
        "typical_k": "50-200",
        "pros": "Prevents sampling very unlikely tokens",
        "cons": "Fixed cutoff regardless of distribution shape"
    },

    "top_p_nucleus": {
        "description": "Sample from smallest set summing to probability p",
        "typical_p": "0.9-0.95",
        "pros": "Adapts to distribution shape dynamically",
        "cons": "More compute than top-k"
    },

    "min_p": {
        "description": "Filter tokens below minimum probability threshold",
        "typical": "0.05-0.1",
        "pros": "Better than top-p for maintaining coherence",
        "note": "Increasingly popular in 2024"
    },

    "beam_search": {
        "description": "Maintain B best partial sequences simultaneously",
        "beam_size": "4-10 beams",
        "pros": "Higher quality, finds better sequences",
        "cons": "B× more memory and compute, less diverse",
        "use_case": "Translation, summarization where quality matters"
    },

    "contrastive_search": {
        "description": "Balance between confidence and degeneration",
        "alpha": "0.6 (degeneration penalty)",
        "pros": "Coherent and diverse output",
        "use_case": "Long-form generation"
    },

    "speculative_decoding": {
        "description": "Small draft model proposes tokens, large model verifies",
        "speedup": "2-4× generation speed",
        "same_distribution": True,
        "use_case": "Production inference optimization"
    }
}
```

### Major LLM Models Comparison

```python
llm_landscape = {
    "GPT-4o": {
        "creator": "OpenAI",
        "params": "~1.8T (MoE, estimated)",
        "context": "128K tokens",
        "capabilities": ["text", "vision", "audio"],
        "access": "API only",
        "strengths": "Coding, reasoning, instruction following"
    },
    "Claude 3.5 Sonnet": {
        "creator": "Anthropic",
        "params": "Unknown",
        "context": "200K tokens",
        "capabilities": ["text", "vision"],
        "access": "API only",
        "strengths": "Long context, safety, analysis, writing"
    },
    "Gemini 1.5 Pro": {
        "creator": "Google",
        "params": "Unknown",
        "context": "1M tokens (!)   ",
        "capabilities": ["text", "vision", "audio", "video"],
        "access": "API + Vertex AI",
        "strengths": "Multimodal, ultra-long context"
    },
    "Llama 3.1 405B": {
        "creator": "Meta",
        "params": "405B",
        "context": "128K tokens",
        "capabilities": ["text"],
        "access": "Open weights",
        "strengths": "Open source, customizable, self-hostable"
    },
    "Mistral Large": {
        "creator": "Mistral AI",
        "params": "~123B",
        "context": "128K tokens",
        "capabilities": ["text"],
        "access": "API + Open weights",
        "strengths": "Efficiency, multilingual, coding"
    },
    "Qwen2.5 72B": {
        "creator": "Alibaba",
        "params": "72B",
        "context": "128K tokens",
        "capabilities": ["text", "vision"],
        "access": "Open weights",
        "strengths": "Multilingual, coding, math"
    }
}
```

---

## 6. Prompt Engineering

### Core Prompting Principles

```python
# ─── Basic Prompt Structure ───────────────────────────────────────────
basic_prompt = """
You are a helpful assistant specialized in Python programming.

Task: Explain how decorators work in Python.

Format: 
- Start with a one-sentence definition
- Provide a simple code example
- Explain each part
- Give a real-world use case
"""

# ─── Zero-Shot Prompting ──────────────────────────────────────────────
zero_shot = """
Classify the sentiment of this review as Positive, Negative, or Neutral:

Review: "The product arrived on time but the quality was disappointing."

Sentiment:"""

# ─── Few-Shot Prompting ───────────────────────────────────────────────
few_shot = """
Classify the sentiment of reviews.

Review: "Amazing product, exactly what I needed!" → Positive
Review: "Terrible quality, broke after one day." → Negative
Review: "It's okay, nothing special." → Neutral
Review: "Fast shipping and good packaging." → Positive
Review: "Don't waste your money on this." → Negative

Review: "The battery life is decent but the camera disappoints." → """

# ─── Chain-of-Thought (CoT) ───────────────────────────────────────────
chain_of_thought = """
Solve the following math problem step by step.

Problem: A store buys 500 items at $3.50 each and sells them at $5.25 each.
If they sell 80% of the items, what is their profit?

Let me think through this step by step:
"""
# The "Let me think through this step by step:" triggers CoT reasoning
# This alone dramatically improves accuracy on reasoning tasks

# ─── Zero-Shot CoT ────────────────────────────────────────────────────
zero_shot_cot = """
Q: If a train travels 240 miles in 3 hours, how long will it take to travel 400 miles?

A: Let's think step by step.
"""
# Adding "Let's think step by step" improves GSM8K accuracy from 17% to 78%!

# ─── Self-Consistency ─────────────────────────────────────────────────
# Generate multiple CoT paths, take majority vote answer
# Improves accuracy by 10-20% on reasoning benchmarks

def self_consistency(question, model, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = model.generate(
            f"{question}\nLet's think step by step.",
            temperature=0.7   # add randomness to get different paths
        )
        answer = extract_answer(response)
        answers.append(answer)
    # Return most common answer
    return max(set(answers), key=answers.count)
```

### Advanced Prompting Techniques

```python
# ─── Role / Persona Prompting ────────────────────────────────────────
role_prompt = """
You are Dr. Sarah Chen, a world-renowned cybersecurity expert with 20 years 
of experience at CISA and MIT's Computer Science department. You communicate 
complex security concepts clearly to both technical and non-technical audiences.

You always:
- Provide practical, actionable advice
- Cite specific CVEs and attack vectors when relevant
- Think about both offensive and defensive perspectives
- Acknowledge uncertainty when it exists
"""

# ─── Tree of Thoughts (ToT) ───────────────────────────────────────────
tot_prompt = """
Solve this problem by exploring multiple solution paths and selecting the best one.

Problem: Design a rate limiting system for an API.

Approach 1 (Token Bucket):
[Explore this approach...]
Pros: ...
Cons: ...

Approach 2 (Sliding Window):
[Explore this approach...]
Pros: ...
Cons: ...

Approach 3 (Fixed Window):
[Explore this approach...]
Pros: ...
Cons: ...

Best approach for our needs: [Select and justify]
"""

# ─── ReAct (Reason + Act) ─────────────────────────────────────────────
react_prompt = """
Answer the following question using the provided tools.
At each step: Thought → Action → Observation → (repeat) → Final Answer

Question: What is the population of the capital city of Australia?

Thought: I need to find Australia's capital city, then find its population.
Action: search("capital of Australia")
Observation: Canberra is the capital of Australia.
Thought: Now I need to find Canberra's population.
Action: search("Canberra population 2024")
Observation: Canberra has approximately 467,194 people (2024).
Thought: I have the answer.
Final Answer: The capital of Australia is Canberra, with a population of approximately 467,194.
"""

# ─── Structured Output Prompting ──────────────────────────────────────
structured_prompt = """
Extract information from the following text and return it as valid JSON only.
Do not include any explanation, markdown, or text outside the JSON.

Text: "Sarah Johnson, a 32-year-old software engineer from Austin, TX,
       joined the company on March 15, 2023. Her employee ID is EMP-4829."

Required JSON schema:
{
  "name": "string",
  "age": "number",
  "occupation": "string",
  "location": "string",
  "join_date": "string (YYYY-MM-DD)",
  "employee_id": "string"
}

JSON:"""

# ─── Constitutional Prompting ─────────────────────────────────────────
constitutional_prompt = """
[Task]
Write marketing copy for our product.

[Draft]
{generate_draft}

[Critique]
Review the draft against these principles:
1. Is it honest and not misleading?
2. Does it avoid unverifiable superlatives?
3. Is it inclusive and respectful?

[Revised Version]
Based on the critique, write an improved version:
"""

# ─── Meta-Prompting ───────────────────────────────────────────────────
meta_prompt = """
You are a prompt engineering expert. Generate an optimal prompt for the following task.

Task to accomplish: Summarize legal contracts to extract key obligations and deadlines.

Requirements:
- The prompt should be used with GPT-4
- Output should be structured and consistent
- Should handle contracts up to 50 pages

Generate the optimal prompt:
"""

# ─── Prompt Chaining ──────────────────────────────────────────────────
async def analyze_document(doc_text):
    # Step 1: Extract key points
    key_points = await llm.generate(f"Extract the 5 most important points from:\n{doc_text}")

    # Step 2: Identify risks
    risks = await llm.generate(f"Identify potential risks in these points:\n{key_points}")

    # Step 3: Generate recommendations
    recommendations = await llm.generate(
        f"Given these risks:\n{risks}\nProvide actionable recommendations:"
    )

    # Step 4: Create executive summary
    summary = await llm.generate(
        f"Create a 3-sentence executive summary:\n"
        f"Points: {key_points}\nRisks: {risks}\nRecommendations: {recommendations}"
    )

    return summary
```

### System Prompts

```python
# System prompts set the behavior, persona, and constraints for the entire conversation

system_prompt = """
## Role
You are an expert software architect and senior engineer with deep expertise in 
distributed systems, cloud architecture, and modern software development practices.

## Capabilities  
- Design scalable, production-ready system architectures
- Review and critique code for correctness, performance, and security
- Explain complex technical concepts at the appropriate level
- Suggest best practices and industry standards

## Behavior Guidelines
1. Always ask clarifying questions before proposing solutions to complex problems
2. Present trade-offs for major architectural decisions
3. Use diagrams (ASCII art) when explaining system designs
4. Cite specific technologies, versions, and alternatives
5. If you're uncertain about something, say so explicitly

## Constraints
- Only recommend production-proven solutions, not experimental ones
- Always consider security implications of any recommendation
- Do not write more than 300 lines of code in a single response

## Output Format
For code: Always include language identifier in code blocks
For architectures: Include a component diagram
For explanations: Use headers, bullets, and examples
"""
```

---

## 7. Fine-Tuning & Training

### Training Stages for LLMs

```
┌─────────────────────────────────────────────────────────────────────┐
│               LLM TRAINING PIPELINE                                  │
│                                                                      │
│  Stage 1: Pre-training                                               │
│  ─────────────────────                                               │
│  Data: Trillions of web tokens (Common Crawl, GitHub, Books...)      │
│  Objective: Next-token prediction (self-supervised)                  │
│  Cost: $1M - $100M+ compute                                          │
│  Result: Base model (knows language, but not how to chat)            │
│                                                                      │
│  Stage 2: Supervised Fine-Tuning (SFT)                               │
│  ──────────────────────────────────────                              │
│  Data: High-quality (prompt, response) pairs curated by humans       │
│  Objective: Imitate helpful assistant behavior                       │
│  Cost: $10K - $1M                                                    │
│  Result: SFT model (follows instructions)                            │
│                                                                      │
│  Stage 3: RLHF / RLAIF / DPO                                        │
│  ──────────────────────────────                                      │
│  Data: Human preference rankings (A is better than B)               │
│  Objective: Align with human values, be helpful/harmless/honest      │
│  Result: Chat model (Claude, ChatGPT, etc.)                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Supervised Fine-Tuning (SFT)

```python
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch

# ─── Prepare dataset ──────────────────────────────────────────────────
training_data = [
    {
        "instruction": "Translate the following English text to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    },
    {
        "instruction": "Summarize this in one sentence.",
        "input": "MongoDB is a NoSQL database that stores data as BSON documents...",
        "output": "MongoDB is a document-oriented NoSQL database for flexible, scalable data storage."
    }
]

def format_prompt(sample):
    """Alpaca-style prompt format"""
    if sample["input"]:
        prompt = f"""### Instruction:
{sample["instruction"]}

### Input:
{sample["input"]}

### Response:
{sample["output"]}"""
    else:
        prompt = f"""### Instruction:
{sample["instruction"]}

### Response:
{sample["output"]}"""
    return {"text": prompt}

# ─── Load model ───────────────────────────────────────────────────────
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ─── Training arguments ───────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./finetuned-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch = 16
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    bf16=True,                        # use bfloat16 precision
    gradient_checkpointing=True,      # trade compute for memory
    optim="adamw_torch_fused",        # faster optimizer
    report_to="wandb"
)
```

### Parameter-Efficient Fine-Tuning (PEFT)

```python
# ─── LoRA (Low-Rank Adaptation) ───────────────────────────────────────
# Key insight: weight updates during fine-tuning have low intrinsic rank
# Instead of updating W (d×d), we learn two small matrices A (d×r) and B (r×d)
# where r << d (rank is much smaller than model dimension)
#
# W' = W + ΔW = W + BA  (r=16 vs d=4096 → 250× fewer parameters!)

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # rank (4, 8, 16, 32, 64 typical)
    lora_alpha=32,                 # scaling factor (usually 2r)
    target_modules=[               # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 33,554,432 || all params: 8,030,261,248 || trainable%: 0.42%

# ─── QLoRA — LoRA on quantized model ─────────────────────────────────
# Quantize model to 4-bit, then apply LoRA
# Allows fine-tuning 70B model on single 48GB GPU!

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # Normal Float 4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True     # nested quantization
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ─── Using TRL's SFTTrainer (easiest path) ───────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=quantized_model,
    args=SFTConfig(
        output_dir="./qlora-model",
        max_seq_length=2048,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
    ),
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./qlora-output")

# ─── Other PEFT Methods ───────────────────────────────────────────────
# DoRA:     Weight-Decomposed LoRA (even better than LoRA on some tasks)
# IA3:      Scales activations, fewer params than LoRA
# Prefix tuning: Prepend trainable tokens to input
# Prompt tuning: Soft prompt — learnable token embeddings
# Adapter:  Small bottleneck layers inserted between transformer layers
```

### RLHF & Alignment

```python
# ─── RLHF Pipeline ────────────────────────────────────────────────────
#
# 1. Collect human preferences
#    Human compares two model responses: "A is better than B"
#
# 2. Train Reward Model (RM)
#    RM learns to score responses according to human preferences
#    Loss: -log σ(r_θ(x, y_w) - r_θ(x, y_l))
#    where y_w = preferred, y_l = rejected
#
# 3. Optimize policy with PPO
#    Maximize: E[r_θ(x, y)] - β * KL(π_RL || π_SFT)
#    KL term prevents model from drifting too far from SFT policy

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import pipeline

# Reward model (trained on human preferences)
reward_model = pipeline("text-classification", model="reward-model-checkpoint")

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,      # β — KL penalty coefficient
    adap_kl_ctrl=True,     # adaptive KL control
    target_kl=6.0
)

# ─── DPO (Direct Preference Optimization) — RLHF without RL ──────────
# Reformulates RLHF as a supervised classification problem
# No reward model needed! No PPO! Much simpler!
# Used by Llama 3, Mistral, many open-source models

from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,                  # temperature for DPO loss
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    loss_type="sigmoid"        # sigmoid, hinge, ipo, kto_pair
)

# Dataset format for DPO:
dpo_dataset = Dataset.from_dict({
    "prompt":   ["Write a poem about nature"],
    "chosen":   ["The trees sway gently in the breeze..."],    # preferred
    "rejected": ["Nature is green trees and blue sky..."]      # not preferred
})

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,   # frozen SFT model (reference policy)
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer
)

# ─── Other Alignment Methods ──────────────────────────────────────────
# RLAIF:  Use AI feedback instead of human feedback (cheaper, scales better)
# ORPO:   Odds Ratio Preference Optimization (no reference model needed)
# SimPO:  Simple Preference Optimization
# KTO:    Kahneman-Tversky Optimization (unpaired preference data)
# SPIN:   Self-Play Fine-Tuning
```

---

## 8. Retrieval-Augmented Generation (RAG)

### Why RAG?

```
Problem with pure LLMs:
✗ Knowledge cutoff (can't answer about recent events)
✗ Hallucinations (makes up facts)
✗ No access to private/proprietary data
✗ Context window limits for large document sets

RAG Solution:
✓ Retrieve relevant documents at query time
✓ Ground answers in retrieved evidence
✓ Works with any up-to-date knowledge source
✓ Easily updatable (just add documents)
```

### Basic RAG Pipeline

```python
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ── Step 1: Load Documents ────────────────────────────────────────────
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

# ── Step 2: Chunk Documents ───────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # characters per chunk
    chunk_overlap=200,      # overlap to maintain context
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# ── Step 3: Create Embeddings & Store ────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# ── Step 4: Create Retriever ──────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",       # Maximum Marginal Relevance (diverse results)
    search_kwargs={"k": 5, "fetch_k": 20}
)

# ── Step 5: Generate with Context ────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",         # stuff/map_reduce/refine/map_rerank
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is the vacation policy?"})
print(result["result"])
print("Sources:", [doc.metadata for doc in result["source_documents"]])
```

### Advanced RAG Techniques

```python
# ─── Chunking Strategies ──────────────────────────────────────────────

# 1. Fixed-size chunking
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 2. Recursive character splitting (recommended default)
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# 3. Semantic chunking — split at topic boundaries
from langchain_experimental.text_splitter import SemanticChunker
splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95
)

# 4. Document-specific (markdown, code, HTML)
from langchain.text_splitter import MarkdownHeaderTextSplitter
headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

# ─── Hybrid Search (Vector + Keyword) ────────────────────────────────
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25 (TF-IDF based keyword search)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Semantic search
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine both (0.5 weight each)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]   # keyword:semantic ratio
)

# ─── HyDE (Hypothetical Document Embeddings) ─────────────────────────
# Generate a hypothetical answer → embed that → search with it
# Bridges gap between question style and document style

from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=OpenAIEmbeddings(),
    custom_prompt=PromptTemplate(
        input_variables=["QUESTION"],
        template="Write a passage that answers this question:\n{QUESTION}"
    )
)

# ─── Query Expansion / Rewriting ─────────────────────────────────────
query_expansion_prompt = """
Generate 3 different versions of this query to improve retrieval.
Original: {query}

Variations (one per line):
1."""

def expand_query(query, llm):
    prompt = query_expansion_prompt.format(query=query)
    variations = llm.predict(prompt).strip().split("\n")
    return [query] + [v.strip("0123456789. ") for v in variations]

# Search with all variations, take union of results (+ dedup)

# ─── Self-RAG (selective retrieval) ──────────────────────────────────
# LLM decides WHEN to retrieve (not always)
# Uses special tokens: [Retrieve], [Relevant], [Supported], [Utility]
# More efficient than always-retrieve RAG

# ─── CRAG (Corrective RAG) ────────────────────────────────────────────
# Evaluate retrieved documents quality
# If poor quality → fall back to web search
# If ambiguous → combine local retrieval + web search

# ─── Contextual Retrieval (Anthropic) ────────────────────────────────
# Before embedding: prepend chunk with its context using LLM
contextual_prompt = """
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a succinct context (2-3 sentences) explaining where this chunk fits.
Context:"""
# Prepend that context to the chunk before embedding → much better retrieval
```

### RAG Evaluation

```python
# ─── RAG Triad Metrics ────────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,           # Is answer grounded in retrieved context?
    answer_relevancy,       # Does answer address the question?
    context_recall,         # Are relevant docs retrieved?
    context_precision,      # Are retrieved docs relevant?
    answer_correctness      # Is the answer factually correct?
)

results = evaluate(
    dataset=test_dataset,   # question, answer, ground_truth, contexts
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]
)

print(results)
# faithfulness:      0.89   (answer supported by context)
# answer_relevancy:  0.92   (answer addresses the question)
# context_recall:    0.85   (relevant docs were retrieved)
# context_precision: 0.78   (retrieved docs are relevant)
```

---

## 9. Vector Databases & Embeddings

### Embeddings Explained

```python
# Embeddings convert semantic meaning into numerical vectors
# Similar meaning → nearby vectors in embedding space

from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

# Example
e1 = get_embedding("The cat sat on the mat")
e2 = get_embedding("A feline rested on a rug")    # similar meaning
e3 = get_embedding("I love eating pizza")          # different meaning

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(e1, e2))   # ~0.89  (very similar)
print(cosine_sim(e1, e3))   # ~0.35  (dissimilar)

# ─── Major Embedding Models ───────────────────────────────────────────
embedding_models = {
    "text-embedding-3-large": {
        "dims": 3072,
        "cost": "$0.00013/1K tokens",
        "provider": "OpenAI"
    },
    "text-embedding-3-small": {
        "dims": 1536,
        "cost": "$0.00002/1K tokens",
        "provider": "OpenAI"
    },
    "embed-english-v3.0": {
        "dims": 1024,
        "provider": "Cohere",
        "note": "Great for RAG"
    },
    "voyage-3": {
        "dims": 1024,
        "provider": "Voyage AI",
        "note": "Top performance"
    },
    "BAAI/bge-large-en-v1.5": {
        "dims": 1024,
        "provider": "Open source (HuggingFace)",
        "cost": "Free"
    },
    "nomic-embed-text": {
        "dims": 768,
        "provider": "Nomic (via Ollama)",
        "cost": "Free, runs locally"
    }
}
```

### Vector Similarity Search

```python
# ─── ANN (Approximate Nearest Neighbor) algorithms ───────────────────

# HNSW (Hierarchical Navigable Small World) — used by most vector DBs
# Build a multi-layer graph, search starts at top (coarse), drills down
# Trade-off: accuracy vs speed via ef_construction and ef_search params

# IVF (Inverted File Index)
# Partition vectors into clusters, search only nearest clusters
# nlist = number of clusters, nprobe = clusters to search

# FAISS (Meta) — best for in-memory search
import faiss
import numpy as np

dimension = 1536
n_vectors = 100000

# Create index
index = faiss.IndexFlatL2(dimension)    # exact search (L2 distance)
# or:
index = faiss.IndexHNSWFlat(dimension, 32)  # HNSW (fast approximate)

# For large datasets, use quantization:
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=1024,      # number of clusters
    m=16,            # number of sub-quantizers
    nbits=8          # bits per sub-quantizer
)
index.train(vectors)   # IVF needs training

# Add vectors
vectors = np.random.random((n_vectors, dimension)).astype('float32')
index.add(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
D, I = index.search(query, k=10)   # k nearest neighbors
# D = distances, I = indices
```

### Major Vector Databases

```python
# ─── Pinecone ─────────────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",    # cosine, euclidean, dotproduct
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-index")

# Upsert vectors (with metadata)
index.upsert(vectors=[
    {
        "id": "doc-001",
        "values": embedding.tolist(),
        "metadata": {
            "text": "original chunk text",
            "source": "handbook.pdf",
            "page": 5,
            "category": "HR"
        }
    }
])

# Query with filter
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    filter={"category": {"$eq": "HR"}},
    include_metadata=True
)

# ─── Weaviate ─────────────────────────────────────────────────────────
import weaviate

client = weaviate.Client("http://localhost:8080")

# Define schema
client.schema.create_class({
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]}
    ]
})

# Hybrid search (vector + BM25)
result = client.query.get("Document", ["content", "source"]) \
    .with_hybrid(query="machine learning", alpha=0.5) \
    .with_limit(5) \
    .do()

# ─── Qdrant ───────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=embedding.tolist(),
            payload={"text": "chunk text", "source": "file.pdf"}
        )
    ]
)

# Filtered search
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=query_embedding.tolist(),
    query_filter=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value="file.pdf"))]
    ),
    limit=5
)

# ─── ChromaDB (local, great for development) ──────────────────────────
import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=["doc1", "doc2"],
    embeddings=[emb1.tolist(), emb2.tolist()],
    documents=["chunk text 1", "chunk text 2"],
    metadatas=[{"source": "file1.pdf"}, {"source": "file2.pdf"}]
)

results = collection.query(
    query_embeddings=[query_emb.tolist()],
    n_results=5,
    where={"source": "file1.pdf"}    # metadata filter
)
```

---

## 10. AI Agents & Agentic Systems

### What are AI Agents?

```
AI Agent = LLM + Tools + Memory + Planning + Execution

Unlike a simple chatbot (one-shot response), an agent:
- Plans multi-step approaches to complex goals
- Takes actions (call APIs, browse web, write code, etc.)
- Observes results and adapts
- Loops until goal is achieved
```

### Building Agents from Scratch

```python
from openai import OpenAI
import json

client = OpenAI()

# ─── Define Tools ──────────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    }
]

# ─── Tool Execution ────────────────────────────────────────────────────
def execute_tool(tool_name, tool_args):
    if tool_name == "search_web":
        return search_the_web(tool_args["query"])   # your implementation
    elif tool_name == "execute_python":
        return run_python_code(tool_args["code"])
    elif tool_name == "read_file":
        with open(tool_args["path"]) as f:
            return f.read()

# ─── Agent Loop ────────────────────────────────────────────────────────
def run_agent(user_request, max_iterations=10):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with access to tools. Use them to accomplish the user's request."},
        {"role": "user", "content": user_request}
    ]

    for iteration in range(max_iterations):
        # LLM decides next action
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"   # "auto", "none", or specific tool
        )

        message = response.choices[0].message
        messages.append(message)

        # Check if done (no tool calls)
        if not message.tool_calls:
            return message.content

        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 Calling: {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            print(f"📤 Result: {result[:200]}...")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "Max iterations reached without completing the task."

# Run the agent
result = run_agent("Find the current weather in Tokyo and write a Python script to convert it to Fahrenheit")
```

### Multi-Agent Systems

```python
from typing import List, Dict, Any
import asyncio

# ─── Agent Architectures ──────────────────────────────────────────────

# 1. Sequential (Pipeline)
# Research Agent → Writer Agent → Editor Agent → Publisher Agent

# 2. Hierarchical (Orchestrator + Workers)
# Manager Agent
#   ├── Research Agent
#   ├── Analysis Agent
#   └── Writer Agent

# 3. Collaborative (Peer-to-peer)
# All agents can communicate with each other

# ─── CrewAI Framework ─────────────────────────────────────────────────
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate and comprehensive information on any given topic",
    backstory="You are an expert researcher with access to vast information sources. You are thorough, analytical, and always cite your sources.",
    tools=[search_tool, web_scraper_tool],
    llm="gpt-4o",
    verbose=True,
    max_iterations=10
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging, clear, and accurate articles based on research",
    backstory="You are a skilled writer who transforms complex research into readable content for general audiences.",
    llm="gpt-4o",
    verbose=True
)

editor = Agent(
    role="Senior Editor",
    goal="Polish and improve content quality, ensure accuracy and style consistency",
    backstory="You are a meticulous editor with an eye for detail and high standards.",
    llm="gpt-4o"
)

# Define tasks
research_task = Task(
    description="Research the latest developments in quantum computing in 2024. Focus on practical applications and recent breakthroughs.",
    expected_output="A comprehensive research report with key findings, statistics, and sources",
    agent=researcher
)

writing_task = Task(
    description="Using the research provided, write a 1500-word article about quantum computing for a technical but non-specialist audience.",
    expected_output="A well-structured, engaging article with introduction, body sections, and conclusion",
    agent=writer,
    context=[research_task]   # depends on research task
)

editing_task = Task(
    description="Review and edit the article for clarity, accuracy, grammar, and engagement. Provide the final polished version.",
    expected_output="The final, publication-ready article",
    agent=editor,
    context=[writing_task]
)

# Create crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,   # sequential or hierarchical
    verbose=True
)

result = crew.kickoff(inputs={"topic": "quantum computing"})
```

### Agent Memory Systems

```python
# ─── Types of Agent Memory ────────────────────────────────────────────

class AgentMemory:
    """
    Short-term memory:  Current conversation context (in-context)
    Long-term memory:   Persisted facts, experiences (vector DB)
    Episodic memory:    Past conversation summaries
    Procedural memory:  How to do things (tools, workflows)
    Semantic memory:    World knowledge (pre-training)
    """

    def __init__(self, llm, vectorstore):
        self.conversation_history = []     # short-term
        self.vectorstore = vectorstore     # long-term
        self.episode_summaries = []        # episodic

    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

        # Extract and store important facts to long-term memory
        if len(self.conversation_history) % 10 == 0:
            self._compress_to_episodic()

    def _compress_to_episodic(self):
        """Summarize recent conversation and store as episodic memory"""
        recent = self.conversation_history[-10:]
        summary = self.llm.summarize(recent)
        self.vectorstore.add(summary, metadata={"type": "episode", "ts": now()})
        self.episode_summaries.append(summary)

    def recall(self, query, k=5):
        """Retrieve relevant memories"""
        return self.vectorstore.search(query, k=k)

    def get_context(self, query):
        """Build full context: recent history + relevant memories"""
        relevant_memories = self.recall(query)
        return {
            "history": self.conversation_history[-20:],   # last 20 messages
            "relevant_past": relevant_memories
        }
```

### Popular Agent Frameworks

```python
# ─── LangGraph — graph-based agent orchestration ──────────────────────
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str

def researcher_node(state):
    # Research agent logic
    return {"messages": [research_result], "next": "writer"}

def writer_node(state):
    # Writer agent logic
    return {"messages": [written_content], "next": "editor"}

def editor_node(state):
    result = edit_content(state["messages"])
    if needs_revision(result):
        return {"messages": [feedback], "next": "writer"}  # loop back!
    return {"messages": [final], "next": END}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("editor", lambda s: s["next"])
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage("Write about AI")]})

# ─── AutoGen — conversational multi-agent ─────────────────────────────
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

assistant = AssistantAgent(
    name="Assistant",
    llm_config={"model": "gpt-4o"},
    system_message="You are a helpful AI assistant."
)

code_executor = UserProxyAgent(
    name="CodeExecutor",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "./workspace", "use_docker": True}
)

code_executor.initiate_chat(
    assistant,
    message="Write and run a Python script that generates prime numbers up to 100"
)
```

---

## 11. Image Generation Models

### Diffusion Models

```python
# ─── How Diffusion Models Work ────────────────────────────────────────
#
# Forward process: add Gaussian noise to image over T steps
# img_T ~ N(0, I)  (pure noise)
#
# Reverse process: learn to denoise step by step
# Neural network ε_θ predicts the noise at each step
# Sampling: start from noise, denoise T times → image
#
# DDPM (Denoising Diffusion Probabilistic Models)
# DDIM (Deterministic, fewer steps needed)
# DPM-Solver (even fewer steps, 20-50 vs 1000)

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# ─── Stable Diffusion ─────────────────────────────────────────────────
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Generate image
image = pipe(
    prompt="A majestic golden dragon soaring over misty mountains, fantasy art, highly detailed",
    negative_prompt="blurry, low quality, bad anatomy, watermark",
    num_inference_steps=25,      # denoising steps (20-50 typical)
    guidance_scale=7.5,          # CFG scale: how strongly to follow prompt
    width=768, height=512,
    generator=torch.manual_seed(42)   # reproducibility
).images[0]

image.save("dragon.png")

# ─── SDXL (Stable Diffusion XL) ──────────────────────────────────────
from diffusers import StableDiffusionXLPipeline

pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# ─── FLUX.1 (2024, state of the art) ─────────────────────────────────
from diffusers import FluxPipeline

pipe_flux = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe_flux(
    "A beautiful woman reading a book in a sunlit library",
    height=1024, width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]

# ─── Image-to-Image ───────────────────────────────────────────────────
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("sketch.png").resize((768, 512))

image = pipe_i2i(
    prompt="A professional logo based on this sketch",
    image=init_image,
    strength=0.6,      # how much to transform (0=keep original, 1=ignore)
    guidance_scale=7.5
).images[0]

# ─── ControlNet ───────────────────────────────────────────────────────
# Condition generation on: edges, poses, depth maps, segmentation, etc.
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from controlnet_aux import CannyDetector

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe_ctrl = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Extract edges
detector = CannyDetector()
control_image = detector(Image.open("reference.png"))

image = pipe_ctrl(
    prompt="A beautiful landscape painting",
    image=control_image,
    controlnet_conditioning_scale=0.8
).images[0]
```

### DALL-E & Midjourney APIs

```python
# ─── DALL-E 3 (OpenAI) ───────────────────────────────────────────────
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A photorealistic image of a futuristic city with flying cars at sunset",
    size="1792x1024",      # 1024x1024, 1024x1792, 1792x1024
    quality="hd",          # standard or hd
    style="vivid",         # vivid or natural
    n=1
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt   # DALL-E modifies your prompt

# DALL-E 3 also does image edits and variations
response = client.images.edit(
    model="dall-e-2",
    image=open("photo.png", "rb"),
    mask=open("mask.png", "rb"),     # transparent = area to edit
    prompt="Add a golden crown on the cat's head",
    n=1, size="1024x1024"
)

# ─── Stable Diffusion via API (Stability AI) ─────────────────────────
import requests

response = requests.post(
    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "text_prompts": [
            {"text": "A tranquil forest scene", "weight": 1},
            {"text": "blurry, low quality", "weight": -1}   # negative prompt
        ],
        "cfg_scale": 7,
        "width": 1024, "height": 1024,
        "steps": 30, "samples": 1
    }
)
```

### Key Concepts in Image Generation

```python
image_generation_concepts = {

    "cfg_scale": {
        "name": "Classifier-Free Guidance Scale",
        "range": "1-20 typical",
        "low_1_5": "Image follows distribution, ignores prompt",
        "medium_7": "Balanced (recommended)",
        "high_15+": "Strongly follows prompt, can oversaturate",
        "formula": "output = uncond + scale * (cond - uncond)"
    },

    "negative_prompt": {
        "description": "What to exclude from the image",
        "common": "blurry, low quality, bad anatomy, extra limbs, watermark, text"
    },

    "lora_for_images": {
        "description": "Fine-tune diffusion model for specific style/person/object",
        "use_cases": ["Consistent character", "Art style", "Product shots"],
        "training_images": "15-50 images",
        "trigger_word": "custom keyword to activate LoRA"
    },

    "textual_inversion": {
        "description": "Learn a new 'word' (token) for a concept",
        "lighter_than": "LoRA",
        "example": "S* = your specific person/object/style"
    },

    "inpainting": {
        "description": "Edit specific masked region of image",
        "use_cases": ["Remove objects", "Replace backgrounds", "Fill gaps"]
    },

    "outpainting": {
        "description": "Extend image beyond its borders",
        "also_called": "image extension"
    },

    "upscaling": {
        "description": "Increase resolution with AI",
        "tools": ["Real-ESRGAN", "ESRGAN", "Topaz Gigapixel", "SDXL Refiner"]
    }
}
```

---

## 12. Audio & Speech Generation

```python
# ─── Text-to-Speech (TTS) ─────────────────────────────────────────────

# OpenAI TTS
from openai import OpenAI
from pathlib import Path

client = OpenAI()

speech = client.audio.speech.create(
    model="tts-1-hd",          # tts-1 (fast) or tts-1-hd (quality)
    voice="nova",               # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! Welcome to our AI-powered application.",
    response_format="mp3",      # mp3, opus, aac, flac, wav, pcm
    speed=1.0                   # 0.25 to 4.0
)

speech.stream_to_file(Path("output.mp3"))

# ─── Voice Cloning (ElevenLabs) ──────────────────────────────────────
from elevenlabs import ElevenLabs, Voice, VoiceSettings

client = ElevenLabs(api_key="YOUR_API_KEY")

# Clone a voice
voice = client.clone(
    name="My Custom Voice",
    files=["voice_sample_1.mp3", "voice_sample_2.mp3"],
    description="Professional narrator voice"
)

# Generate speech with cloned voice
audio = client.generate(
    text="The future of AI is both exciting and challenging.",
    voice=voice,
    model="eleven_turbo_v2"
)

# ─── Speech-to-Text (Whisper) ─────────────────────────────────────────
# OpenAI Whisper — state of the art ASR, 99 languages
import whisper

model = whisper.load_model("large-v3")   # tiny, base, small, medium, large

# Transcribe
result = model.transcribe("recording.mp3")
print(result["text"])

# With word-level timestamps
result = model.transcribe(
    "recording.mp3",
    word_timestamps=True,
    language="en"     # or None for auto-detect
)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s → {segment['end']:.2f}s]: {segment['text']}")

# Via API
with open("recording.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

# ─── Music Generation ─────────────────────────────────────────────────
# MusicGen (Meta) — open source
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("facebook/musicgen-large")
model.set_generation_params(duration=30)   # 30 seconds

descriptions = ["An upbeat jazz piano with light percussion and bass"]
wav = model.generate(descriptions)
audio_write("output_music", wav[0].cpu(), model.sample_rate, strategy="loudness")

# Suno, Udio — text-to-music via web API (commercial)
```

---

## 13. Video Generation

```python
# Video generation models work with temporal consistency across frames
# Key challenge: maintaining coherence through time

video_models = {
    "Sora (OpenAI)": {
        "architecture": "Video Diffusion Transformer (DiT)",
        "max_length": "60 seconds",
        "resolution": "Up to 1080p",
        "key_tech": "Spacetime patches, temporal attention",
        "access": "Limited API"
    },
    "Runway Gen-3 Alpha": {
        "max_length": "10 seconds",
        "access": "API available",
        "strengths": "High quality, follows prompts well"
    },
    "Stable Video Diffusion": {
        "type": "Image-to-video",
        "frames": "14-25 frames",
        "open_source": True
    },
    "CogVideoX": {
        "type": "Text-to-video",
        "open_source": True,
        "frames": "49 frames at 8fps"
    }
}

# ─── Image-to-Video with SVD ──────────────────────────────────────────
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16
)

image = load_image("car.png")
image = image.resize((1024, 576))

frames = pipe(
    image,
    num_frames=25,
    fps=7,
    noise_aug_strength=0.02,
    motion_bucket_id=127    # 1-255: lower=less motion, higher=more motion
).frames[0]

export_to_video(frames, "car_driving.mp4", fps=7)

# ─── Runway Gen-3 API ────────────────────────────────────────────────
import requests

response = requests.post(
    "https://api.runwayml.com/v1/image_to_video",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "promptImage": image_url,
        "promptText": "A car driving down a scenic mountain road",
        "model": "gen3a_turbo",
        "duration": 5,       # 5 or 10 seconds
        "ratio": "1280:768"
    }
)
task_id = response.json()["id"]
```

---

## 14. Multimodal AI

### Vision-Language Models

```python
# ─── GPT-4 Vision ────────────────────────────────────────────────────
from openai import OpenAI
import base64

client = OpenAI()

# Encode image
with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                        "detail": "high"    # low, high, auto
                    }
                },
                {
                    "type": "text",
                    "text": "Analyze this chart. What are the key trends? Extract all numerical values."
                }
            ]
        }
    ],
    max_tokens=1000
)

# Multiple images
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/before.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/after.jpg"}},
            {"type": "text", "text": "What changed between these two images?"}
        ]
    }]
)

# ─── Claude Vision (Anthropic) ───────────────────────────────────────
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in detail."
                }
            ]
        }
    ]
)

# ─── LLaVA (Open Source Vision-Language Model) ───────────────────────
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16, device_map="auto"
)

image = Image.open("scene.jpg")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is happening in this image?"}
        ]
    }
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=500)
result = processor.decode(output[0], skip_special_tokens=True)

# ─── Document Understanding ───────────────────────────────────────────
# Extract text, tables, forms from documents (PDFs, invoices, etc.)

# Using Claude for document analysis
with open("invoice.pdf", "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}
            },
            {"type": "text", "text": "Extract: invoice number, date, total amount, line items as JSON"}
        ]
    }]
)
```

---

## 15. Model Evaluation & Benchmarks

### Common Benchmarks

```python
benchmarks = {
    "reasoning": {
        "MMLU": {
            "description": "Massive Multitask Language Understanding",
            "tasks": "57 subjects (math, law, medicine, etc.)",
            "metric": "Accuracy",
            "top_models": "GPT-4o: 88.7%, Claude 3.5: 88.3%"
        },
        "HellaSwag": {
            "description": "Commonsense NLI — choose correct sentence ending",
            "metric": "Accuracy"
        },
        "ARC-Challenge": {
            "description": "Grade school science questions",
            "metric": "Accuracy"
        },
        "WinoGrande": {
            "description": "Commonsense reasoning with ambiguous pronouns",
            "metric": "Accuracy"
        }
    },
    "math_coding": {
        "GSM8K": "Grade school math word problems",
        "MATH": "Competition math (very hard)",
        "HumanEval": "Python coding — pass@k metric",
        "MBPP": "Mostly Basic Programming Problems",
        "SWE-bench": "Real GitHub issues — agent based"
    },
    "safety": {
        "TruthfulQA": "Does model produce truthful answers?",
        "BBQ": "Bias Benchmark for QA",
        "HarmBench": "Adversarial jailbreak resistance"
    },
    "long_context": {
        "RULER": "Long context understanding",
        "NIAH": "Needle In A Haystack — find specific info in long context",
        "LongBench": "Diverse long-form tasks"
    },
    "instruction_following": {
        "MT-Bench": "Multi-turn conversation quality",
        "AlpacaEval": "Win rate vs GPT-4",
        "IFEval": "Instruction Following Evaluation"
    }
}

# ─── LLM-as-Judge ─────────────────────────────────────────────────────
# Use a powerful LLM to evaluate other LLM outputs
judge_prompt = """
You are an expert evaluator. Rate the following response on a scale of 1-10
for each criterion, then provide an overall score.

Question: {question}
Response: {response}

Criteria:
1. Accuracy (1-10): Is the information correct?
2. Completeness (1-10): Does it fully address the question?
3. Clarity (1-10): Is it clear and well-explained?
4. Conciseness (1-10): Is it appropriately concise?

Provide scores as JSON:
{"accuracy": X, "completeness": X, "clarity": X, "conciseness": X, "overall": X, "reasoning": "..."}
"""
```

---

## 16. AI Safety & Alignment

### Core Safety Concepts

```python
safety_concepts = {

    "hallucination": {
        "definition": "Model generates false information confidently",
        "types": [
            "Factual hallucination: Wrong facts",
            "Faithfulness: Contradicts provided context",
            "Logical: Self-contradictory reasoning"
        ],
        "mitigation": ["RAG", "Constitutional AI", "RLHF", "Fact-checking"]
    },

    "alignment": {
        "definition": "Model's objectives match human intentions and values",
        "goal": "Helpful, Harmless, Honest (HHH — Anthropic framework)",
        "challenges": [
            "Reward hacking: Optimizing for proxy metric not true goal",
            "Inner misalignment: Trained vs deployed behavior differ",
            "Scalable oversight: Can't evaluate superhuman AI performance"
        ]
    },

    "constitutional_ai": {
        "by": "Anthropic",
        "description": "Model critiques and revises its own outputs against a constitution",
        "steps": [
            "1. Generate response",
            "2. Critique against principles",
            "3. Revise based on critique",
            "4. RLAIF with AI feedback instead of human"
        ]
    },

    "jailbreaking": {
        "definition": "Bypassing safety guardrails",
        "methods": [
            "Role-playing (pretend you're DAN)",
            "Many-shot jailbreaking (many harmful examples in context)",
            "Prompt injection (instructions in retrieved data)",
            "Adversarial suffixes (appended token sequences)"
        ],
        "defenses": ["Input filtering", "Output monitoring", "Constitutional prompts"]
    },

    "bias": {
        "types": ["Gender bias", "Racial bias", "Cultural bias", "Confirmation bias"],
        "sources": ["Training data", "RLHF rater preferences", "Fine-tuning data"],
        "evaluation": ["WinoBias", "BBQ", "StereoSet"]
    },

    "privacy": {
        "risks": ["PII in training data", "Memorization", "Data extraction attacks"],
        "protections": ["Differential privacy", "Data filtering", "GDPR compliance"]
    }
}

# ─── Guardrails Implementation ────────────────────────────────────────
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, DetectPII, ValidJSON

# Input guardrails
guard = Guard().use_many(
    ToxicLanguage(on_fail=OnFailAction.EXCEPTION),
    DetectPII(pii_entities=["EMAIL", "PHONE", "SSN"], on_fail=OnFailAction.FIX)
)

# Output guardrails
output_guard = Guard().use_many(
    ValidJSON(on_fail=OnFailAction.REASK),
    ToxicLanguage(on_fail=OnFailAction.FILTER)
)

# Validate LLM output
validated = output_guard.parse(llm_output, llm_api=openai.chat.completions.create)

# ─── NeMo Guardrails (NVIDIA) ─────────────────────────────────────────
# Topical rails:    Keep conversation on allowed topics
# Safety rails:     Block harmful content
# Security rails:   Prevent prompt injection
# Dialog rails:     Control conversation flow
```

---

## 17. Inference Optimization

### Quantization

```python
# ─── Model Quantization — reduce memory & increase speed ─────────────
# FP32: 4 bytes/param  (full precision, baseline)
# FP16: 2 bytes/param  (half precision, good quality)
# BF16: 2 bytes/param  (better range than FP16, preferred for training)
# INT8: 1 byte/param   (good quality, 2× smaller)
# INT4: 0.5 bytes/param (small, some quality loss)
# NF4:  0.5 bytes/param (best 4-bit for LLMs)
# GGUF: variable (CPU quantization, used by llama.cpp)

# Model size formula:
# params × bytes_per_param = model_size_bytes
# Llama 3 70B in FP16: 70B × 2 = 140GB
# Llama 3 70B in INT4: 70B × 0.5 = 35GB

# ─── bitsandbytes (in-place quantization) ────────────────────────────
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit (QLoRA-style)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# ─── GGUF / llama.cpp (CPU inference) ────────────────────────────────
# GGUF format allows running LLMs on CPU with various quantization levels
# Q2_K: lowest quality, smallest
# Q4_K_M: good balance (recommended)
# Q5_K_M: better quality
# Q8_0: near full quality

from llama_cpp import Llama

model = Llama(
    model_path="llama-3.1-8b-q4_k_m.gguf",
    n_ctx=4096,        # context window
    n_gpu_layers=35,   # layers to offload to GPU (0 = CPU only)
    n_threads=8,       # CPU threads
    verbose=False
)

output = model.create_chat_completion(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500, temperature=0.7
)

# ─── AWQ (Activation-aware Weight Quantization) ───────────────────────
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized("Qwen/Qwen2-7B-Instruct-AWQ")
# AWQ often outperforms GPTQ at same bit-width

# ─── ExLlamaV2 ────────────────────────────────────────────────────────
# Fastest quantized inference on consumer GPUs
# Supports 2-8 bit quantization
```

### Inference Servers

```python
# ─── vLLM — production inference server ──────────────────────────────
# Key innovation: PagedAttention — manage KV cache like virtual memory
# Achieves near-theoretical max throughput

# bash: pip install vllm
# bash: python -m vllm.entrypoints.openai.api_server \
#          --model meta-llama/Llama-3.1-8B-Instruct \
#          --quantization awq \
#          --max-model-len 8192 \
#          --tensor-parallel-size 2   # multi-GPU

# Use OpenAI-compatible API
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)

# ─── Ollama — easy local LLM serving ─────────────────────────────────
# bash: ollama pull llama3.1:8b
# bash: ollama serve

import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Why is the sky blue?"}]
)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="llama3.1", messages=[...], stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# ─── Batching Strategies ──────────────────────────────────────────────
# Continuous batching (vLLM): add new requests while others are running
# Dynamic batching: group requests with similar lengths
# Chunked prefill: process prompt in chunks to reduce latency

# ─── Speculative Decoding ─────────────────────────────────────────────
# Small draft model (e.g., 1B) proposes K tokens
# Large target model verifies all K in parallel
# If accepted: K tokens in price of 1
# Speedup: 2-4× with same output quality

# ─── Tensor Parallelism ───────────────────────────────────────────────
# Split model across multiple GPUs
# Each GPU holds subset of attention heads / MLP neurons
# vLLM --tensor-parallel-size 4   # for 4 GPUs
```

---

## 18. LangChain Framework

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

# ─── LCEL (LangChain Expression Language) ─────────────────────────────
# Chains built with | operator (pipe)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Basic chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "Python programming"})

# RAG chain
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
Answer based only on the following context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is our refund policy?")

# ─── Conversation Memory ──────────────────────────────────────────────
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

memory_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain_with_history = RunnableWithMessageHistory(
    chat_prompt | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Multi-turn conversation
chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "user-123"}}
)
chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user-123"}}
)
# Returns: "Your name is Alice."

# ─── Structured Output ────────────────────────────────────────────────
from pydantic import BaseModel, Field

class ProductInfo(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Whether item is in stock")
    features: list[str] = Field(description="Key features")

structured_llm = llm.with_structured_output(ProductInfo)
product = structured_llm.invoke("Tell me about the iPhone 15 Pro")
print(product.name, product.price)

# ─── Streaming ────────────────────────────────────────────────────────
for chunk in chain.stream({"topic": "databases"}):
    print(chunk, end="", flush=True)

# ─── Async & Batch ────────────────────────────────────────────────────
import asyncio

async def async_example():
    result = await chain.ainvoke({"topic": "AI"})
    return result

# Batch — process multiple inputs efficiently
results = chain.batch([
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"}
], config={"max_concurrency": 3})

# ─── Callbacks & Tracing ──────────────────────────────────────────────
from langchain.callbacks import LangChainTracer
from langsmith import Client

# LangSmith integration (automatic with env var)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "my-rag-project"
# All chain invocations are now traced in LangSmith dashboard
```

---

## 19. LlamaIndex Framework

```python
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    Settings, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer

# ─── Global Settings ──────────────────────────────────────────────────
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 128000

# ─── Build Index ──────────────────────────────────────────────────────
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Persist index
index.storage_context.persist(persist_dir="./storage")

# Load persisted index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ─── Query Engine ─────────────────────────────────────────────────────
# Simple query engine
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What are the main features?")
print(response)

# Custom retriever + postprocessor
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
response_synthesizer = get_response_synthesizer(response_mode="compact")

custom_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)

# ─── Sub-Question Query Engine ────────────────────────────────────────
# Breaks complex questions into sub-questions, answers each, combines
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

tools = [
    QueryEngineTool(
        query_engine=q_engine,
        metadata=ToolMetadata(name="company_docs", description="Company documentation")
    )
]

sub_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools)
response = sub_engine.query("Compare Q1 and Q2 performance and identify key trends")

# ─── Router Query Engine ──────────────────────────────────────────────
# Routes questions to the right index/tool automatically
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools
)

# ─── Chat Engine ──────────────────────────────────────────────────────
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    verbose=True
)

response = chat_engine.chat("Tell me about the product")
response = chat_engine.chat("How does that compare to last year?")  # uses history

# ─── Agents with LlamaIndex ───────────────────────────────────────────
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=add)
]

agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True)
response = agent.chat("What is 25 multiplied by 4, then add 15?")
```

---

## 20. OpenAI API Deep Dive

```python
from openai import OpenAI, AsyncOpenAI
import json

client = OpenAI(api_key="YOUR_API_KEY")

# ─── Chat Completions ─────────────────────────────────────────────────
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain neural networks in simple terms."}
    ],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    frequency_penalty=0.0,   # reduce word repetition (-2 to 2)
    presence_penalty=0.0,    # encourage new topics (-2 to 2)
    stop=["\n\n", "###"],    # stop sequences
    logprobs=True,           # return token probabilities
    top_logprobs=5,          # top N token alternatives
    seed=42,                 # for reproducibility
    response_format={"type": "text"}   # text or json_object
)

print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")

# ─── Streaming ────────────────────────────────────────────────────────
with client.chat.completions.stream(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about AI"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# ─── Function Calling (Tool Use) ──────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    weather_result = get_weather(args["city"])

    # Continue conversation with tool result
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": weather_result}
    ]
    final_response = client.chat.completions.create(model="gpt-4o", messages=messages)

# ─── Structured Outputs (100% reliable JSON) ──────────────────────────
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "Alice and Bob meeting on Dec 5 to discuss AI roadmap"}
    ],
    response_format=CalendarEvent
)

event = response.choices[0].message.parsed
print(event.name, event.date, event.participants)

# ─── Embeddings ───────────────────────────────────────────────────────
embeddings = client.embeddings.create(
    model="text-embedding-3-large",
    input=["First document", "Second document"],
    dimensions=1536   # reduce from 3072 for efficiency (Matryoshka embeddings)
)

# ─── Assistants API ───────────────────────────────────────────────────
# Managed stateful conversation with built-in tools

assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="You analyze data and provide insights. Always use code interpreter for calculations.",
    model="gpt-4o",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"}
    ]
)

thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id, role="user",
    content="Analyze this CSV and show me the top 5 categories by revenue"
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)

# ─── Batch API (50% cheaper, async) ──────────────────────────────────
import jsonlines

# Create batch file
with open("batch_requests.jsonl", "w") as f:
    for i, query in enumerate(queries):
        f.write(json.dumps({
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 500
            }
        }) + "\n")

# Submit batch
with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Check status & retrieve results
batch_status = client.batches.retrieve(batch.id)
if batch_status.status == "completed":
    results = client.files.content(batch_status.output_file_id)
```

---

## 21. Anthropic Claude API

```python
import anthropic

client = anthropic.Anthropic(api_key="YOUR_API_KEY")

# ─── Basic Message ────────────────────────────────────────────────────
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are an expert software engineer with deep knowledge of distributed systems.",
    messages=[
        {"role": "user", "content": "Explain CAP theorem with examples"}
    ]
)
print(message.content[0].text)

# ─── System Prompts & Multi-turn ──────────────────────────────────────
conversation = [
    {"role": "user",      "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user",      "content": "What's my name?"}
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=200,
    messages=conversation
)

# ─── Streaming ────────────────────────────────────────────────────────
with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Write a Python web scraper"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# ─── Tool Use ─────────────────────────────────────────────────────────
tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g., AAPL)"},
                "currency": {"type": "string", "default": "USD"}
            },
            "required": ["ticker"]
        }
    }
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1000,
    tools=tools,
    messages=[{"role": "user", "content": "What's Apple's stock price?"}]
)

# Handle tool use
if response.stop_reason == "tool_use":
    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    result = get_stock_price(tool_use_block.input["ticker"])

    # Continue with tool result
    final = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's Apple's stock price?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": str(result)
                }]
            }
        ]
    )

# ─── Extended Thinking (Claude 3.7+) ─────────────────────────────────
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve: x³ + 2x² - 5x - 6 = 0"}]
)

for block in response.content:
    if block.type == "thinking":
        print("THINKING:", block.thinking)   # internal reasoning
    elif block.type == "text":
        print("ANSWER:", block.text)

# ─── Vision ──────────────────────────────────────────────────────────
import base64

with open("diagram.png", "rb") as f:
    img_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_data}},
            {"type": "text", "text": "Describe this architecture diagram in detail"}
        ]
    }]
)

# ─── Prompt Caching (reduce cost up to 90%) ───────────────────────────
# Cache large, reusable content (system prompt, docs) — charged once
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1000,
    system=[
        {
            "type": "text",
            "text": "You are an expert analyst.",
        },
        {
            "type": "text",
            "text": very_long_document,  # 100K tokens cached here
            "cache_control": {"type": "ephemeral"}   # mark for caching
        }
    ],
    messages=[{"role": "user", "content": "Summarize the key risks"}]
)
print(response.usage.cache_creation_input_tokens)  # tokens cached
print(response.usage.cache_read_input_tokens)       # tokens from cache
```

---

## 22. Hugging Face Ecosystem

```python
# ─── Transformers — the core library ──────────────────────────────────
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline, Pipeline
)

# High-level pipeline API
# Task types: text-generation, summarization, translation,
#             sentiment-analysis, ner, question-answering,
#             fill-mask, zero-shot-classification, image-classification,
#             image-to-text, text-to-image, speech-recognition, etc.

# Text generation
generator = pipeline("text-generation", model="gpt2", device="cuda")
result = generator("The future of AI is", max_length=100, num_return_sequences=3)

# Sentiment analysis
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier(["I love this product!", "This is terrible quality."])

# Named Entity Recognition
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
entities = ner("Apple CEO Tim Cook visited London last Tuesday.")

# Zero-shot classification (no training needed for new labels)
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = zero_shot(
    "This is a tutorial about machine learning",
    candidate_labels=["technology", "sports", "cooking", "science"]
)

# ─── Tokenizers deep dive ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Chat formatting (apply_chat_template)
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are...

# Batch tokenization
inputs = tokenizer(
    ["First text", "Second text"],
    padding=True,             # pad to same length
    truncation=True,          # truncate if too long
    max_length=512,
    return_tensors="pt"
)
# inputs: {input_ids, attention_mask}

# ─── Datasets ─────────────────────────────────────────────────────────
from datasets import load_dataset, Dataset, DatasetDict

# Load from Hub
dataset = load_dataset("squad", split="train")
dataset = load_dataset("json", data_files="data.jsonl")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Dataset operations
dataset = dataset.filter(lambda x: len(x["text"]) > 100)
dataset = dataset.map(lambda x: {"tokens": tokenizer(x["text"])["input_ids"]})
dataset = dataset.select(range(1000))           # first 1000 examples
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1)

# Create from dict
custom_dataset = Dataset.from_dict({
    "text": ["example 1", "example 2"],
    "label": [0, 1]
})

# Push to Hub
dataset.push_to_hub("username/my-dataset", token="hf_token")

# ─── Model Hub ────────────────────────────────────────────────────────
from huggingface_hub import HfApi, login

login(token="hf_token")

api = HfApi()

# List models
models = api.list_models(task="text-generation", library="transformers", limit=10)

# Download specific file
api.hf_hub_download(repo_id="meta-llama/Llama-3.1-8B", filename="config.json")

# Upload model
api.upload_folder(
    folder_path="./my-finetuned-model",
    repo_id="username/my-model",
    repo_type="model"
)

# ─── Inference API (serverless) ───────────────────────────────────────
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({"inputs": "The quick brown fox"})
```

---

## 23. Local LLMs & Ollama

```python
# ─── Ollama — run any model locally ──────────────────────────────────
# Install: curl -fsSL https://ollama.com/install.sh | sh
# Pull model: ollama pull llama3.1:8b
# Pull vision: ollama pull llava:13b
# List models: ollama list
# Run CLI: ollama run mistral

import ollama

# Chat
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string"}
    ],
    options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 4096,      # context window
        "num_predict": 500    # max tokens
    }
)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(model="llama3.1", messages=[...], stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# Generate embeddings
embedding = ollama.embeddings(model="nomic-embed-text", prompt="Hello world")
vector = embedding["embedding"]   # list of floats

# Vision (multimodal models)
response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": "Describe this image",
        "images": ["./photo.jpg"]
    }]
)

# Custom Modelfile (fine-tune prompt format)
modelfile = """
FROM llama3.1:8b
PARAMETER temperature 0.8
PARAMETER top_p 0.9
SYSTEM You are a SQL expert. Always explain your queries. Format output as:
SQL: <query>
EXPLANATION: <explanation>
"""
ollama.create(model="sql-expert", modelfile=modelfile)

# ─── OpenAI-compatible endpoint ───────────────────────────────────────
from openai import OpenAI

# Ollama exposes an OpenAI-compatible API at port 11434
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Explain gradient descent"}]
)

# ─── Popular Models for Local Use ────────────────────────────────────
local_models = {
    "llama3.1:8b":      "Best all-around 8B (Meta)",
    "llama3.1:70b":     "Near-GPT4 quality, needs 48GB VRAM",
    "mistral:7b":       "Great for coding and reasoning",
    "gemma2:9b":        "Google's efficient model",
    "qwen2.5:7b":       "Excellent multilingual + coding",
    "phi3:mini":        "Microsoft's tiny but capable (3.8B)",
    "codellama:13b":    "Specialized for code generation",
    "llava:13b":        "Multimodal vision + language",
    "nomic-embed-text": "Fast local embeddings",
    "deepseek-coder:6.7b": "State-of-art code model"
}
```

---

## 24. Model Context Protocol (MCP)

```python
# MCP — open standard by Anthropic for connecting AI to data sources/tools
# Like USB-C for AI: standardized way to plug in capabilities

# ─── MCP Architecture ─────────────────────────────────────────────────
# Host (Claude Desktop, IDE) ←→ MCP Client ←→ MCP Server ←→ Resource
#
# MCP Server exposes:
# - Resources:  Files, databases, API data (read-only)
# - Tools:      Functions the AI can call (read/write)
# - Prompts:    Reusable prompt templates

# ─── Building an MCP Server ───────────────────────────────────────────
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, Resource
import mcp.server.stdio
import asyncio
import json

server = Server("my-data-server")

# ─── Define Resources ─────────────────────────────────────────────────
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file://data/report.csv",
            name="Sales Report",
            description="Monthly sales data",
            mimeType="text/csv"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "file://data/report.csv":
        with open("data/report.csv") as f:
            return f.read()

# ─── Define Tools ─────────────────────────────────────────────────────
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_database",
            description="Run a SQL query against the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="send_email",
            description="Send an email",
            inputSchema={
                "type": "object",
                "properties": {
                    "to":      {"type": "string"},
                    "subject": {"type": "string"},
                    "body":    {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_database":
        result = execute_sql(arguments["query"])
        return [TextContent(type="text", text=json.dumps(result))]

    elif name == "send_email":
        success = send_email(arguments["to"], arguments["subject"], arguments["body"])
        return [TextContent(type="text", text=f"Email sent: {success}")]

# ─── Run server ───────────────────────────────────────────────────────
async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="my-data-server",
            server_version="1.0.0",
            capabilities=server.get_capabilities(
                notification_options=None, experimental_capabilities={}
            )
        ))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 25. AI Memory Systems

```python
# ─── Memory Types for AI Applications ────────────────────────────────

from mem0 import Memory
from openai import OpenAI

# ─── Mem0 — managed memory layer for AI ──────────────────────────────
m = Memory()

# Add memory
result = m.add(
    "Alice is a software engineer who prefers Python and dislikes Java",
    user_id="alice-123"
)

# Search memory
memories = m.search("programming preferences", user_id="alice-123")
print(memories)   # [{"memory": "Alice prefers Python...", "score": 0.95}]

# Get all memories
all_memories = m.get_all(user_id="alice-123")

# Update memory (automatic — detects contradictions)
m.add("Alice now also likes Rust", user_id="alice-123")

# ─── Building Memory from Scratch ─────────────────────────────────────
class LongTermMemory:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.session_memories = {}

    def extract_facts(self, conversation: str) -> list[str]:
        """Extract memorable facts from conversation"""
        prompt = f"""
        Extract important facts about the user from this conversation.
        Return as JSON array of strings. Only extract explicit facts.

        Conversation:
        {conversation}

        Facts (JSON array):"""

        response = self.llm.generate(prompt)
        return json.loads(response)

    def store(self, facts: list[str], user_id: str):
        for fact in facts:
            embedding = get_embedding(fact)
            self.vectorstore.upsert({
                "id": f"{user_id}-{hash(fact)}",
                "values": embedding,
                "metadata": {"fact": fact, "user_id": user_id, "ts": time.time()}
            })

    def recall(self, query: str, user_id: str, k: int = 5) -> list[str]:
        results = self.vectorstore.query(
            vector=get_embedding(query),
            filter={"user_id": {"$eq": user_id}},
            top_k=k
        )
        return [r["metadata"]["fact"] for r in results["matches"]]

    def build_context(self, query: str, user_id: str) -> str:
        memories = self.recall(query, user_id)
        if not memories:
            return ""
        return "What I remember about you:\n" + "\n".join(f"- {m}" for m in memories)

# ─── Conversation Summarization Memory ───────────────────────────────
class SummarizationMemory:
    """
    Maintain a rolling summary of conversation history.
    Prevents context window overflow for long conversations.
    """
    def __init__(self, llm, max_messages=20):
        self.llm = llm
        self.max_messages = max_messages
        self.summary = ""
        self.recent_messages = []

    def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        if len(self.recent_messages) > self.max_messages:
            self._summarize_and_trim()

    def _summarize_and_trim(self):
        to_summarize = self.recent_messages[:10]
        new_summary = self.llm.summarize(
            previous_summary=self.summary,
            new_messages=to_summarize
        )
        self.summary = new_summary
        self.recent_messages = self.recent_messages[10:]

    def get_context(self) -> list[dict]:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}"
            })
        return context + self.recent_messages
```

---

## 26. Structured Output & Function Calling

```python
# ─── Pydantic + OpenAI Structured Output ─────────────────────────────
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from openai import OpenAI
import json

client = OpenAI()

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"

class Person(BaseModel):
    name: str = Field(description="Full legal name")
    age: int = Field(ge=0, le=150, description="Age in years")
    email: Optional[str] = Field(default=None, description="Email address")
    occupation: str
    address: Optional[Address] = None
    skills: list[str] = Field(default_factory=list)
    experience_level: Literal["junior", "mid", "senior", "principal"]

    @field_validator("email")
    def validate_email(cls, v):
        if v and "@" not in v:
            raise ValueError("Invalid email")
        return v

# Guaranteed structured output
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{
        "role": "user",
        "content": "Alice Johnson is a 30-year-old senior Python developer at Google. alice@example.com"
    }],
    response_format=Person
)

person = completion.choices[0].message.parsed
print(person.name)         # "Alice Johnson"
print(person.age)          # 30
print(person.experience_level)  # "senior"

# ─── Instructor Library ───────────────────────────────────────────────
import instructor

client = instructor.from_openai(OpenAI())

# Works with nested models, lists, validation
class Product(BaseModel):
    name: str
    price: float
    category: Literal["electronics", "clothing", "food"]
    features: list[str]

class ProductList(BaseModel):
    products: list[Product]
    total_count: int

result = client.chat.completions.create(
    model="gpt-4o",
    response_model=ProductList,
    messages=[{"role": "user", "content": "List 3 smartphone products"}]
)
print(result.products[0].name)

# instructor also supports Anthropic, Cohere, Gemini, and more
claude_client = instructor.from_anthropic(anthropic.Anthropic())

# ─── JSON Mode ────────────────────────────────────────────────────────
# Simpler but less strict than structured output
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": "Get info for user Bob: age 25, likes Python"}
    ]
)
data = json.loads(response.choices[0].message.content)
```

---

## 27. AI Observability & Monitoring

```python
# ─── LangSmith ────────────────────────────────────────────────────────
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__key"
os.environ["LANGCHAIN_PROJECT"] = "production-rag"

# All LangChain runs are automatically traced
# Tracks: latency, tokens, costs, errors, intermediate steps

from langsmith import Client
ls_client = Client()

# Create datasets for evaluation
dataset = ls_client.create_dataset("rag-test-set")
ls_client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "Retrieval-Augmented Generation is..."}],
    dataset_id=dataset.id
)

# Run evaluations
from langsmith.evaluation import evaluate

results = evaluate(
    lambda inputs: qa_chain.invoke(inputs),
    data="rag-test-set",
    evaluators=[correctness_evaluator, faithfulness_evaluator],
    experiment_prefix="gpt4o-rag-v2"
)

# ─── Langfuse (open source alternative) ──────────────────────────────
from langfuse import Langfuse
from langfuse.openai import openai   # drop-in replacement

langfuse = Langfuse()

# Manual tracing
with langfuse.trace(name="rag-pipeline") as trace:
    with trace.span(name="retrieval") as retrieval_span:
        docs = retriever.retrieve(query)
        retrieval_span.end(output={"num_docs": len(docs)})

    with trace.span(name="generation") as gen_span:
        answer = llm.generate(query, docs)
        gen_span.end(output={"answer": answer})

# Score outputs
langfuse.score(
    trace_id=trace.id,
    name="faithfulness",
    value=0.95,
    comment="Answer well supported by context"
)

# ─── Key Metrics to Monitor ───────────────────────────────────────────
monitoring_metrics = {
    "latency": {
        "TTFT": "Time to first token (streaming UX)",
        "E2E_latency": "Total request time",
        "p50_p95_p99": "Percentile latencies"
    },
    "cost": {
        "input_tokens": "Tokens in prompt",
        "output_tokens": "Generated tokens",
        "cost_per_query": "$ per query",
        "daily_spend": "Budget tracking"
    },
    "quality": {
        "faithfulness": "Answer supported by context",
        "relevancy": "Answer relevant to question",
        "hallucination_rate": "% of hallucinated responses",
        "user_thumbs_up": "User satisfaction signal"
    },
    "reliability": {
        "error_rate": "Failed API calls",
        "retry_rate": "Retried requests",
        "cache_hit_rate": "Prompt cache hits"
    }
}

# ─── Phoenix (Arize) — open source observability ─────────────────────
import phoenix as px

# Launch UI: px.launch_app()
session = px.launch_app()

from phoenix.otel import register
tracer_provider = register(project_name="my-llm-app")

from opentelemetry.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# All OpenAI calls now traced in Phoenix UI
```

---

## 28. Responsible AI & Ethics

### Bias Detection & Mitigation

```python
# ─── Detecting Bias in LLM Outputs ───────────────────────────────────
from transformers import pipeline

# Use a bias detection model
bias_detector = pipeline("text-classification", model="valurank/distilroberta-bias")

def check_bias(text: str) -> dict:
    result = bias_detector(text)[0]
    return {"label": result["label"], "score": result["score"]}

# Test for gender bias
prompts = [
    "The nurse said she would",
    "The engineer said he would",
    "The CEO made a decision about"
]
for p in prompts:
    print(f"{p!r} → {check_bias(p)}")

# ─── Counterfactual Testing ───────────────────────────────────────────
def counterfactual_test(model, prompt_template, demographic_pairs):
    """
    Test if model treats demographic groups differently
    Swap names/pronouns and compare outputs
    """
    results = {}
    for group_a, group_b in demographic_pairs:
        response_a = model.generate(prompt_template.format(name=group_a))
        response_b = model.generate(prompt_template.format(name=group_b))
        sentiment_diff = analyze_sentiment(response_a) - analyze_sentiment(response_b)
        results[(group_a, group_b)] = abs(sentiment_diff)

    return results

counterfactual_test(
    model=llm,
    prompt_template="Write a reference letter for {name}, a software engineer",
    demographic_pairs=[("Emily", "James"), ("Maria", "David"), ("Aisha", "Michael")]
)

# ─── Privacy-Preserving AI ────────────────────────────────────────────
import re

def sanitize_pii(text: str) -> tuple[str, dict]:
    """Remove PII before sending to LLM"""
    replacements = {}

    # Email
    for i, email in enumerate(re.findall(r'\b[\w.]+@[\w.]+\.\w+\b', text)):
        placeholder = f"[EMAIL_{i}]"
        replacements[placeholder] = email
        text = text.replace(email, placeholder)

    # Phone
    for i, phone in enumerate(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)):
        placeholder = f"[PHONE_{i}]"
        replacements[placeholder] = phone
        text = text.replace(phone, placeholder)

    # SSN
    for i, ssn in enumerate(re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)):
        placeholder = f"[SSN_{i}]"
        replacements[placeholder] = ssn
        text = text.replace(ssn, placeholder)

    return text, replacements

def restore_pii(text: str, replacements: dict) -> str:
    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    return text

# Usage
sanitized, pii_map = sanitize_pii("John's email is john@corp.com, SSN 123-45-6789")
response = llm.generate(sanitized)
final_response = restore_pii(response, pii_map)
```

---

## 29. Production AI Systems

### Architecture Patterns

```python
# ─── Caching Strategies ───────────────────────────────────────────────
import hashlib
import redis
import json

r = redis.Redis(host="localhost", port=6379)

def cached_llm_call(prompt: str, model: str, ttl: int = 3600) -> str:
    """Exact match caching for identical prompts"""
    cache_key = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    response = llm.generate(prompt)
    r.setex(cache_key, ttl, json.dumps(response))
    return response

# Semantic caching (cache similar questions)
def semantic_cached_llm_call(query: str, threshold: float = 0.95) -> str:
    query_embedding = get_embedding(query)

    # Search for similar cached queries
    similar = vector_cache.search(query_embedding, k=1)

    if similar and similar[0]["score"] > threshold:
        return similar[0]["cached_response"]

    response = llm.generate(query)

    # Cache this query + response
    vector_cache.upsert({
        "embedding": query_embedding,
        "query": query,
        "cached_response": response
    })
    return response

# ─── Rate Limiting & Retry Logic ──────────────────────────────────────
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def resilient_llm_call(prompt: str) -> str:
    return await async_llm.generate(prompt)

# ─── Load Balancing Across Providers ──────────────────────────────────
class LLMRouter:
    def __init__(self):
        self.providers = [
            {"client": openai_client, "model": "gpt-4o", "weight": 0.5},
            {"client": anthropic_client, "model": "claude-opus-4-5", "weight": 0.3},
            {"client": google_client, "model": "gemini-1.5-pro", "weight": 0.2}
        ]
        self.error_counts = {p["model"]: 0 for p in self.providers}

    def get_provider(self):
        """Weighted random selection with error-based routing"""
        available = [p for p in self.providers if self.error_counts[p["model"]] < 5]
        total = sum(p["weight"] for p in available)
        r = random.uniform(0, total)
        cumulative = 0
        for provider in available:
            cumulative += provider["weight"]
            if r <= cumulative:
                return provider

    async def generate(self, prompt: str) -> str:
        provider = self.get_provider()
        try:
            response = await provider["client"].generate(prompt, provider["model"])
            self.error_counts[provider["model"]] = 0   # reset on success
            return response
        except Exception as e:
            self.error_counts[provider["model"]] += 1
            return await self.generate(prompt)   # fallback to another

# ─── A/B Testing LLMs ────────────────────────────────────────────────
import random

class ABTestingRouter:
    def __init__(self, experiments: dict):
        # experiments = {"exp-1": {"model": "gpt-4o", "traffic": 0.5},
        #                "exp-2": {"model": "claude-3-opus", "traffic": 0.5}}
        self.experiments = experiments

    def route(self, user_id: str) -> str:
        # Deterministic routing by user_id (consistent experience)
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100

        cumulative = 0
        for exp_name, config in self.experiments.items():
            cumulative += config["traffic"] * 100
            if hash_val < cumulative:
                return config["model"]

# ─── Cost Management ──────────────────────────────────────────────────
token_costs = {
    "gpt-4o":           {"input": 5.00/1e6,   "output": 15.00/1e6},
    "gpt-4o-mini":      {"input": 0.15/1e6,   "output": 0.60/1e6},
    "claude-opus-4-5":  {"input": 15.00/1e6,  "output": 75.00/1e6},
    "claude-haiku-3":   {"input": 0.25/1e6,   "output": 1.25/1e6},
}

def estimate_cost(prompt: str, model: str, expected_output_tokens: int = 500) -> float:
    input_tokens = count_tokens(prompt)
    costs = token_costs[model]
    return (input_tokens * costs["input"]) + (expected_output_tokens * costs["output"])

# Smart model routing based on task complexity
def smart_route(query: str) -> str:
    complexity = estimate_complexity(query)
    if complexity < 3:
        return "gpt-4o-mini"    # simple: cheap model
    elif complexity < 7:
        return "gpt-4o"         # medium: standard model
    else:
        return "claude-opus-4-5"  # complex: best model
```

### Deployment

```bash
# ─── FastAPI LLM Service ──────────────────────────────────────────────
# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI(title="LLM API")

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    max_tokens: int = 1000
    stream: bool = False

@app.post("/generate")
async def generate(request: GenerateRequest):
    if request.stream:
        async def stream_response():
            async for chunk in async_llm.stream(request.prompt):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_response(), media_type="text/event-stream")

    response = await async_llm.generate(request.prompt)
    return {"response": response, "model": request.model}

# Docker deployment
# FROM python:3.11-slim
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 30. Future of Generative AI

### Emerging Trends & Technologies

```python
future_trends = {

    "reasoning_models": {
        "examples": ["o1", "o3", "DeepSeek-R1", "QwQ"],
        "description": "Models that 'think' before answering using extended CoT",
        "how_it_works": "Trained to generate long reasoning chains before final answer",
        "use_cases": ["Math proofs", "Complex coding", "Scientific reasoning"],
        "tradeoff": "Much slower and more expensive, but significantly more accurate"
    },

    "agentic_ai": {
        "description": "AI that takes long-horizon actions autonomously",
        "current": "Multi-step agents with tools",
        "future": "Fully autonomous AI workers (Computer use, coding agents)",
        "examples": ["Claude Computer Use", "OpenAI Operator", "Devin"],
        "challenges": ["Reliability", "Safety", "Error recovery", "Trust"]
    },

    "multimodal_everything": {
        "description": "All models become natively multimodal",
        "modalities": ["Text", "Image", "Audio", "Video", "3D", "Code", "Actions"],
        "examples": ["GPT-4o (omni)", "Gemini 1.5 Pro", "Claude 3.5"]
    },

    "mixture_of_experts": {
        "description": "Route tokens to specialized sub-networks (experts)",
        "benefit": "Large model capacity with less compute per token",
        "examples": ["GPT-4 (rumored)", "Mixtral 8x7B", "Grok-1"],
        "formula": "Total params >> Active params per token"
    },

    "test_time_compute": {
        "description": "Scale compute at inference, not just training",
        "methods": ["Chain of Thought", "Self-consistency", "Monte Carlo Tree Search"],
        "insight": "More thinking at test time = better answers (trading $ for quality)"
    },

    "world_models": {
        "description": "AI that understands and predicts how the world works",
        "examples": ["Sora (video understanding)", "Genie (game world models)"],
        "future": "Foundation for robotics and embodied AI"
    },

    "small_language_models": {
        "examples": ["Phi-3", "Gemma 2", "Llama 3.2 1B/3B"],
        "description": "Tiny models with surprisingly good performance",
        "use_cases": ["Edge devices", "Mobile", "Low-latency applications"],
        "key_insight": "Quality of training data matters more than model size"
    },

    "long_context": {
        "current": "GPT-4: 128K, Gemini: 1M, Claude: 200K",
        "future": "Unlimited effective context via better architectures",
        "challenges": ["Lost in the middle problem", "Cost", "Latency"]
    },

    "continual_learning": {
        "description": "Models that learn from new data without forgetting old",
        "challenge": "Catastrophic forgetting",
        "approaches": ["EWC", "Replay buffers", "Parameter isolation"]
    },

    "neurosymbolic_ai": {
        "description": "Combining neural networks with symbolic reasoning",
        "benefit": "Better logical consistency, interpretability",
        "examples": ["Program synthesis", "Formal verification + LLMs"]
    }
}
```

### Architecture of a Modern GenAI Application

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODERN GenAI APPLICATION STACK                    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              USER INTERFACE (Web / Mobile / CLI)               │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
│                            │                                         │
│  ┌─────────────────────────▼─────────────────────────────────────┐  │
│  │                   APPLICATION LAYER                            │  │
│  │  Rate Limiting │ Auth │ A/B Testing │ Cost Tracking │ Caching  │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
│                            │                                         │
│  ┌─────────────────────────▼─────────────────────────────────────┐  │
│  │                  AI ORCHESTRATION LAYER                        │  │
│  │  LangChain / LlamaIndex / Custom │ Agents │ Memory │ RAG      │  │
│  └──────┬──────────────────┬──────────────────┬──────────────────┘  │
│         │                  │                  │                      │
│  ┌──────▼───────┐  ┌───────▼──────┐  ┌───────▼──────────────────┐  │
│  │  LLM APIs    │  │  Vector DB   │  │  External Tools / MCP    │  │
│  │  OpenAI      │  │  Pinecone    │  │  Web Search              │  │
│  │  Anthropic   │  │  Weaviate    │  │  Code Execution          │  │
│  │  Local/Ollama│  │  Qdrant      │  │  File System             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              OBSERVABILITY (LangSmith / Langfuse)             │  │
│  │           Tracing │ Evaluation │ Cost │ Latency │ Quality     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Quick Reference Cheatsheet

```python
# ── LLM Providers ─────────────────────────────────────────────────────
openai_client   = OpenAI(api_key="sk-...")
anthropic_client = Anthropic(api_key="sk-ant-...")
# Google: import google.generativeai as genai; genai.configure(api_key="...")
# Local:  ollama.chat(model="llama3.1", messages=[...])

# ── Prompting Tips ────────────────────────────────────────────────────
# Zero-shot:  "Classify: {text}"
# Few-shot:   "Examples... → Now classify: {text}"
# CoT:        "Think step by step..."
# Role:       "You are an expert in..."
# Output:     "Return as JSON: {schema}"

# ── RAG Steps ─────────────────────────────────────────────────────────
# Load → Split → Embed → Store → Retrieve → Generate

# ── Fine-Tuning Order ─────────────────────────────────────────────────
# Pre-train → SFT → RLHF/DPO → Deploy

# ── Model Size vs Use Case ────────────────────────────────────────────
# 1-3B:   Edge, mobile, simple tasks
# 7-13B:  Good all-around, runs on consumer GPU
# 70B:    Near GPT-4 quality, needs A100
# 405B+:  Best quality, datacenter only

# ── Sampling (Temperature Guide) ──────────────────────────────────────
# temp=0.0: Deterministic (classification, factual QA)
# temp=0.3: Focused (summarization, extraction)
# temp=0.7: Balanced (chat, coding)
# temp=1.0: Creative (brainstorming, poetry)
# temp=1.5: Very random (creative experiments)

# ── Token Math ───────────────────────────────────────────────────────
# 1 token ≈ 0.75 words   |   1000 tokens ≈ 750 words ≈ 3 pages
# GPT-4o context: 128K tokens ≈ ~96K words ≈ ~350 pages

# ── Cost Ballpark (2024) ──────────────────────────────────────────────
# gpt-4o-mini:    $0.15/1M in, $0.60/1M out  ← cheapest capable
# gpt-4o:         $5/1M in, $15/1M out
# claude-haiku:   $0.25/1M in, $1.25/1M out
# claude-opus:    $15/1M in, $75/1M out
# local (ollama): $0 (electricity only)

# ── Vector DB Comparison ─────────────────────────────────────────────
# Development: ChromaDB (local, easy)
# Production:  Pinecone (managed), Weaviate (hybrid), Qdrant (fast)
# In-memory:   FAISS (Meta, most control)

# ── Evaluation ────────────────────────────────────────────────────────
# RAG: faithfulness, answer_relevancy, context_recall, context_precision
# Chat: MT-Bench, AlpacaEval 2.0
# Code: HumanEval, SWE-bench
# Safety: TruthfulQA, HarmBench
```

---

## 📖 Further Resources

### Papers
- **Attention Is All You Need** (2017) — The original Transformer paper
- **GPT-3: Language Models are Few-Shot Learners** (2020)
- **Constitutional AI** (Anthropic, 2022)
- **Llama 2** (Meta, 2023) — open weights + training details
- **Scaling Laws for Neural Language Models** (OpenAI, 2020)
- **RLHF: Training Language Models to Follow Instructions** (2022)
- **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP** (2020)
- **Flash Attention 2** (2023)
- **Mixtral of Experts** (Mistral, 2024)

### Courses & Learning
- **fast.ai Practical Deep Learning** — free, hands-on
- **Stanford CS224N** — NLP with Deep Learning
- **DeepLearning.AI Short Courses** — Andrew Ng, many free courses on LLMs
- **Hugging Face NLP Course** — free, comprehensive
- **Andrej Karpathy's YouTube** — nanoGPT, LLM from scratch

### Tools & Frameworks
| Tool | Purpose |
|------|---------|
| [LangChain](https://langchain.com) | LLM application framework |
| [LlamaIndex](https://llamaindex.ai) | RAG & data indexing |
| [Hugging Face](https://huggingface.co) | Models, datasets, training |
| [Ollama](https://ollama.ai) | Local LLM runner |
| [vLLM](https://vllm.ai) | Production inference |
| [LangSmith](https://smith.langchain.com) | LLM observability |
| [Langfuse](https://langfuse.com) | Open-source observability |
| [Weights & Biases](https://wandb.ai) | ML experiment tracking |
| [Unsloth](https://unsloth.ai) | Fast fine-tuning |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Fine-tuning toolkit |

---

*Guide reflects the state of Generative AI as of early 2025. This field evolves extremely rapidly — always verify with latest documentation.*