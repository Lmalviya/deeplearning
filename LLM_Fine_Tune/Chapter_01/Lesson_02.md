# Lesson 1.2 — Transformer Architecture Internals

---

## 1. The Problem Story

Rahul got asked this in an interview: *"You used QLoRA with target_modules=['q_proj', 'v_proj']. Why those two matrices? What are they doing?"*

He had copied that from the QLoRA paper's example config without understanding it. He said "those are the query and value projections in attention" and then went silent when the interviewer said "yes, and what do they do exactly?"

He did not get the offer.

This lesson means you will never be in that position. You will know exactly what Q, K, and V do, what each layer in the transformer does, and why LoRA targets attention matrices specifically.

---

## 2. The Concept

### The Big Picture

A transformer language model is a stack of layers. At the bottom, tokens are converted to vectors. Those vectors flow through many identical layers. At the top, vectors are converted back to probability distributions over the vocabulary.

```
Text Input: "The cat sat"
     ↓
[Tokenizer]
     ↓
Token IDs: [450, 6635, 7960]
     ↓
[Embedding Layer]   → convert IDs to dense vectors
     ↓
[Transformer Layer 1]
[Transformer Layer 2]
     ...
[Transformer Layer N]
     ↓
[LM Head]           → convert final vectors to vocab probabilities
     ↓
Probability distribution over next token
```

### The Embedding Layer

Each token ID maps to a learned vector. If the vocabulary has 32,000 tokens and the model dimension is 4096, the embedding table is a matrix of shape `(32000, 4096)`.

When the model processes token ID 450, it does a lookup: row 450 of the embedding matrix. This gives a 4096-dimensional vector that represents the "meaning" of that token as the model has learned it.

This is the embedding layer — the very first trainable part of the model that your input touches.

```
Token ID: 450
Embedding table[450] → [0.23, -0.11, 0.87, ..., 0.04]  # 4096 numbers
```

The embedding table is large. For a 7B model with vocab size 32,000 and dim 4,096: that is 131 million parameters — just in the embeddings.

### Positional Encoding

Self-attention has no built-in sense of order. Token 1 and token 3 are treated identically unless you tell the model their positions. Positional encodings add position information to each token's vector.

Modern models use **RoPE (Rotary Position Embedding)** — position information is embedded in the angle of the query and key vectors during attention computation, rather than added to the token embedding. This is why modern models generalize better to longer sequences.

### One Transformer Layer

Every transformer layer has two sub-components. They run in sequence, and both have a residual connection (skip connection) around them:

```
Input (x)
  ↓
LayerNorm(x) → Multi-Head Self-Attention → output (a)
  ↓
x = x + a          ← residual connection
  ↓
LayerNorm(x) → Feed-Forward Network → output (f)
  ↓
x = x + f          ← residual connection
  ↓
Output (x) — goes to next layer
```

Let us look at each component in depth.

### Multi-Head Self-Attention (The Core)

This is where the model learns which tokens to pay attention to when processing each token.

**Step 1: Create Q, K, V matrices**

For each token's vector, we create three projections using three weight matrices:
- Q (Query): "What am I looking for?"
- K (Key): "What do I represent?"
- V (Value): "What information do I carry?"

```
token_vector (dim=4096)

Q = token_vector × W_Q   → shape: (seq_len, head_dim)
K = token_vector × W_K   → shape: (seq_len, head_dim)
V = token_vector × W_V   → shape: (seq_len, head_dim)

Where W_Q, W_K, W_V are learned weight matrices.
```

**Step 2: Compute attention scores**

For every pair of tokens (i, j), compute how much token i should attend to token j:

```
score(i, j) = Q_i · K_j / sqrt(head_dim)
```

The division by `sqrt(head_dim)` prevents the dot products from getting too large (which would cause the softmax to saturate).

**Why Do We Divide Attention Scores by** $\sqrt{d_k}$?

#### 1. Variance Grows with Dimension

The attention score is a dot product:

$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$

If each component has mean 0 and variance $\approx 1$ then sum of those d terms varaince will be: $
\text{Var}(q \cdot k) \approx d_k $

Therefore, $ \text{Std}(q \cdot k) \approx \sqrt{d_k} $

As `head_dim` increases, attention scores become larger.

---

#### 2. Large Scores Cause Problems

Large dot products are passed to the softmax:
$
\text{softmax}(QK^T)
$

As the scores grow, the largest value dominates.

Example:

```text
[10, 20, 30]
```

becomes approximately

```text
[0, 0, 1]
```

The model attends almost entirely to a single token that lead to softmax saturation.

#### 3. Softmax Saturation

When softmax outputs are close to 0 or 1:

```text
[0, 0, 1]
```

their gradients become very small.

$
\frac{\partial \text{softmax}}{\partial x} = p(1-p)
$

If:
$
p \approx 0 \quad \text{or} \quad p \approx 1
$

then:
$
p(1-p) \approx 0
$

This leads to:

- Tiny gradients
- Slow learning
- Unstable training

---

#### 4. The Solution: Scale the Scores

Scale the dot product before softmax:
$
\frac{QK^T}{\sqrt{d_k}}
$

Since the standard deviation of the dot product is roughly $\sqrt{d_k}$:

$
\text{Std}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\approx 1
$

This keeps attention scores in a reasonable range.

---

### Key Takeaway

Without scaling:

```text
Large dot products
    ↓
Softmax saturation
    ↓
Tiny gradients
    ↓
Poor training
```

With scaling by $\sqrt{d_k}$:

```text
Controlled variance
    ↓
Healthy softmax outputs
    ↓
Stable gradients
    ↓
Better training
```
---

**Step 3: Apply causal mask (for decoder-only models)**

In GPT-style models, token i can only attend to tokens at position j ≤ i (tokens to its left). Future tokens are masked to -infinity before softmax. This is the "causal" in causal language modeling.

**Step 4: Softmax to get attention weights**

```
attention_weights(i) = softmax(scores(i, :))
```

This gives a probability distribution: how much should token i attend to each other token?

**Step 5: Weighted sum of values**

```
output(i) = sum(attention_weights(i, j) * V_j for all j)
```

Token i's output is a weighted combination of all value vectors, weighted by how much attention i pays to each j.

**Step 6: Multi-head**

This is done H times in parallel with different W_Q, W_K, W_V matrices (called "heads"). Each head can learn to attend to different types of relationships (syntax, coreference, semantics, etc.). The outputs of all heads are concatenated and projected:

```
output = concat(head_1, head_2, ..., head_H) × W_O
```

`W_O` is the output projection — another learned weight matrix.

So a full attention block has 4 weight matrices per layer: `W_Q, W_K, W_V, W_O`.

These are what you see as `q_proj, k_proj, v_proj, o_proj` in HuggingFace models.

**This is exactly why QLoRA targets these matrices.** They are the most information-dense, most "semantic" weights in the model. Fine-tuning them gives the most behavioral change per parameter.

### Feed-Forward Network (FFN)

After attention, each token's vector goes through a two-layer (or three-layer) feed-forward network independently:

```
FFN(x) = activation(x × W_gate) * (x × W_up) × W_down
```

Modern models use the SwiGLU activation, which involves a gating mechanism (hence `gate_proj`, `up_proj`, `down_proj` in LLaMA):

- `W_up`: expands dimension (e.g., 4096 → 11008)
- `W_gate`: another expansion for gating
- `W_down`: projects back down (11008 → 4096)

The FFN is thought to act as a "memory" — storing factual knowledge. Research has shown that factual associations (like "Paris is the capital of France") are stored in FFN weights, while attention weights handle routing and relational reasoning.

### Layer Normalization

Before each sub-component, the input is normalized. Modern models use RMSNorm (a simpler variant of LayerNorm):

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

Where `γ` is a learned scaling parameter. Normalization keeps the activations stable as they flow through many layers, making training much more stable.

**Pre-norm vs Post-norm:**
- Pre-norm: normalize before the sublayer (modern models, including LLaMA)
- Post-norm: normalize after the sublayer + residual (original transformer paper)
- Pre-norm is more stable for deep models and is now the standard.

### The LM Head

After all transformer layers, each token has a final hidden vector. The LM head projects this to vocabulary probabilities:

```
logits = final_hidden_state × W_lm_head   # shape: (seq_len, vocab_size)
```

The LM head is often tied to the embedding table (same weight matrix transposed), saving parameters.

### Model Dimensions

Different model sizes have different dimensions:

| Model | Layers | Hidden dim | Heads | FFN dim | Params |
|-------|--------|-----------|-------|---------|--------|
| GPT-2 | 12 | 768 | 12 | 3072 | 117M |
| LLaMA-3 8B | 32 | 4096 | 32 | 14336 | 8B |
| LLaMA-3 70B | 80 | 8192 | 64 | 28672 | 70B |
| Phi-3 Mini | 32 | 3072 | 32 | 8192 | 3.8B |

---

## 3. The Intuition Bridge

**Self-attention as a database query:**

Imagine you are at a library. You have a query (what you are looking for). The library has shelves of books with keys (subject tags) and values (content).

You compare your query against all the keys, figure out which books are most relevant, and read a weighted combination of their content.

That is exactly what attention does. Q is your query. K is the "subject tag" of each token. V is the content of each token. Attention figures out which tokens to "read" when processing each token.

**Why multiple heads?**

Different relationships matter simultaneously. When reading "The animal didn't cross the street because it was too tired" — the word "it" needs to resolve to "animal" (coreference). But the word "cross" also relates to "street" (semantic), and "tired" modifies the subject (syntactic). Different heads capture different types of relationships.

**The FFN as a key-value memory:**

Research shows FFN layers store factual knowledge. Factual associations are retrieved in the FFN ("The capital of France is ___"). This is why fine-tuning FFN layers helps with domain knowledge adaptation, while fine-tuning attention layers helps with behavioral/style changes.

---

## 4. Why This Matters for Fine-Tuning

**Why LoRA targets Q and V (not K, not FFN by default):**
- Q and V are the most impactful for changing what the model pays attention to and what it outputs
- K determines what tokens "offer" to attention, which changes less with task adaptation
- Targeting only Q+V reduces trainable parameters while keeping most of the benefit

**Why deeper layers are more task-specific:**
- Early layers handle syntax and basic token patterns (change little during fine-tuning)
- Later layers handle semantics and task-specific reasoning (change more during fine-tuning)
- This is why some approaches freeze early layers and only fine-tune later ones

**Why residual connections matter for fine-tuning:**
- Pre-trained weights are a good starting point
- LoRA's low-rank updates are added on top of these weights: `W_new = W_old + ΔW`
- Residual connections throughout the model preserve the original signal while allowing updates

---

## 5. The Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a small model to inspect
model_name = "microsoft/phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ── Explore model architecture ──────────────────────────────────

print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)

# ── Count parameters ────────────────────────────────────────────

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"In billions: {total_params / 1e9:.2f}B")

# ── Inspect a single layer ──────────────────────────────────────

# Access the first transformer layer
first_layer = model.model.layers[0]
print("\n" + "=" * 60)
print("LAYER 0 COMPONENTS")
print("=" * 60)
print(first_layer)

# ── Inspect attention weight shapes ────────────────────────────

attn = first_layer.self_attn
print("\n── Attention weight matrix shapes ──")
print(f"q_proj (W_Q): {attn.q_proj.weight.shape}")
print(f"k_proj (W_K): {attn.k_proj.weight.shape}")
print(f"v_proj (W_V): {attn.v_proj.weight.shape}")
print(f"o_proj (W_O): {attn.o_proj.weight.shape}")

# ── Inspect FFN weight shapes ───────────────────────────────────

ffn = first_layer.mlp
print("\n── FFN weight matrix shapes ──")
for name, param in ffn.named_parameters():
    print(f"  {name}: {param.shape}")

# ── Count params per component ──────────────────────────────────

print("\n── Parameters per component in layer 0 ──")
attn_params = sum(p.numel() for p in attn.parameters())
ffn_params = sum(p.numel() for p in ffn.parameters())
norm_params = sum(p.numel() for p in first_layer.input_layernorm.parameters())

print(f"  Attention:   {attn_params:,}")
print(f"  FFN:         {ffn_params:,}")
print(f"  LayerNorm:   {norm_params:,}")

# ── See all named modules ────────────────────────────────────────

print("\n── All named modules ──")
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"  {name}: {module.weight.shape}")
```

---

## 6. The Experiment

**Experiment 1.2.A — The Architecture Inspector**

Using the code above as a starting point, answer all of these by running code:

1. How many transformer layers does Phi-3 Mini have?
2. What is the hidden dimension (`d_model`)?
3. What is the FFN intermediate dimension?
4. How many attention heads?
5. How many parameters are in the attention component of each layer?
6. How many parameters are in the FFN component of each layer?
7. What fraction of total parameters are in attention vs FFN?
8. What is the embedding table size?

Write these down. Then ask yourself: if LoRA targets `q_proj` and `v_proj` only, in a model with 32 layers, how many LoRA parameters are there with rank=16? Calculate it.

```
LoRA params = 2 layers_targeted × num_layers × (d_model × rank + rank × d_model)
```

---

## 7. Interview Checkpoint

**Q: What do the Q, K, V matrices do in attention?**

> A: Q (query) represents what each token is looking for in other tokens. K (key) represents what each token "advertises" about itself. V (value) holds the actual content that will be aggregated. Attention computes dot products between each token's Q and all tokens' K to get relevance scores, then uses those scores to take a weighted sum of V vectors. The output is a context-aware representation of each token.

**Q: Why does LoRA typically target q_proj and v_proj?**

> A: Q and V matrices are the most impactful for changing attention behavior. Q determines what the model looks for; V determines what information is aggregated. K (what tokens advertise) tends to change less during task adaptation. By targeting Q and V, LoRA achieves good task adaptation with fewer parameters than targeting all four attention matrices.

**Q: What is the purpose of the feed-forward network in a transformer?**

> A: The FFN processes each token's representation independently after attention. It acts as a per-token computation step that adds non-linearity and is thought to store factual and world knowledge. Research (Meng et al., 2022) suggests factual associations are encoded in FFN weights, while attention handles routing and relational reasoning. This is why modifying FFN layers helps with domain adaptation while attention layers help with behavioral changes.

**Q: Why do modern transformers use pre-norm (normalize before the sublayer) instead of post-norm?**

> A: Pre-norm puts the residual path outside the normalization, which means gradients flow more cleanly through the residual connections during training. Post-norm (normalize after sublayer + residual) leads to training instability in deep models because the normalization can shrink gradient magnitude in the residual path. Pre-norm allows training much deeper models reliably.

---

## 8. Common Mistakes & Misconceptions

❌ **"Attention is the whole transformer."**
Attention is one of two sub-components per layer. The FFN is equally important and has more parameters in most modern architectures. Some researchers argue the FFN stores more "knowledge" than the attention layers.

❌ **"More attention heads = better."**
Heads have a fixed total budget (hidden_dim / num_heads = head_dim). More heads means each head has smaller capacity. Research shows many heads are redundant and can be pruned. It is a tradeoff, not "more is better."

❌ **"LoRA always targets q_proj and v_proj."**
This is the common default, but it is not always optimal. For some tasks, targeting all attention matrices (q, k, v, o) or even FFN matrices gives better results. Lesson 5.5 will cover this systematically.

❌ **"The residual connection just adds the input back."**
It does exactly that, but the implication is profound: the model can learn to output near-zero from each sublayer and rely entirely on the residual, effectively learning residuals. This is what makes fine-tuning from a pre-trained point effective — the pre-trained output is preserved via the residual, and the model only needs to learn small corrections.
