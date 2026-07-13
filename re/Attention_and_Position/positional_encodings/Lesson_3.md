# Lesson 3 — Learned Absolute Positional Embeddings

> *Prerequisites: Lesson 2 (Sinusoidal PE)*
> *Papers: BERT (Devlin et al. 2018), GPT (Radford et al. 2018), ViT (Dosovitskiy et al. 2020)*

---

## The Problem

Sinusoidal PE is hand-designed with fixed frequencies. The frequencies were chosen to look mathematically elegant and to satisfy certain properties — but there's no reason the optimal positional representation for a given task should follow a mathematical formula.

If the model is going to learn everything else (token embeddings, attention weights, feedforward weights), why not also learn **what positional information is useful**? Learned positional embeddings let the model discover the positional structure that actually helps it solve its task.

---

## The Mechanism: A Lookup Table

Instead of computing positional vectors with a formula, create a weight matrix of shape `[max_seq_len × d_model]`:

```
P ∈ R^(N × d_model)     # N = max sequence length

positional embedding for position pos = P[pos]   # just a row lookup
```

Each row is a learnable parameter. Randomly initialized and updated during training via backpropagation, exactly like token embeddings.

The input to the transformer becomes:
```
input[pos] = token_embedding[pos] + P[pos]
```

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout=0.1):
        super().__init__()
        # This is just an Embedding table — a learnable (max_seq_len, d_model) matrix
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model) — already token-embedded
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # [0, 1, ..., seq_len-1]
        pos_emb = self.embedding(positions)                  # (seq_len, d_model)
        return self.dropout(x + pos_emb)

# Used in BERT exactly like this:
# bert_embedding = token_embedding + segment_embedding + positional_embedding
```

This is exactly what BERT, GPT, and GPT-2 use.

---

## How It Differs from Sinusoidal

| | Sinusoidal | Learned |
|---|---|---|
| How computed | Fixed formula (sin/cos) | Looked up from trained embedding matrix |
| Learned during training | No | Yes |
| Can adapt to task | No | Yes |
| Handles unseen positions | Yes (formula always works) | **No** — hard failure |
| Type | Absolute, fixed | Absolute, learned |
| Parameters added | 0 | `max_seq_len × d_model` |

The paper "Attention Is All You Need" directly compared both on machine translation and found **nearly identical performance**. This suggests that for tasks within the training context length, the sinusoidal structure is not uniquely important — the model learns equally well with or without it.

---

## Who Uses It and Their Limits

| Model | PE Type | Max Length |
|---|---|---|
| **BERT base/large** | Learned | 512 tokens — hard limit |
| **GPT** | Learned | 512 tokens |
| **GPT-2** | Learned | 1,024 tokens |
| **GPT-3** | Learned | 2,048 tokens |
| **Vision Transformer (ViT)** | Learned | Fixed grid of image patches |
| **Original Transformer** | Sinusoidal | No hard limit |

The hard limits above are a direct consequence of the learned PE architecture.

---

## The Hard Length Limit — The Critical Weakness

The weight matrix P has exactly `max_seq_len` rows. Position `pos` looks up row `P[pos]`. If a sequence has 513 tokens but BERT was trained with `max_seq_len = 512`, there is **no row 513**.

```
Trained with max_seq_len = 512
Input at inference: 600 tokens → positions 513–600 have no embedding
```

This is a hard architectural constraint, not a soft performance degradation. The model literally cannot process sequences beyond `max_seq_len`.

Sinusoidal PE has no such limit — the formula works for any position. This difference becomes increasingly important as context lengths grow from 512 → 4K → 32K → 128K.

> **Interview note:** "Why did BERT have a 512 token limit?" — BERT uses learned absolute positional embeddings with `max_seq_len = 512`. Beyond position 512, there is no learned positional embedding — the architecture is physically unable to process longer sequences. This was a practical choice (512 covers most NLP tasks) but became a major limitation as the field moved toward longer contexts.

---

## Why It Still Works Well Within Training Length

Within the trained range, the model has full freedom to shape each position's embedding however it finds useful. Gradient descent discovers the structure — and it often doesn't look like sine waves.

Empirically, the learned embeddings tend to:
- Cluster adjacent positions close together in embedding space
- Create distinct clusters for different sentence regions
- Encode coarser structure than sinusoidal (less fine-grained oscillation, more "zones")
- Reflect the statistical patterns of the specific training data

For example, in BERT trained on Wikipedia + Books Corpus, position 0 has a very distinctive embedding (always `[CLS]` token in BERT's format), while positions 1–10 cluster near each other.

---

## No Generalization to Relative Distances

Like sinusoidal PE, learned absolute embeddings are **position-dependent, not distance-dependent**. If token A is at position 3 and token B is at position 7, the model sees `P[3]` and `P[7]`. It must learn that the *difference* between these two specific embedding rows encodes "4 positions apart" — through the Q and K weight matrices.

This implicit learning is harder than providing relative distances directly. It also fails to generalize: if the model sees "4 positions apart" at positions (3, 7) during training, but needs to apply the same pattern at positions (203, 207) at inference, it must rely on the learned Q/K weights having captured the general pattern — not the specific PE values.

**Relative PE methods** (Lesson 4, 5, 6) solve this by encoding `pos_i - pos_j` directly into the attention score, making distance explicitly available to every layer.

---

## Extending Learned PE — Fine-tuning Approaches

When you need longer context than the trained `max_seq_len`, several approaches exist:

**1. Interpolation:** Linearly interpolate existing embeddings to fill the new positions. For a new position `pos` in an extended range, compute: `P_new[pos] = P[pos * (old_max / new_max)]` using bilinear interpolation.

**2. Extrapolation + fine-tuning:** Initialize new embeddings for positions beyond `max_seq_len` (randomly or by repeating the last learned embedding), then fine-tune on long sequences.

**3. Replace with RoPE or ALiBi:** Replace the learned absolute PE entirely with a relative method and fine-tune. This is what most modern models do when upgrading from BERT/GPT-style to long-context.

---

## Summary

- Learned PE is a lookup table `P ∈ R^(N × d_model)` trained with the model
- Within training length: matches or slightly outperforms sinusoidal PE
- **Hard length limit** at `max_seq_len` — physical inability to process longer sequences
- BERT (512), GPT-2 (1024), GPT-3 (2048) all hit this limit
- Like sinusoidal PE: absolute (not relative), so relative distances must be inferred implicitly

---

## Interview Q&A

**Q: Why did BERT have a 512 token limit?**
BERT uses a learned positional embedding matrix with 512 rows — one per position. Positions beyond 512 have no embedding — the architecture is physically unable to handle them. This was a practical choice for BERT's NLP tasks (most sentences fit in 512 tokens), but became a major limitation for long-document tasks.

**Q: Can you fine-tune a BERT model to handle longer sequences?**
Yes, but it requires adding and initializing new positional embeddings for the extended positions, then fine-tuning. The new embeddings (for positions 513+) are randomly initialized and have no learned structure from pretraining. You need significant fine-tuning data at the longer lengths to make them useful.

**Q: Why did learned PE beat sinusoidal PE in some benchmarks?**
The model can adapt the positional representation to the specific task and data distribution. If certain positional patterns (e.g., position 0 is always special in BERT) are consistently important, learned PE can concentrate positional information in the most useful way for that task. Sinusoidal PE provides a generic multi-scale structure, not a task-adapted one.

**Q: What's the parameter cost of learned PE?**
`max_seq_len × d_model`. For BERT (512 × 768) ≈ 393K parameters — small compared to BERT's 110M total, but nonzero. For a model with 128K context, learned PE would add `128,000 × 4096 ≈ 500M parameters` — as large as a 3B parameter model. This is why modern long-context models don't use learned absolute PE.
