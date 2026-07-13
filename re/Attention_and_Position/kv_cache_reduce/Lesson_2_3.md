# Lesson 2.3 — Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

> *Builds on: Lesson 2.2 (KV Cache)*
> *Papers: "Fast Transformer Decoding: One Write-Head is All You Need" — Shazeer (2019); "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — Ainslie et al. (2023)*

---

## The Problem: KV Cache is Too Heavy

From Lesson 2.2, DeepSeek-R1/V3 MHA needs **131 GB just for KV cache** at 32K context. Even a medium model like LLaMA-2 7B (32 heads, 32 KV heads) at 4K context requires ~0.5 GB per sequence in BF16.

The root cause: **every attention head maintains its own independent K and V vectors**. With h = 32 heads and d_k = 128, each token stores:
```
K: 32 × 128 = 4096 values per token per layer
V: 32 × 128 = 4096 values per token per layer
Total: 8192 values × 2 bytes = 16,384 bytes = 16 KB per token per layer
```

For 4K context, 32 layers: `16 KB × 4096 × 32 ≈ 2 GB`.

The question: **do all 32 heads really need independent K and V?** Or can we share them?

---

## Multi-Query Attention (MQA) — Shazeer 2019

MQA takes the most aggressive stance: **all query heads share a single K and V head**.

```
Standard MHA (h=4 heads):
  Q: [Q1, Q2, Q3, Q4]     each has its own Wq
  K: [K1, K2, K3, K4]     each has its own Wk  ← 4 separate K projections
  V: [V1, V2, V3, V4]     each has its own Wv  ← 4 separate V projections

MQA (h=4 query heads, 1 KV head):
  Q: [Q1, Q2, Q3, Q4]     each has its own Wq
  K: [K,  K,  K,  K ]     single K shared across all query heads
  V: [V,  V,  V,  V ]     single V shared across all query heads
```

![MQA: 4 query heads share a single K and V; W^K ∈ R^(d × d_k×1), W^V ∈ R^(d × d_v×1); KV cache per token: MHA 4MB → MQA 31KB (128× reduction)](../../assets/attentions/Screenshot%202026-03-17%20101018.png)

*MQA with 4 query heads. K^W has only 1 column (dimension d_k × 1 instead of d_k × h). Same for V. The KV cache drops from 4 MB/token (MHA with 128 heads) to 31 KB/token — a 128× reduction.*

### Attention Computation in MQA

Each query head Q_i runs attention against the **same** K and V:

```
head_i = softmax( Q_i · Kᵀ / √d_k ) · V     (same K, V for all i)
```

The K and V are broadcast across heads — no separate storage per head.

### KV Cache Reduction

```
MHA: h × d_k + h × d_v = h × (d_k + d_v) values per token per layer
MQA: 1 × d_k + 1 × d_v = (d_k + d_v) values per token per layer

Reduction factor: h× (= 128× for h=128, as in the DeepSeek example)
```

From the image: MHA = 4 MB/token, MQA = 31 KB/token, reduction = 128×.

### MQA Limitations

**Quality degradation.** Having one K/V shared across all query heads significantly reduces model capacity:
- All heads look at the same "what's important" (K) and retrieve the same "payload" (V)
- The model loses the ability to route different types of information to different heads
- Performance drops are task-dependent but real — especially on tasks requiring nuanced, multi-faceted reasoning

This limitation motivated GQA: can we find a point between MHA (h separate K/V) and MQA (1 shared K/V)?

---

## Grouped-Query Attention (GQA) — Ainslie et al. 2023

GQA introduces **g groups** of query heads, each sharing one K/V pair. It interpolates between MHA and MQA:

```
GQA with h=4 query heads, g=2 groups:
  Group 1: Q1, Q2 → share K1, V1
  Group 2: Q3, Q4 → share K2, V2

  heads per group: h / g = 4 / 2 = 2
```

![GQA with 4 query heads and 2 KV groups: K1 shared by Q1,Q2; K2 shared by Q3,Q4; KV cache: MHA 4MB → MQA 31KB → GQA 500KB (8× reduction)](../../assets/attentions/Screenshot%202026-03-17%20101121.png)

*GQA(g=2) with 4 query heads. Two distinct K/V pairs. KV cache = 500 KB/token — 8× reduction vs MHA, but much better quality than MQA's 128× reduction.*

### GQA as a Unified Framework

```
g = h   →   GQA(g=h) = MHA     (every head has its own K/V)
g = 1   →   GQA(g=1) = MQA     (one K/V for all heads)
1 < g < h  →   GQA(g)          (intermediate trade-off)
```

GQA gives you a **knob** to trade off quality (more groups) against memory (fewer groups).

### Memory Comparison

| Method | KV heads | Memory/token (per layer) | Reduction vs MHA |
|---|---|---|---|
| **MHA** | h = 128 | 4 MB | 1× (baseline) |
| **GQA** (g = 16) | 16 | 500 KB | 8× |
| **MQA** | 1 | 31 KB | 128× |

*Exact numbers from the DeepSeek example (d_k = d_v = 128, BF16). GQA provides an 8× reduction with much less quality impact than MQA.*

---

## Key Expansion: How GQA Runs Attention

When a group of query heads shares one K and V, the K/V must be expanded to match the query batch before computing attention:

![GQA key expansion: block matrix expansion where each K is repeated h/g times to match query heads](../../assets/attentions/Screenshot%202026-03-17%20101226.png)

![GQA key expansion diagram 2: repeat_interleave expanding K1 to match Q1, Q2 and K2 to match Q3, Q4](../../assets/attentions/Screenshot%202026-03-17%20101238.png)

```python
def expand_kv_for_gqa(K, num_query_heads):
    """
    K: (batch, num_kv_heads, seq_len, d_k)
    Returns: (batch, num_query_heads, seq_len, d_k)
    """
    num_kv_heads = K.shape[1]
    heads_per_group = num_query_heads // num_kv_heads
    # Repeat each KV head to match the query heads in its group
    return K.repeat_interleave(heads_per_group, dim=1)

# Example: 4 query heads, 2 KV heads
# K shape: (B, 2, N, d_k) → expand → (B, 4, N, d_k)
# K1 copied to positions 0,1 (group 1); K2 copied to positions 2,3 (group 2)
```

At training time, the gradients flow through the expanded K/V back to the single K/V parameters — all query heads in a group share the same K/V gradient.

---

## The Low-Rank Interpretation — Bridge to MLA

GQA can be understood as a **structured low-rank factorization** of the full MHA KV projection matrices.

In MHA, the KV projection has shape `d_model × (d_k × h)`. In GQA with g groups, effectively:

```
W_K_full (d × d_k × h) = W_K_gqa (d × d_k × g)  ⊗  repeat_interleave
```

![Combined KV projection W^KV showing grouped structure](../../assets/attentions/Screenshot%202026-03-17%20101316.png)

![Low-rank factorization view of GQA: the g KV heads form a low-rank approximation to the full h-head KV projection](../../assets/attentions/Screenshot%202026-03-17%20101403.png)

The key insight: GQA's reduction is **structured** — it fixes a block-replication pattern (K1 is shared by exactly heads 1,2,...,h/g). The blocks are not learned independently.

**MLA (Lesson 3.1)** takes this further: instead of a structured block pattern, it learns a **fully unstructured low-rank compression** of K and V — more expressive with potentially fewer bits.

---

## Weight Transfer: Converting MHA → GQA

Ainslie et al. (2023) propose a practical conversion of existing MHA checkpoints to GQA:

1. Take h KV head weights `{Wk_1, ..., Wk_h}` and `{Wv_1, ..., Wv_h}`
2. Group into g groups of h/g heads each
3. Mean-pool the Wk weights within each group: `Wk_group_g = mean(Wk_{g*(h/g) : (g+1)*(h/g)})`
4. Fine-tune for ~5% of original training compute to recover quality

This makes GQA practical for teams that want to reduce KV cache of an already-trained MHA model without full retraining.

---

## Production Models: Who Uses What

| Model | KV Architecture | Details |
|---|---|---|
| **LLaMA-1** | MHA | 32 heads, 32 KV heads |
| **LLaMA-2 7B** | MHA | 32 heads, 32 KV heads |
| **LLaMA-2 70B** | GQA | 64 query heads, 8 KV heads (8× reduction) |
| **LLaMA-3 8B** | GQA | 32 query heads, 8 KV heads |
| **LLaMA-3 70B** | GQA | 64 query heads, 8 KV heads |
| **Mistral 7B** | GQA | 32 query heads, 8 KV heads |
| **Falcon 7B** | MQA | 71 query heads, 1 KV head |
| **Falcon 40B** | MQA | 128 query heads, 8 KV heads (actually GQA-like) |
| **Gemma 2B** | MQA | — |
| **DeepSeek-R1/V3** | MLA | (next lesson) |

GQA with 4–8 KV heads is the current standard for frontier open-source models.

---

## Limitations of GQA

**1. Quality gap vs MHA at aggressive reductions:**
GQA(g=1) = MQA has real quality loss. Even GQA(g=2) can show degradation on tasks requiring the model to simultaneously track multiple relationship types. The optimal g depends on the task and model scale.

**2. Structured replication is not fully expressive:**
GQA's KV sharing uses a fixed block pattern — heads 1 and 2 always share K1 and V1. The model cannot learn which heads *should* share information. MLA (Lesson 3.1) solves this by using learned low-rank projections rather than structured repetition.

**3. Still grows linearly with sequence length:**
Even with g=8 KV heads instead of h=32, KV cache still grows as O(S) — just with a smaller constant. For very long contexts (128K+), it remains substantial.

**4. Head group design is a hyperparameter:**
Choosing g requires experimentation. Too few groups (small g) → quality drop. Too many groups (large g → MHA) → no memory benefit.

---

## Summary

- **MQA** shares one K/V across all query heads → 128× KV cache reduction, significant quality loss
- **GQA** shares one K/V across g groups of query heads → configurable trade-off; GQA(g=1)=MQA, GQA(g=h)=MHA
- Memory: MHA 4 MB → GQA 500 KB → MQA 31 KB per token (DeepSeek example)
- KV expansion via `repeat_interleave` is how GQA is implemented efficiently
- GQA has a **low-rank interpretation** — bridge to MLA (Lesson 3.1) which replaces structured repetition with learned compression
- GQA with 4–8 KV heads is the dominant architecture in current frontier models (LLaMA-3, Mistral)

---

## Interview Q&A

**Q: What is the difference between MQA and GQA?**
MQA is the extreme case: all query heads share exactly one K/V pair. GQA generalizes this: g groups, each sharing one K/V. GQA(g=1) = MQA, GQA(g=h) = MHA. GQA is more flexible and achieves a better quality-memory trade-off.

**Q: Why didn't MQA replace MHA entirely?**
MQA degrades quality significantly, especially on reasoning and multi-faceted tasks. The single shared K/V cannot represent multiple distinct relationship types simultaneously. GQA found a middle ground that retains most quality while achieving meaningful KV reduction.

**Q: How do you choose the number of KV groups g?**
Empirically — typically g is chosen so that the KV reduction is 4–8× (g = h/4 or h/8). LLaMA-3 and Mistral use g = h/4. Larger models can tolerate more aggressive reduction (larger h gives more redundancy to exploit).

**Q: Does GQA change the forward pass during training?**
Yes. During training, gradients from all query heads in a group flow back to the single shared K/V projection — it's trained with the GQA structure from the start (or fine-tuned from MHA via the mean-pooling conversion).

**Q: What is the "low-rank interpretation" of GQA?**
GQA's g KV projections can be seen as a rank-g approximation of the full h-head KV projection. MHA has h independent KV heads (rank h); GQA has g groups (rank g < h). But the compression is structured (block replication), not learned. MLA (Lesson 3.1) learns the compression instead.
