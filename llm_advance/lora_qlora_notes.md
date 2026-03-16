# LoRA & QLoRA — Complete Interview + Implementation Notes

> **Prerequisite:** These notes assume you already have notes on Quantization and its variants (INT8, FP16, NF4, etc.). Cross-references are made where relevant.

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [LoRA — Low-Rank Adaptation](#2-lora--low-rank-adaptation)
3. [Why Low-Rank? The Linear Algebra Intuition](#3-why-low-rank-the-linear-algebra-intuition)
4. [LoRA Variants (Interview Depth)](#4-lora-variants-interview-depth)
5. [QLoRA — Quantized Low-Rank Adaptation](#5-qlora--quantized-low-rank-adaptation)
6. [LoRA vs QLoRA — Comparison](#6-lora-vs-qlora--comparison)
7. [LoRA vs Other PEFT Methods](#7-lora-vs-other-peft-methods)
8. [LoRA at Inference Time](#8-lora-at-inference-time)
9. [Multi-task & Multi-Adapter LoRA](#9-multi-task--multi-adapter-lora)
10. [Evaluation & Validation](#10-evaluation--validation)
11. [Practical Implementation](#11-practical-implementation)
12. [Failure Modes](#12-failure-modes)
13. [Limitations](#13-limitations)
14. [Summary & Key Takeaways](#14-summary--key-takeaways)

---

## 1. Introduction & Motivation

### 1.1 Why Fine-Tuning LLMs Is Expensive

Modern Large Language Models (LLMs) like GPT-3, LLaMA-2, or Falcon contain **billions of parameters**. Full fine-tuning means updating every single parameter during training.

**Example — LLaMA-2 7B:**

| Item | Size |
|------|------|
| Model parameters | 7 billion |
| FP32 weights | ~28 GB |
| FP32 gradients | ~28 GB |
| FP32 optimizer states (Adam: 2 moments) | ~56 GB |
| **Total GPU memory needed** | **~112 GB** |

This exceeds even high-end data center GPUs (A100 = 80 GB). For a 70B model, this becomes completely infeasible for most teams.

**Cost dimensions:**
- **Memory:** Gradients + optimizer states dwarf the model itself
- **Compute:** Backprop through all layers
- **Time:** Weeks of training on large clusters
- **Storage:** Each fine-tuned version = a full copy of the model

### 1.2 The Parameter Efficiency Problem

The core question:

> *Do we really need to update ALL parameters to teach a model a new task?*

Empirical evidence says **no**. Fine-tuning updates tend to live in a much lower-dimensional subspace than the full parameter space. This observation is the seed of Parameter-Efficient Fine-Tuning (PEFT).

**Key insight from Li et al. (2018) — Intrinsic Dimensionality:**
Many optimization problems in deep learning can be solved effectively in a surprisingly small subspace. Fine-tuning is one of them.

### 1.3 Where LoRA Fits in the PEFT Landscape

```
PEFT Methods
├── Additive Methods
│   ├── Adapter Layers       → Insert small bottleneck modules between layers
│   ├── Prefix Tuning        → Prepend learnable tokens to key/value sequences
│   └── Prompt Tuning        → Learn soft prompt embeddings only
├── Selective Methods
│   └── Sparse Fine-tuning   → Only update a subset of original weights
└── Reparameterization Methods
    └── LoRA ✓               → Decompose weight updates into low-rank matrices
        ├── AdaLoRA
        ├── LoRA+
        └── DoRA
```

**LoRA's positioning:**
- No added inference latency (weights can be merged)
- No new architecture components at inference time
- Competitive quality with much lower memory footprint
- Works across model families (BERT, GPT, LLaMA, T5, etc.)

---

## 2. LoRA — Low-Rank Adaptation

### 2.1 Core Intuition — Why Low-Rank Works

Imagine you have a pre-trained weight matrix **W₀** (e.g., the query matrix in an attention head). During fine-tuning, you learn a change **ΔW** to this matrix.

**The hypothesis:** The update **ΔW** has a low *intrinsic rank*. That is, even though **ΔW** lives in a high-dimensional space, it can be well-approximated by a matrix of much lower rank.

**Analogy:**
Think of a photo. A 1000×1000 pixel image has 1,000,000 values. But if the image is mostly a plain background with a simple shape, you can compress it to a much smaller representation (like JPEG) without losing important details. LoRA applies the same logic to weight updates.

**Formal statement:**
If **ΔW ∈ ℝ^(d×k)** and its true rank is **r << min(d, k)**, then:

```
ΔW ≈ B · A    where B ∈ ℝ^(d×r),  A ∈ ℝ^(r×k)
```

Instead of storing **d × k** values, we store **(d × r) + (r × k)** values.

**Parameter savings example:**
- d = 4096, k = 4096, r = 8
- Full ΔW: 4096 × 4096 = **16,777,216** parameters
- LoRA: (4096×8) + (8×4096) = **65,536** parameters
- **Reduction: ~256×**

### 2.2 Mathematical Formulation

#### Standard Forward Pass (no LoRA):

```
h = W₀ · x
```

where **W₀ ∈ ℝ^(d×k)** is the frozen pre-trained weight.

#### LoRA Modified Forward Pass:

```
h = W₀ · x + ΔW · x
  = W₀ · x + B · A · x
```

where:
- **W₀** is **frozen** (no gradients computed)
- **A ∈ ℝ^(r×k)** — the "down-projection" matrix
- **B ∈ ℝ^(d×r)** — the "up-projection" matrix
- **r** is the rank (hyperparameter, typically 4–64)

#### Scaling Factor α:

In practice, the update is scaled:

```
h = W₀ · x + (α/r) · B · A · x
```

- **α** (alpha) is a scaling hyperparameter (often set equal to r, making the scale = 1)
- The **α/r** term controls how much the adapter influences the output
- Higher α → stronger LoRA influence
- This is analogous to a learning rate specifically for the adapter

#### Initialization Strategy:

| Matrix | Initialization | Reason |
|--------|---------------|---------|
| **A** | Random Gaussian (σ=0.02) | Introduces variability for learning |
| **B** | **Zero** | Ensures ΔW = B·A = 0 at start → model begins identical to pre-trained |

**Critical insight:** B is initialized to zero so that at t=0, the fine-tuned model is exactly the original pre-trained model. Training starts from a known good state.

#### Dropout:

A dropout layer is optionally applied to the input of A:

```
h = W₀ · x + (α/r) · B · Dropout(A · x)
```

Dropout rate is typically 0.05–0.1, used to regularize the adapter.

### 2.3 Forward Pass with LoRA — Step by Step

Given input **x ∈ ℝ^k**:

```
Step 1: Compute frozen path      →  z₁ = W₀ · x         [d-dimensional]
Step 2: Down-project             →  z₂ = A · x           [r-dimensional]  ← small!
Step 3: Up-project               →  z₃ = B · z₂          [d-dimensional]
Step 4: Scale                    →  z₄ = (α/r) · z₃
Step 5: Add (residual)           →  h  = z₁ + z₄
```

**Memory savings during training:**
- Only A and B accumulate gradients
- W₀ requires no gradient storage
- Optimizer states (Adam moments) only for A and B

### 2.4 Where LoRA Is Applied — Attention Matrices

In a Transformer, each self-attention layer computes:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

where:
  Q = x · W_Q
  K = x · W_K
  V = x · W_V
  Output projection: W_O
```

**LoRA targets these weight matrices: W_Q, W_K, W_V, W_O**

```
Transformer Layer (with LoRA)
┌─────────────────────────────────────────┐
│  Input x                                │
│     │                                   │
│  ┌──┴──┐  LoRA on W_Q:                 │
│  │ W_Q │──→ Q = x·W_Q + x·Aᵩ·Bᵩ       │
│  └─────┘                                │
│  ┌─────┐  LoRA on W_K:                 │
│  │ W_K │──→ K = x·W_K + x·A_K·B_K     │
│  └─────┘                                │
│  ┌─────┐  LoRA on W_V:                 │
│  │ W_V │──→ V = x·W_V + x·A_V·B_V     │
│  └─────┘                                │
│  [Attention computation]                │
│  ┌─────┐  LoRA on W_O:                 │
│  │ W_O │──→ out = attn·W_O + attn·A_O·B_O │
│  └─────┘                                │
└─────────────────────────────────────────┘
```

**Original LoRA paper applies adapters to W_Q and W_V only** (found sufficient). Many implementations extend to all four.

### 2.5 Why NOT Other Layers? (And When You Should)

This is a commonly asked interview question.

#### Why LoRA Focuses on Attention Matrices

**Reason 1 — Attention captures task-specific relational structure**

Attention weights determine *which tokens attend to which*. Task adaptation (e.g., instruction following vs. code generation) primarily changes *what the model pays attention to*, not how it transforms individual token representations.

**Reason 2 — Attention matrices have large d×d dimensions**

For a 7B model with d_model=4096:
- Each of W_Q, W_K, W_V, W_O = 4096×4096 = 16M params
- Applying LoRA with r=8: 65K params per matrix (256× compression)
- FFN layers are even larger (4096×16384 = 67M each) but modifying them risks disrupting factual knowledge

**Reason 3 — FFN layers store factual knowledge**

Research (Geva et al., 2021 — "Transformer Feed-Forward Layers Are Key-Value Memories") shows that FFN layers act as *associative memories* storing factual associations. Modifying them aggressively can cause catastrophic forgetting of pre-trained knowledge.

**Reason 4 — Empirical evidence from ablations**

The original LoRA paper showed that applying LoRA to W_Q + W_V with moderate rank achieves near full-fine-tuning performance. Adding more matrices gave diminishing returns.

#### When You SHOULD Apply LoRA to Other Layers

| Layer | When to target it |
|-------|------------------|
| **FFN / MLP layers** | When the task requires learning new factual knowledge or domain-specific vocabulary patterns |
| **Embedding layers** | When fine-tuning on a new language or specialized vocabulary |
| **LayerNorm** | Almost never — these are tiny and critical for training stability |
| **All linear layers** | When maximum expressiveness is needed and memory allows (e.g., QLoRA makes this feasible) |

**Practical rule:** Start with W_Q + W_V. If underperforming, add W_K + W_O. If still underperforming, add FFN down/up projections.

### 2.6 Merging Weights at Inference

After training, LoRA adapters can be **merged** into the base weights:

```
W_merged = W₀ + (α/r) · B · A
```

This produces a standard model with **no inference overhead**. The merged model:
- Has identical architecture to the original
- Runs at the same speed as the original
- Requires no special LoRA-aware inference code

**When NOT to merge:** When you need to swap adapters dynamically (multi-task serving), keep A and B separate.

### 2.7 Hyperparameters

| Hyperparameter | Typical Range | Effect |
|----------------|--------------|--------|
| **r** (rank) | 4, 8, 16, 32, 64 | Higher r = more expressiveness, more params |
| **α** (alpha) | 8–64 (often = r or 2r) | Scales adapter output; α/r is the effective scale |
| **dropout** | 0.0–0.1 | Regularization; use 0.05 for small datasets |
| **target_modules** | Q, V (default) or all linear | Which weight matrices get adapters |
| **lora_bias** | none / all / lora_only | Whether to train bias terms |

**Rule of thumb for rank r:**
- Simple style transfer / chat fine-tuning: r = 4–8
- Domain adaptation: r = 16–32
- Complex task learning: r = 32–64
- Very complex / large vocabulary shift: r = 64–128
pter has shifted the output — the model has been fine-tuned without touching W₀.

### 2.9 Diagram: LoRA Adapter Architecture

```
                    INPUT x
                       │
          ┌────────────┼────────────┐
          │                         │
          ▼                         ▼
   ┌─────────────┐           ┌────────────┐
   │  W₀ (frozen)│           │  A (r×k)   │  ← trained, random init
   │  d×k matrix │           │  DOWN-PROJ │
   └──────┬──────┘           └─────┬──────┘
          │                        │
          │                        ▼
          │                 ┌────────────┐
          │                 │  B (d×r)   │  ← trained, zero init
          │                 │  UP-PROJ   │
          │                 └─────┬──────┘
          │                        │
          │                        ▼
          │                  × (α/r) scale
          │                        │
          ▼                        ▼
         z₁ ──────────────────────(+)
                                   │
                                   ▼
                               OUTPUT h
```

---

## 3. Why Low-Rank? The Linear Algebra Intuition

### 3.1 Intrinsic Dimensionality of Fine-Tuning Updates

**Core observation (Aghajanyan et al., 2020):**
Pre-trained models can be fine-tuned in a very low-dimensional subspace — sometimes as low as **d = 200** for a model with millions of parameters — with minimal performance loss.

**What this means practically:**
Fine-tuning doesn't require exploring the full parameter space. The "useful" updates cluster in a low-dimensional manifold.

**Why this happens:**
Pre-training has already learned rich representations. Fine-tuning only needs to *redirect* these representations, not build them from scratch. Redirection is a low-rank operation.

### 3.2 SVD Connection

**Singular Value Decomposition (SVD)** provides the theoretical foundation.

Any matrix **M ∈ ℝ^(d×k)** can be decomposed as:

```
M = U · Σ · Vᵀ

where:
  U ∈ ℝ^(d×d)  — left singular vectors
  Σ ∈ ℝ^(d×k)  — diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
  V ∈ ℝ^(k×k)  — right singular vectors
```

**Low-rank approximation:** Keep only the top-r singular values:

```
M ≈ Mᵣ = U[:, :r] · Σ[:r, :r] · Vᵀ[:r, :]
        = B_svd · A_svd
```

This is the best rank-r approximation of M (Eckart–Young theorem).

**LoRA connection:**
LoRA's B·A is a learned low-rank factorization of ΔW. It's not exactly SVD (A is not orthogonal), but the spirit is the same — represent the update in a compressed form.

**If singular values of ΔW decay rapidly:**
```
σ₁ >> σ₂ >> σ₃ >> ... ≈ 0
```
→ ΔW is well-approximated by low-rank, and LoRA works well.

**If singular values are roughly equal:**
```
σ₁ ≈ σ₂ ≈ σ₃ ≈ ... (flat spectrum)
```
→ ΔW is not low-rank, and LoRA will underfit. This is a failure mode (see §12).

### 3.3 Rank vs. Expressiveness Trade-off

```
Rank r │  Expressiveness  │  # Parameters  │  Risk
───────┼──────────────────┼─────────────────┼──────────────
  1    │  Very low        │  Minimal        │  Underfit
  4    │  Low             │  Low            │  Underfit (complex tasks)
  8    │  Moderate        │  Moderate       │  Good default
 16    │  Good            │  Moderate       │  Balanced
 32    │  High            │  High           │  Risk of overfit (small data)
 64    │  Very high       │  High           │  Approaches full fine-tune
full   │  Maximum         │  Full model     │  Catastrophic forgetting risk
```

**Key insight:** Higher rank does NOT always mean better performance. On small datasets, low rank acts as implicit regularization.

---

## 4. LoRA Variants (Interview Depth)

### 4.1 AdaLoRA — Adaptive Rank Allocation

**Problem with standard LoRA:**
All weight matrices get the same rank r. But not all matrices are equally important for a given task. Assigning uniform rank is wasteful.

**Key innovation:**
Dynamically allocate rank budget based on the *importance* of each weight matrix, using an SVD-based importance scoring.

**How it works:**

1. Parameterize ΔW using explicit SVD: **ΔW = P · Λ · Qᵀ**
   - P = left singular vectors
   - Λ = diagonal (singular values, learned)
   - Q = right singular vectors

2. Define importance score for each singular triplet (pᵢ, λᵢ, qᵢ):
   ```
   Sᵢ = |λᵢ| · (||pᵢ||₁ + ||qᵢ||₁)
   ```

3. Prune singular values with low importance scores to zero → effectively reduces rank

4. Redistribute the rank budget to more important matrices

**Result:** Critical matrices (often W_Q, W_V in later layers) get higher rank; less important matrices get rank 0 or 1.

**Interview angle:** AdaLoRA is better than LoRA when you have a fixed parameter budget and want to use it optimally across layers.

**When AdaLoRA fails:**
- Importance scores can be noisy early in training → bad pruning decisions
- Higher compute overhead (SVD at each step)
- Rank collapse: if pruning is too aggressive, some matrices lose all rank

### 4.2 LoRA+ — Asymmetric Learning Rates

**Problem with standard LoRA:**
Both A and B are trained with the same learning rate. But they play asymmetric roles:
- A projects *down* (k → r): learns to extract relevant features
- B projects *up* (r → d): learns to reconstruct the update

**Key insight (Hayou et al., 2024):**
Using the same learning rate for A and B is suboptimal due to their different scales and roles. B should have a *higher* learning rate than A.

**LoRA+ modification:**

```
η_B = λ · η_A    where λ > 1 (typically λ = 16)
```

Train B with learning rate **16× higher** than A.

**Why this works:**
- A collapses faster (smaller parameter space, more redundancy)
- B needs more learning signal to reconstruct high-dimensional updates
- The asymmetry matches the signal-to-noise ratio in each matrix

**Performance gain:** LoRA+ can achieve similar results with ~2× less training time or better results at the same compute budget.

**When LoRA+ fails:**
- Very small datasets: the aggressive B update can overfit
- λ is sensitive to task — may need tuning

### 4.3 DoRA — Weight-Decomposed Low-Rank Adaptation

**Problem:** LoRA modifies the *magnitude* and *direction* of weight updates simultaneously. This entanglement can make learning less stable and less efficient.

**Key innovation (Liu et al., 2024):**
Decompose the pre-trained weight into *magnitude* and *direction* components, then apply LoRA only to the *direction*.

**Decomposition:**

```
W = m · (V / ||V||_c)

where:
  m = ||W||_c     ← column-wise norm (magnitude vector, ∈ ℝ^(1×k))
  V / ||V||_c     ← direction matrix (unit column norms)
```

**DoRA fine-tuning:**

```
W' = (m + Δm) · ((W₀ + BA) / ||W₀ + BA||_c)
```

- **m** is a learnable magnitude vector (small, 1×k)
- **BA** is the standard LoRA adapter
- The direction is updated via LoRA, magnitude is updated directly

**Why this matters:**
- Direction updates = *what* the weight does (task-specific)
- Magnitude updates = *how strongly* it does it (scale)
- Separating them leads to more stable, efficient learning

**DoRA mimics full fine-tuning more closely** — the learning pattern of DoRA matches full fine-tuning's pattern (which also shows large directional change + small magnitude change).

**When DoRA fails:**
- Extra overhead of computing column norms at each step
- Unstable in early training if magnitude initialization is poor
- Not yet as widely adopted — library support is more limited

### 4.4 When to Prefer Each Variant

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| **Standard fine-tuning, unknown task** | LoRA | Simple, well-tested baseline |
| **Fixed parameter budget, multiple layers** | AdaLoRA | Allocates rank where it matters |
| **Training time is bottleneck** | LoRA+ | ~2× faster convergence |
| **Matching full fine-tune quality is critical** | DoRA | Closest behavior to full FT |
| **Extreme memory constraint** | QLoRA (§5) | 4-bit base + LoRA |
| **Production, simplicity is key** | LoRA | Mergeable, zero inference overhead |

### 4.5 Summary Comparison Table: LoRA Variants

| Variant | Key Innovation | Extra Overhead | Best For |
|---------|---------------|----------------|----------|
| LoRA | Low-rank ΔW = BA | None | General purpose |
| AdaLoRA | Adaptive rank per matrix | SVD computation | Parameter budget optimization |
| LoRA+ | η_B >> η_A | None (just config) | Faster training |
| DoRA | Separate magnitude/direction | Norm computation | Quality matching full FT |

---

## 5. QLoRA — Quantized Low-Rank Adaptation

### 5.1 Motivation — Fitting LoRA on Consumer Hardware

Even with LoRA, the **base model** must be loaded in full precision (FP16/BF16) to compute the frozen W₀·x path. For a 65B model:

| Model | FP16 Memory |
|-------|------------|
| LLaMA 7B | ~14 GB |
| LLaMA 13B | ~26 GB |
| LLaMA 33B | ~66 GB |
| LLaMA 65B | ~130 GB |

Even with LoRA, this is too large for a single consumer GPU (RTX 3090/4090 = 24 GB).

**QLoRA's solution (Dettmers et al., 2023):**
Quantize the base model to 4-bit, keep LoRA adapters in BF16/FP16, and dequantize on-the-fly during the forward pass.

**Result:** Fine-tune a 65B model on a **single 48 GB GPU** (A6000) with near full-precision quality.

### 5.2 Three Key Innovations

QLoRA introduces three novel components that work together:

```
QLoRA = NF4 Quantization + Double Quantization + Paged Optimizers
           ↓                      ↓                    ↓
    Better 4-bit dtype     Less memory per param   Handle memory spikes
```

### 5.3 Full QLoRA Pipeline

```
                    BASE MODEL (pre-trained)
                           │
                    ┌──────▼──────┐
                    │  Quantize   │  ← NF4 4-bit quantization
                    │  to NF4     │     (offline, before training)
                    └──────┬──────┘
                           │
          ┌────────────────▼────────────────────┐
          │          TRAINING FORWARD PASS        │
          │                                       │
          │  ┌──────────────────────────────┐    │
          │  │  W_NF4 (frozen, 4-bit)       │    │
          │  │       │                       │    │
          │  │  Dequantize to BF16          │    │
          │  │  (per-block, on the fly)     │    │
          │  │       │                       │    │
          │  │  z₁ = W_BF16 · x            │    │
          │  └──────────────────────────────┘    │
          │                                       │
          │  ┌──────────────────────────────┐    │
          │  │  LoRA Adapter (BF16)         │    │
          │  │  z₂ = B · A · x             │    │
          │  └──────────────────────────────┘    │
          │                                       │
          │         h = z₁ + (α/r) · z₂          │
          └─────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  BACKWARD   │
                    │  (only LoRA │
                    │   adapters) │
                    └─────────────┘
```

**Key points:**
- W₀ is stored as 4-bit NF4 — never updated
- Dequantization to BF16 happens per forward pass, per block
- Gradients only flow through A and B (LoRA adapters)
- Optimizer states only for A and B → tiny memory footprint

### 5.4 Mathematical Formulation of QLoRA Forward Pass

```
h = f_dequant(W_NF4, c₁, c₂) · x + (α/r) · B · A · x
```

where:
- **W_NF4** — 4-bit NormalFloat quantized weight
- **c₁** — per-block quantization constants (FP32)
- **c₂** — double-quantized constants for c₁ (FP8)
- **f_dequant** — dequantization function: maps NF4 → BF16

More precisely:

```
W_BF16 = dequant(W_NF4, c₁)    ← reconstruct BF16 weight per block
h = W_BF16 · x + (α/r) · B · A · x
```

### 5.5 NF4 Data Type — Intuition and Math

**Why not just use INT4?**
INT4 spaces values uniformly: {-8, -7, ..., 0, ..., 7}. But neural network weights follow a **normal distribution** (bell curve). Uniform spacing wastes precision near the tails where few values live, and underrepresents the dense region near zero.

**NF4 = NormalFloat 4-bit**

NF4 spaces values to be **information-optimal for normally distributed data**.

**Construction:**
1. Assume weights ~ N(0, 1) after normalization
2. Divide the normal distribution into **16 equal-area quantiles** (4-bit = 16 values)
3. Use the **quantile midpoints** as the NF4 code values

```
Step 1: Find quantile boundaries q_i for i = 0, 1, ..., 16
        q_i = Q_normal(i/16)   ← inverse CDF of N(0,1)

Step 2: NF4 value for bucket i = midpoint of (q_i, q_{i+1})
        v_i = (q_i + q_{i+1}) / 2

Step 3: Normalize so max(|v_i|) = 1
```

**NF4 values (approximate):**
```
{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0}
```

Notice: denser near 0, sparser at extremes — matching the normal distribution.

**Per-block scaling:**
Each block of 64 weights is scaled by its own constant c:
```
W_quant[i] = NF4_quantize(W[i] / c)    where c = max(|W[block]|)
```

This ensures each block uses the full [-1, 1] dynamic range.

### 5.6 Double Quantization

**Problem:**
NF4 uses per-block quantization constants (one FP32 constant per 64 weights).
Memory cost of these constants: 32 bits / 64 weights = **0.5 bits/weight**.
That's 12.5% overhead on top of the 4-bit weights — significant!

**Solution: Double Quantization**
Quantize the quantization constants themselves!

```
Level 1 (weights):
  W_4bit = NF4_quantize(W / c₁)    c₁ ∈ FP32, 1 per 64 weights

Level 2 (constants):
  c₁_quantized = FP8_quantize(c₁ / c₂)    c₂ ∈ FP32, 1 per 256 c₁ values
```

**Memory savings example:**

| Item | Before DQ | After DQ | Savings |
|------|-----------|----------|---------|
| Weights | 4 bits/param | 4 bits/param | — |
| Level-1 constants | 0.5 bits/param | ~0.127 bits/param | 0.373 bits |
| Level-2 constants | — | ~0.004 bits/param | — |
| **Total overhead** | **0.5 bits** | **~0.131 bits** | **~3.6× reduction** |

**Overall:** QLoRA uses approximately **4.13 bits/parameter** vs. 4.5 bits without double quantization — a ~9% memory reduction on constant storage.

### 5.7 Paged Optimizers and Memory Spike Handling

**Problem:**
GPU memory usage is not constant during training. When processing long sequences or large batches, gradient checkpointing and optimizer updates cause **memory spikes** that can cause OOM errors even when average usage is fine.

**Solution: Paged Optimizers (NVIDIA unified memory)**

QLoRA uses NVIDIA's unified memory paging to handle optimizer states:

```
┌─────────────┐     overflow      ┌──────────────┐
│  GPU VRAM   │ ────────────────► │  CPU RAM     │
│  (primary)  │ ◄──────────────── │  (overflow)  │
│             │    page back in   │              │
└─────────────┘                   └──────────────┘
```

**How it works:**
1. Optimizer states (Adam moments for A, B) normally live on GPU
2. During memory spikes, pages are automatically evicted to CPU RAM
3. Pages are retrieved back to GPU when needed
4. Uses NVIDIA's `cudaMallocManaged` API

**Cost:** Minor slowdown (~1-5%) during paging events, but prevents OOM crashes.

**Why only for optimizer states?**
- Model weights and activations must stay on GPU for compute efficiency
- Optimizer states are accessed less frequently (only during weight updates)
- Adam's moment buffers are the largest optimizer memory component

### 5.8 Diagram: QLoRA Full Training Pipeline

```
STORAGE LAYOUT DURING TRAINING:
──────────────────────────────────────────────────────────
GPU VRAM
  ├── W_NF4 (4-bit weights)        ← ~4.13 bits/param
  ├── LoRA A, B (BF16)             ← tiny (r << d)
  ├── Activations (BF16)           ← temporary, per forward pass
  └── Dequantized blocks (BF16)    ← temporary, per computation

CPU RAM (paged)
  └── Adam moments for A, B        ← paged in/out as needed
──────────────────────────────────────────────────────────

MEMORY COMPARISON (LLaMA 65B):
  Full FT (FP16):      ~130 GB weights + ~260 GB optimizer = ~390 GB
  LoRA (BF16 base):    ~130 GB weights + ~1 GB adapters = ~131 GB
  QLoRA (NF4 base):    ~32.5 GB weights + ~1 GB adapters = ~33.5 GB ✓
──────────────────────────────────────────────────────────
```

---

## 6. LoRA vs QLoRA — Comparison

| Dimension | LoRA | QLoRA |
|-----------|------|-------|
| **Base model precision** | BF16 / FP16 | NF4 (4-bit) |
| **Adapter precision** | BF16 / FP16 | BF16 / FP16 |
| **Memory (7B model)** | ~14 GB | ~5 GB |
| **Memory (65B model)** | ~130 GB | ~35 GB |
| **Training speed** | Faster (no dequant overhead) | ~30% slower |
| **Quality vs full FT** | ~95-99% | ~93-97% |
| **Quality vs LoRA** | Baseline | Slightly lower (quantization error) |
| **Hardware requirement** | High-end GPU (A100) | Consumer GPU (RTX 3090) |
| **Inference (merged)** | No overhead | No overhead (merge to BF16) |
| **Setup complexity** | Moderate | Higher (bitsandbytes dependency) |

**When to use LoRA:**
- You have access to A100/H100 GPUs
- Training speed matters
- Maximum quality is critical

**When to use QLoRA:**
- Limited to consumer or mid-tier GPUs
- Fine-tuning large models (13B+) on single GPUs
- Quality slightly below LoRA is acceptable

---

## 7. LoRA vs Other PEFT Methods

### 7.1 Prefix Tuning

**Mechanism:** Prepend **learnable continuous vectors** (the "prefix") to the key and value sequences of every attention layer.

```
Normal attention:  Attention(Q, K, V)
Prefix attention:  Attention(Q, [P_K; K], [P_V; V])

where P_K, P_V are learned prefix matrices prepended to K and V
```

**Key properties:**
- No modification to model weights
- Adds context/instruction implicitly through the prefix
- Prefix is essentially "virtual tokens" that steer attention
- Original model is untouched — prefix is the only addition

**Parameters:** `prefix_length × num_layers × d_model × 2` (for K and V)

### 7.2 Prompt Tuning

**Mechanism:** Learns a set of **soft prompt embeddings** prepended to the input embedding layer only (not all layers like Prefix Tuning).

```
Input: [P₁, P₂, ..., Pₙ, x₁, x₂, ..., xₜ]
         └── learnable ──┘  └── frozen ──┘
```

**Simpler than Prefix Tuning:** Only the input layer is modified. At large model scale (11B+), quality approaches full fine-tuning.

### 7.3 Adapter Layers

**Mechanism:** Insert small **bottleneck modules** inside each Transformer layer, between sub-layers.

```
Sub-layer output → LayerNorm → Down-project (d→m) → Activation → Up-project (m→d) → Residual
```

where m << d (bottleneck dimension).

**Key properties:**
- Adds new parameters as new layers
- Original weights are frozen
- **Inference overhead:** Each adapter adds 2 matrix multiplications per layer
- Weights cannot be merged into original model (unlike LoRA)

### 7.4 Full Comparison Table

| Dimension | LoRA | Prefix Tuning | Prompt Tuning | Adapter Layers |
|-----------|------|--------------|---------------|----------------|
| **Where parameters added** | Inside weight matrices | K,V prefixes in all layers | Input embeddings only | Between sub-layers |
| **Modifies weights?** | No (additive) | No | No | No |
| **Mergeable (zero inference overhead)?** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Inference latency** | Zero (if merged) | Longer sequences → more compute | Longer sequences | Extra FF per layer |
| **Task performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (scales with model size) | ⭐⭐⭐⭐ |
| **Parameter count** | Low–Medium | Low | Very low | Low–Medium |
| **Interpretability** | Low | Low | Very low | Low |
| **Works well for** | General tasks, generation | NLG tasks, structured generation | Large models, classification | Smaller models, stable tasks |
| **Main weakness** | Fixed rank bottleneck | Longer context window needed | Weak on small models | Inference latency |

### 7.5 Why LoRA Dominates in Practice

1. **Zero inference overhead** — Adapters can be merged. Prefix/Prompt/Adapter all add compute at inference.
2. **Better quality across model sizes** — Unlike Prompt Tuning, LoRA works well even on 1B-7B models.
3. **Flexible targeting** — Can be applied to any weight matrix (Q, K, V, O, FFN).
4. **QLoRA extension** — Enables extreme memory compression, impossible for other methods without significant quality loss.
5. **Ecosystem support** — HuggingFace PEFT, LLaMA-Factory, Axolotl all have first-class LoRA support.
6. **Multi-adapter serving** — Multiple LoRA adapters can be hot-swapped on one base model (e.g., S-LoRA).

---

## 8. LoRA at Inference Time

### 8.1 Merged vs Unmerged Adapter Weights

**Option 1: Merged (recommended for production)**

```python
# Merge LoRA into base model weights
model = model.merge_and_unload()
# Now model has no LoRA components — standard forward pass
```

- W_merged = W₀ + (α/r) · B · A
- Zero inference overhead
- Identical output to unmerged
- Cannot swap adapters dynamically

**Option 2: Unmerged (recommended for multi-adapter serving)**

```
h = W₀ · x + (α/r) · B · A · x
```

- Two forward passes through the adapter (extra compute)
- But adapters can be swapped without reloading the base model
- Critical for multi-tenant serving

### 8.2 Latency Implications

**Merged LoRA:**
- Inference latency = base model latency (no change)
- Memory = base model memory (no LoRA stored)

**Unmerged LoRA:**
- Extra compute: 2 matrix multiplications per adapted layer
- For r=8, d=4096: A·x costs 8×4096 = 32K flops; B·(A·x) costs 4096×8 = 32K flops
- Total extra: ~64K flops per adapted layer
- For comparison, W₀·x: 4096×4096 = 16M flops
- Overhead ≈ **0.4% per layer** — negligible at low rank

### 8.3 Serving Multiple LoRA Adapters on One Base Model

**Use case:** You have one base model (e.g., LLaMA-2 7B) and 100 customer-specific fine-tunes. You want to serve all 100 without loading 100 separate 7B models.

**Solution: S-LoRA (Sheng et al., 2023) / vLLM LoRA serving**

```
┌─────────────────────────────┐
│  Base Model (GPU, shared)   │
│  LLaMA-2 7B (NF4 or BF16)  │
└─────────────┬───────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
  Adapter₁  Adapter₂  Adapter₃   ← stored in CPU RAM
  (user A)  (user B)  (user C)
    │
    └── paged to GPU on demand
```

**How it works:**
1. Base model stays on GPU permanently
2. LoRA adapters (A, B matrices) are stored on CPU
3. At request time, the required adapter is loaded to GPU
4. Batched requests with the same adapter are grouped

**Memory math (serving 100 adapters, r=16):**
- Each adapter: 2 × 4096 × 16 × 28 layers × 2 bytes ≈ 14.7 MB
- 100 adapters: ~1.47 GB on CPU (vs. 100 × 14 GB = 1.4 TB for separate models)

---

## 9. Multi-task & Multi-Adapter LoRA *(Brief)*

### 9.1 Training Multiple LoRA Adapters

**Approach 1: Independent adapters per task**
- Train separate A_i, B_i for each task i
- At inference, load the task-specific adapter
- Simple but no knowledge sharing between tasks

**Approach 2: Shared base + task-specific adapters**
- One frozen base model
- Each task has its own LoRA (A_i, B_i)
- Lightweight: only adapter switches between tasks

### 9.2 Adapter Merging Strategies

**Linear merging (Model Merging / TIES / DARE):**

```
W_merged = W₀ + Σᵢ λᵢ · (α/r) · Bᵢ · Aᵢ
```

Weight each adapter by λᵢ (tuned on validation sets). Useful for combining complementary skills.

**Problems with naive merging:**
- Interference between adapters trained on different tasks
- TIES-merging and DARE address this by handling sign conflicts and pruning small weights

### 9.3 LoRAHub — Composition Without Retraining

**LoRAHub** allows combining pre-trained LoRA adapters via learned coefficients:

```
ΔW = Σᵢ wᵢ · Bᵢ · Aᵢ
```

The weights wᵢ are found via gradient-free optimization (just a few forward passes on a small calibration set). No GPU training required.

**Use case:** Rapid task adaptation by composing community-contributed adapters.

---

## 10. Evaluation & Validation

### 10.1 How to Verify Fine-Tune Quality

**Baseline comparisons (always run these):**

| Baseline | Purpose |
|----------|---------|
| Zero-shot base model | Lower bound — how good was base before fine-tuning? |
| Few-shot base model | Is LoRA better than just prompting? |
| Full fine-tune (if feasible) | Upper bound — how much quality gap vs LoRA? |

**Task-specific metrics:**

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1, AUC |
| Generation (open-ended) | ROUGE, BLEU, BERTScore |
| Instruction following | MT-Bench, AlpacaEval, human eval |
| Code generation | pass@k, HumanEval |
| Reasoning | Accuracy on benchmarks (GSM8K, MATH, BBH) |

### 10.2 Key Signals to Track During Training

| Signal | Healthy | Warning Sign |
|--------|---------|--------------|
| Training loss | Decreasing smoothly | Oscillating or NaN |
| Validation loss | Decreasing, close to train | Increasing while train decreases → overfit |
| Gradient norm | Stable (0.1–10) | Exploding → NaN; vanishing → no learning |
| Adapter weight norms (||B·A||) | Growing slowly | Growing explosively → unstable |
| Singular value spectrum of B·A | Spread across dims | Collapsing to 1 dim → rank collapse |

### 10.3 Overfitting Signals Specific to LoRA

LoRA overfitting is subtle because the adapter is small. Watch for:

1. **Val loss diverging from train loss** on small datasets
2. **High perplexity on held-out distribution** while task metric looks good (memorization)
3. **Output repetition** — model repeats training examples verbatim
4. **Loss of base model capabilities** — test on pre-training benchmarks before/after

**Mitigations:**
- Increase LoRA dropout (0.05 → 0.1)
- Reduce rank r
- Add weight decay on adapter parameters
- Use early stopping based on validation metric

---

## 11. Practical Implementation

### 11.1 Key Libraries

| Library | Role |
|---------|------|
| `peft` (HuggingFace) | LoRA adapter management, `get_peft_model`, adapter saving/loading |
| `bitsandbytes` | 4-bit / 8-bit quantization, NF4, paged optimizers |
| `transformers` | Base model loading, tokenizer, training |
| `trl` | SFTTrainer, reward modeling, RLHF utilities |
| `accelerate` | Multi-GPU / mixed precision training |

### 11.2 Choosing Rank, Alpha, and Target Modules

**Rank selection guide:**

```
Dataset size < 1K examples      → r = 4-8   (low rank = regularization)
Dataset size 1K–10K examples    → r = 16-32
Dataset size > 10K examples     → r = 32-64
Task is very different from PT  → r = 64-128
Task is close to PT (style)     → r = 4-8
```

**Alpha selection:**
- Start with **α = r** (scale = 1.0) — this is the safest default
- If underfitting: increase α (α = 2r gives scale = 2.0)
- If overfitting/unstable: decrease α

**Target modules by model family:**

| Model | Default targets | Extended targets |
|-------|----------------|-----------------|
| LLaMA / Mistral | q_proj, v_proj | q_proj, k_proj, v_proj, o_proj |
| GPT-2 | c_attn, c_proj | c_attn, c_proj, mlp.c_fc |
| T5 | q, v | q, k, v, o, wi, wo |
| BERT | query, value | query, key, value, dense |

### 11.3 Code: 4-bit QLoRA Setup with bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization (NF4 + double quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                        # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",               # Use NF4 (not fp4)
    bnb_4bit_compute_dtype=torch.bfloat16,   # Compute in BF16 after dequant
    bnb_4bit_use_double_quant=True,          # Enable double quantization
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                        # Auto-distribute across GPUs
    trust_remote_code=True,
)

# Required for gradient checkpointing with 4-bit
model.config.use_cache = False
model.enable_input_require_grads()
```

### 11.4 Code: Attaching LoRA Adapters via PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Prepare 4-bit model for LoRA training (adds cast hooks, etc.)
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                              # Rank
    lora_alpha=32,                     # Alpha (scale = alpha/r = 2.0)
    lora_dropout=0.05,                 # Dropout on adapter input
    bias="none",                       # Don't train bias terms
    target_modules=[                   # Which matrices to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj",
    ],
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameter count
model.print_trainable_parameters()
# Output: trainable params: 33,554,432 || all params: 3,773,063,168 || trainable%: 0.89%
```

### 11.5 Code: Training Config + Saving & Merging Adapters

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Training configuration
training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,         # Effective batch = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,                            # Use BF16 for stability
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",             # Paged optimizer (QLoRA)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

# Train with SFTTrainer (handles LoRA + quantized models)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

trainer.train()

# Save LoRA adapter only (not full model)
model.save_pretrained("./lora-adapter")

# ── Merging at inference time ────────────────────────────────
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Load and merge adapter
merged_model = PeftModel.from_pretrained(base_model, "./lora-adapter")
merged_model = merged_model.merge_and_unload()         # W_merged = W₀ + BA
merged_model.save_pretrained("./merged-model")         # Save full merged model
```

### 11.6 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Not calling `prepare_model_for_kbit_training` | NaN loss, training crashes | Always call before `get_peft_model` |
| `use_cache=True` with gradient checkpointing | Error during backward pass | Set `use_cache=False` before training |
| Wrong `target_modules` names | 0 trainable parameters | Check with `model.named_modules()` |
| Too high learning rate | Loss explodes, NaN | Use 1e-4 to 2e-4 for LoRA |
| Mixing FP16 and BF16 | Subtle quality degradation | Use BF16 throughout for Ampere+ GPUs |
| Merging before evaluation | Can't swap adapters | Merge only for final deployment |
| Rank too high on small data | Overfitting | Reduce r, add dropout |

---

## 12. Failure Modes

### 12.1 When LoRA Fails

**Failure 1: Rank Too Low → Underfitting**

*Symptoms:* High validation loss, model doesn't learn task-specific behavior, output still resembles base model.

*Cause:* ΔW's true intrinsic rank > r. The low-rank approximation can't capture enough of the required change.

*Diagnosis:* Check singular value spectrum of B·A after training. If all variance is in the top 1-2 singular values, rank may be too low.

*Fix:* Increase rank (r: 8 → 16 → 32). Add more target modules.

---

**Failure 2: Rank Too High + Small Dataset → Overfitting**

*Symptoms:* Train loss decreases, val loss increases. Model memorizes training examples.

*Cause:* With high rank and little data, the adapter has too much capacity.

*Fix:* Reduce rank. Increase dropout. Add L2 regularization on adapter weights.

---

**Failure 3: Wrong Target Modules**

*Symptoms:* 0 trainable parameters, model doesn't change at all after training.

*Cause:* `target_modules` names don't match the actual module names in the model.

*Fix:* Run `{name: type for name, module in model.named_modules()}` to find correct names.

---

**Failure 4: Catastrophic Forgetting of General Capabilities**

*Symptoms:* Fine-tuned model performs well on target task but fails on previously-solved tasks (reasoning, factual QA, etc.).

*Cause:* LoRA with high rank and many training steps on narrow data can shift representations away from general knowledge.

*Fix:* Mix target task data with general instruction data. Reduce steps. Use smaller rank.

---

**Failure 5: Task Requires Structural Representation Changes**

*Symptoms:* Even with high rank, LoRA cannot match full fine-tuning. Gap > 5%.

*Cause:* The task requires learning fundamentally new token relationships that cannot be captured by modifying attention alone (e.g., learning a new programming language from scratch, extreme domain shift).

*Fix:* Apply LoRA to FFN layers too. Or use full fine-tuning with QLoRA to save memory.

### 12.2 When QLoRA Fails

**Failure 1: Quantization Error Accumulation**

*Symptoms:* QLoRA achieves notably lower quality than LoRA (more than expected ~1-2% gap).

*Cause:* The NF4 quantization error in W₀ propagates through the forward pass. For models with outlier weights (large values in specific dimensions), NF4's per-block scaling can be suboptimal.

*Fix:* Use `bnb_4bit_quant_type="nf4"` (not `"fp4"`). Consider 8-bit quantization if 4-bit quality is insufficient.

---

**Failure 2: Architecture Incompatibility with bitsandbytes**

*Symptoms:* Errors loading model, quantization applied to unsupported layers, segfaults.

*Cause:* Some custom model architectures use non-standard linear layer implementations incompatible with bitsandbytes' CUDA kernels.

*Fix:* Check bitsandbytes compatibility list. Use `skip_modules` parameter to exclude problematic layers.

---

**Failure 3: Paged Optimizer Memory Fragmentation**

*Symptoms:* Gradual memory creep over training, eventually OOM despite paged optimizers.

*Cause:* CUDA unified memory pages not being released properly, especially with long training runs.

*Fix:* Periodic CUDA cache clearing. Use `torch.cuda.empty_cache()` at checkpoint intervals.

### 12.3 When Variants Fail

**AdaLoRA:**
- **Rank collapse:** Pruning too aggressive in early training → some matrices lose all rank → irreversible. *Fix:* Increase warmup steps before rank pruning begins. Use `orth_reg_weight` for regularization.
- **High overhead:** SVD at each step is expensive. *Fix:* Only practical for small models or when parameter budget is very tight.

**LoRA+:**
- **Aggressive B updates on small data:** λ=16 can cause B to overfit quickly. *Fix:* Reduce λ to 4-8 for small datasets.
- **Instability with very high learning rates:** Combined with λ, effective LR for B can be too high. *Fix:* Lower base LR.

**DoRA:**
- **Early training instability:** Magnitude and direction update in opposite directions initially. *Fix:* Use warmup with DoRA only applied after initial training steps.
- **Library support:** Not yet in mainstream PEFT as of 2024 — may require custom implementation.

### 12.4 Solutions & Mitigations Summary

| Problem | Quick Diagnosis | Solution |
|---------|----------------|---------|
| Underfitting | High val loss, no task adaptation | ↑ rank, ↑ target modules, ↑ training steps |
| Overfitting | Val loss > train loss | ↓ rank, ↑ dropout, ↓ steps, mix data |
| Catastrophic forgetting | Good task, bad general | Mix with general data, ↓ rank, ↓ steps |
| OOM during QLoRA | CUDA out of memory | ↓ batch size, ↑ grad accumulation, ↓ seq length |
| No learning (0 grads) | Loss doesn't change | Check target_modules names, check grad hooks |
| NaN loss | Loss → NaN after early steps | ↓ LR, check for FP16 overflow, use BF16 |

---

## 13. Limitations

### 13.1 Inherent LoRA Limitations

These are **by design** — no hyperparameter tuning can resolve them:

**1. Fixed-rank bottleneck:**
LoRA commits to a fixed rank r before training. If the optimal rank is unknown (common in practice), you either under- or over-parameterize. There is no mechanism for the rank to grow during training.

**2. Cannot fully match full fine-tuning on extreme domain shifts:**
When the target domain is radically different from pre-training data (e.g., fine-tuning a general LLM on highly specialized scientific notation, or a new language the model has never seen), LoRA's constrained update space may not be sufficient. The adapter can only redirect existing representations — it cannot build entirely new ones.

**3. Rank-expressiveness-efficiency triangle:**
You cannot simultaneously have low rank (efficient), high expressiveness (quality), and fast convergence. Increasing any one requires compromising another.

**4. No dynamic capacity:**
Standard LoRA uses the same rank throughout training. Even if the task only needs high rank early (rapid adaptation) and low rank later (fine refinement), LoRA cannot adjust. (AdaLoRA partially addresses this.)

**5. Layer-specific rank may be suboptimal:**
All adapted layers get the same rank r. But empirically, different layers have different intrinsic dimensionality. Uniform rank allocation is fundamentally wasteful.

**6. Adapter interference in multi-task scenarios:**
When merging adapters from different tasks, there is no guarantee of orthogonality. Merged adapters can interfere destructively, degrading performance on individual tasks.

### 13.2 Inherent QLoRA Limitations

**1. Quantization is lossy by definition:**
NF4 introduces irreducible quantization error in W₀. Even with optimal LoRA training, the base model's representations are permanently imprecise. This sets a quality ceiling below standard LoRA.

**2. Slower training than LoRA:**
Every forward pass requires dequantization of NF4 blocks to BF16. This adds ~30% wall-clock overhead compared to LoRA on a BF16 base. Cannot be eliminated — it's the cost of 4-bit storage.

**3. Hardware dependency:**
`bitsandbytes` requires NVIDIA GPUs with CUDA. Not supported on:
- Apple Silicon (MPS)
- AMD GPUs (ROCm) — limited support
- CPU-only environments
- Google TPUs

**4. Base model cannot be updated:**
Because W₀ is quantized and frozen, any knowledge gaps in the pre-trained model cannot be corrected. QLoRA can only redirect, not rebuild. This limitation is stronger than standard LoRA because even the existing representations are degraded by quantization.

**5. Inference requires re-loading at full precision:**
If you want to merge and serve the final model at BF16 quality, you must re-load the full BF16 base model, merge, then serve. You cannot serve from the 4-bit quantized + adapter combination at full speed without the dequantization overhead.

### 13.3 Variant-Specific Limitations

**AdaLoRA:**
- Requires SVD decomposition tracking throughout training — computationally expensive
- The importance scoring heuristic is not guaranteed optimal
- Rank budget must be set in advance; wrong budget → poor allocation

**LoRA+:**
- λ (asymmetric LR ratio) is a new hyperparameter requiring tuning
- Not beneficial for very simple tasks where default LoRA already converges well
- Gains diminish with larger datasets (Adam already compensates for the asymmetry)

**DoRA:**
- Adds norm computation overhead per forward pass
- Decomposing pre-trained weights adds initialization complexity
- Not yet widely supported in fine-tuning frameworks
- Marginal gains may not justify complexity for most practical tasks

---

## 14. Summary & Key Takeaways

### 14.1 Decision Tree: Choosing Your Method

```
START: Fine-tuning an LLM
         │
         ▼
Do you have GPU with > 40 GB VRAM?
    ├── YES → Use LoRA (BF16 base)
    │           ├── Simple task, small data → r=4-8, Q+V only
    │           └── Complex task, large data → r=32-64, all linear
    │
    └── NO → Use QLoRA (NF4 base)
              ├── 24 GB GPU → up to 13B models
              └── 48 GB GPU → up to 70B models

After choosing LoRA/QLoRA:
         │
         ▼
Is parameter budget the constraint?
    └── YES → AdaLoRA (adaptive rank)

Is training speed the constraint?
    └── YES → LoRA+ (asymmetric LR)

Is maximum quality the goal?
    └── YES → DoRA (magnitude-direction decomposition)

Need to serve multiple fine-tunes on one GPU?
    └── YES → Unmerged LoRA + S-LoRA serving
```

### 14.2 Top 10 Interview Questions & One-Line Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | What problem does LoRA solve? | Reduces fine-tuning memory from O(d²) to O(dr) by decomposing weight updates into low-rank matrices. |
| 2 | Why is B initialized to zero? | So that ΔW = B·A = 0 at t=0, ensuring fine-tuning starts from the pre-trained model. |
| 3 | What is the role of α (alpha)? | Scales the adapter output; effective scale = α/r; higher α = stronger adapter influence. |
| 4 | Why does LoRA target attention matrices? | Attention captures task-specific relational structure; FFN layers store factual knowledge better left untouched. |
| 5 | What are QLoRA's three innovations? | NF4 4-bit quantization, Double Quantization of constants, and Paged Optimizers for memory spike handling. |
| 6 | Why NF4 over INT4? | NF4 spaces quantization levels to match the normal distribution of weights → lower quantization error. |
| 7 | What is Double Quantization? | Quantizing the per-block quantization constants themselves, saving ~0.37 bits per parameter. |
| 8 | How does LoRA differ from Adapter Layers? | LoRA adapters can be merged into base weights (zero inference overhead); Adapter Layers always add inference latency. |
| 9 | When does LoRA fail? | When the weight update has high intrinsic rank (flat singular value spectrum), LoRA underfits regardless of training. |
| 10 | What is AdaLoRA's key innovation over LoRA? | It allocates rank budget dynamically per weight matrix based on importance, rather than uniform rank across all matrices. |

### 14.3 Quick Reference Cheat Sheet

```
LoRA Memory Formula:
  Trainable params = 2 × r × d × num_adapted_layers
  (where d = hidden_dim, r = rank)

QLoRA Memory Formula:
  GPU memory ≈ (num_params × 4.13 bits) / 8  [bytes]
            + (LoRA params × 2 bytes)          [BF16 adapters]
            + activations

LoRA Hyperparameter Defaults (safe starting point):
  r = 16, alpha = 32, dropout = 0.05
  target_modules = ["q_proj", "v_proj"]
  optimizer = paged_adamw_32bit (QLoRA) or adamw (LoRA)
  lr = 2e-4, scheduler = cosine, warmup = 5%

Quality Ranking (typically):
  Full FT > DoRA > LoRA > QLoRA > Prefix Tuning > Prompt Tuning
  (task and data dependent — benchmark on your task)
```
