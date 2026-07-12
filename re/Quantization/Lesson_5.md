# Quantization Lesson 5 — GPTQ: GPU-Accelerated Post-Training Quantization

---

## The Problem GPTQ Solves

INT8 with LLM.int8() gives you 2× memory reduction. But if you want to run a 70B model on a 2× A100 (80GB each) setup, you need 4× compression or better — you need INT4.

Naive INT4 quantization (per-channel absmax) is not practical for LLMs. With only 16 distinct values across a weight distribution that spans several standard deviations, the rounding errors are so large that model quality degrades severely.

GPTQ (Frantar et al., 2022) is the breakthrough that made INT4 quantization viable for large language models. It achieves INT4 quantization with quality close to FP16 by using a smart error compensation strategy.

---

## The Quantization Scheme: Uniform, Asymmetric, Storage vs. Compute

Before getting into *how* GPTQ compensates for error (the rest of this lesson), it's worth being precise about the quantization scheme it compensates *around* — this is a different design axis from the Hessian-based error correction, and it's easy to conflate the two.

### Uniform, not quantile-based

GPTQ places its quantization levels **evenly spaced** across a group's `[min, max]` range — the paper describes this as *"standard uniform per-row asymmetric quantization on the min-max grid."* This is a different philosophy from NF4 (used in QLoRA, Lesson 3.5), which places its 16 levels at the **quantiles of a Gaussian** — dense near zero, sparse at the tails.

| | NF4 (QLoRA) | GPTQ |
|---|---|---|
| Level placement | Quantile-based (dense near 0) | Uniform (evenly spaced, min→max) |
| Error minimized via | Distribution-matched levels | Hessian-guided compensation (this lesson) |
| Design philosophy | Make the grid match the data's shape | Keep a simple grid, actively correct for its error |

### Asymmetric quantization — why it needs a zero-point

GPTQ's grid is **asymmetric**: it spans the group's actual `min` to `max`, which usually isn't centered at zero — unlike LLM.int8()'s **symmetric** scheme (`[-127, 127]`, one scale, zero always maps to code 0). Because the range isn't centered, "zero" doesn't naturally fall on a round integer code anymore, so an explicit **zero-point** is needed to mark which code represents the value 0 within that shifted grid.

For an n-bit grid (n=4 → codes 0–15), computed per group:
```
scale      = (max − min) / (2ⁿ − 1)
zero_point = round(−min / scale)                    # clamped to [0, 2ⁿ−1]

quantize:    code = clamp(round(w / scale) + zero_point, 0, 2ⁿ − 1)
dequantize:  ŵ    = scale × (code − zero_point)
```
Compare this to symmetric quantization (LLM.int8()), which is just `code = round(w / scale)` — no zero_point term, one fewer number stored per group, but it wastes half the grid if the data isn't actually centered at zero. Asymmetric quantization spends that one extra stored number to use the *entire* grid efficiently regardless of where the data sits.

### Storage vs. compute datatype — same pattern as NF4

Just like NF4, there is no native 4-bit matmul hardware, so GPTQ splits storage and compute:
- **Storage**: 4-bit integer codes (0–15) plus a `scale` and `zero_point` per group.
- **Compute**: right before the matmul, codes are dequantized back to FP16/BF16 using the formula above, and the actual matrix multiplication runs in FP16/BF16.

One practical difference from a naive NF4-style dequantize-then-matmul: production GPTQ inference kernels (ExLlama, Triton, AutoGPTQ's CUDA kernels) **fuse** the dequantization directly into the matmul kernel, so weight tiles are dequantized just as they're consumed rather than as a separate upfront pass. This is a performance detail, not a change to the underlying storage/compute split.

### Does GPTQ need a lookup table like NF4? — No, and here's why

This is the key structural difference from NF4. Because GPTQ's levels are **uniformly spaced**, `code → value` is a simple arithmetic formula (`scale × (code − zero_point)`) — one multiply and one subtract, no memory lookup required. NF4 needed an actual table specifically *because* its levels are irregular (Gaussian quantiles like `-0.6961, -0.5250, ...`) — no formula connects "code 6" to its value, so the only option is to store and index into those numbers directly.

In practice, some GPTQ inference kernels *do* materialize a small per-group table of the 16 possible dequantized values as a speed optimization — for a fixed `scale` and `zero_point` there are only 16 possible outputs, and precomputing them once per group can be cheaper than a multiply per weight on some hardware. But this is a fundamentally different kind of table from NF4's:

| | NF4's table | GPTQ's (optional) per-group table |
|---|---|---|
| Where values come from | Gaussian quantiles — fixed for the *entire model*, computed once, ever | `scale × (i − zero_point)` for i = 0..15 — a formula, re-derived for *every group* |
| Necessity | Required — no formula exists | Optional — a formula already exists; the table just caches it |
| How you'd build it | Compute the inverse CDF of N(0,1) at 16 quantile points, once | Loop i = 0..15, compute `scale*(i - zero_point)` per group |

**Building that optional per-group table by hand**, for a tiny group of weights `[0.2, -0.5, 1.1, 0.3]`, 4-bit (codes 0–15):
```
min = -0.5, max = 1.1
scale = (1.1 - (-0.5)) / 15 = 0.1067
zero_point = round(0.5 / 0.1067) = round(4.69) = 5

Per-group table (code i → dequantized value):
i=0:  0.1067×(0-5)  = -0.533
i=1:  0.1067×(1-5)  = -0.427
i=2:  0.1067×(2-5)  = -0.320
i=3:  0.1067×(3-5)  = -0.213
i=4:  0.1067×(4-5)  = -0.107
i=5:  0.1067×(5-5)  =  0.000    ← where "true zero" lands
i=6:  0.1067×(6-5)  =  0.107
...
i=15: 0.1067×(15-5) =  1.067
```
Each original weight is then matched to its nearest table entry — e.g. `0.2` lands near `i=7` (0.213), the closest available code. The table just saves recomputing this formula thousands of times per group at inference; it's a caching convenience, not a distinct quantization *scheme* the way NF4's table is.

> **Note on the simplified code below:** the `quantize_to_nbit` helper in the "Implementation (Simplified)" section uses a symmetric-style formula (`scale = x.abs().max() / max_int`, no zero_point) for brevity. Real GPTQ implementations use the asymmetric, zero-point formula shown above — which is also why `gptq_quantize_layer` below already returns a separate `zero_points` dict alongside `scales`.

---

Before getting into GPTQ's formulas, it's worth understanding *why* this whole family of methods reaches for the Hessian (second derivative), rather than something simpler.

**The naive baseline — Round-to-Nearest (RTN)** quantizes each weight independently: round `w_ij` to the nearest allowed level and move on. No context, no compensation. This works fine at 8-bit, but at 3–4 bit, the grid is so sparse that rounding error becomes large — and RTN implicitly treats every weight as equally important, which is a bad assumption. Some weights, if nudged slightly, barely change the model's output; others cause a big change. That "how much does perturbing this weight matter" property is exactly what **curvature** measures — and curvature is what the Hessian captures.

**Why gradient (first derivative) is useless here, but curvature (second derivative) isn't.** A pretrained model sits at (or very near) a local minimum of its loss — that's what "trained to convergence" means. At a minimum, the gradient is ≈0 in every direction. Quantizing a weight is a small forced perturbation `δw`, and a second-order Taylor expansion of how the loss responds to that perturbation looks like:

```
ΔLoss ≈ (gradient term, ≈0 near a minimum) + ½ · δwᵀ · H · δw + (higher order, ignored)
```

The gradient term vanishes, so the **Hessian term is the dominant, first non-trivial predictor of how much damage a given weight perturbation will do.** That's the whole justification for reaching for second-order information instead of just rounding independently: the Hessian also captures how weights interact with each other (via its off-diagonal entries), which is what makes it possible to *compensate* — adjust other weights to cancel out the error from the one just quantized.

**Where this idea comes from.** This isn't new to GPTQ — it's inherited from classic neural network pruning literature:
- **Optimal Brain Damage** (LeCun et al., 1990) — used curvature to decide which weights are safest to delete (set to zero).
- **Optimal Brain Surgeon (OBS)** (Hassibi & Wolff, 1993) — extended this: after deleting a weight, use the Hessian to *compensate* by optimally adjusting the remaining weights, rather than leaving them untouched.
- **Optimal Brain Quantization (OBQ)** (Frantar & Alistarh, 2022) — took the same delete-and-compensate machinery and swapped the constraint from "this weight must become exactly 0" to "this weight must land on the nearest quantization grid value." Same math, different target.

GPTQ is what happens when OBQ is restructured to run fast enough for 175B-parameter models — the "two approximations" (really three distinct tricks) described later in this lesson.

---

## The Key Insight: Layer-Wise Error Compensation

Naive quantization quantizes each weight independently: you round `w_ij` to the nearest INT4 value and move on. Each weight incurs a rounding error, and all those errors accumulate.

GPTQ's insight: **when you quantize one weight, you can adjust the remaining unquantized weights to compensate for the error you just introduced.**

If quantizing `w_1` introduces error `δw_1`, you update the remaining weights `w_2, w_3, ...` such that the change in the layer's output is minimized. The layer output stays close to the FP16 output even though individual weights are being rounded aggressively.

This is the **Optimal Brain Quantization (OBQ)** framework that GPTQ extends.

---

## The Mathematical Foundation

### Layer Output Error

For a linear layer with weight matrix **W** and input **X**, the output is **WX**. After quantizing a weight, the error in the layer output is:

```
ΔOutput = ΔW × X

Where ΔW is the quantization error in the weight matrix
```

We want to minimize this output error. Specifically, for each row of W (one output neuron), we want to find the quantization that minimizes:

```
E = ||ΔW_row × X||²
```

Notice this objective is a **local, per-layer proxy** for the network's true loss — not the true loss itself. It's used because, unlike the true loss, it's a sum of squares, which makes the next step exact rather than approximate.

### The Role of the Hessian

The Hessian of the quantization error with respect to the weights is:

```
H = 2 × X × X^T
```

**Where this formula actually comes from.** Let `δ = w₀ − w` be the difference between the original weight row and the (possibly quantized) row being evaluated. Then:

```
E(δ) = ||δ·X||² = (δ·X)·(δ·X)ᵀ = δ · (X·Xᵀ) · δᵀ
```

This is a plain quadratic form in `δ`. Differentiating once gives the gradient, `∂E/∂δ = 2·X·Xᵀ·δ`; differentiating a second time gives the Hessian, `∂²E/∂δ² = 2·X·Xᵀ`. Because `E` is exactly a sum of squares (not an approximation of some other function), this derivative is **exact** — there's no truncated higher-order term being ignored here, unlike the Taylor expansion of the true network loss above. The only approximation in the whole method is the *choice* to use this local reconstruction error as a stand-in for the true global loss — not any error in this specific calculus.

This is computable! You run a calibration dataset through the model and collect the input activations X for each layer. Then compute H = 2XX^T. Because `H` depends only on the input activations `X` — not on which output row of `W` is being quantized — it only needs to be computed **once per layer** and can be reused across every output neuron (every row) in that layer's weight matrix.

The Hessian tells you: **how sensitive is the layer's output to changes in each weight?** Large Hessian diagonal entry for `w_ij` → that weight is sensitive and needs to be quantized carefully. Small entry → that weight can absorb quantization error without much impact.

### Does the Inverse Always Exist? (Dampening)

The update formulas below require **inverting** `H` — worth checking whether that's actually guaranteed, since it's easy to assume it always works.

`H = X·Xᵀ` is a **Gram matrix**, and any matrix of that form is guaranteed to be **positive semi-definite** (all eigenvalues ≥ 0). That's *not* the same as invertible: if some input feature dimensions are duplicated, highly correlated, or always zero for the calibration data used (a "dead" feature), `H` can be **singular** — some eigenvalues exactly 0 — and a singular matrix has no inverse.

This is a real risk in practice, not a hypothetical one, and it's exactly what the `damp_percent` and `dead_weights` handling in the code below are for. Before inverting, a small multiple of the identity is added:
```
H ← H + λI
```
where `λ` is typically ~1% of `H`'s average diagonal magnitude. This nudges every eigenvalue strictly above 0, making `H` **positive definite**, which does guarantee an inverse exists. It's a deliberate, cheap fix (a form of Tikhonov/ridge regularization) rather than something left to chance — this is precisely the `H.diagonal().add_(damp_percent * H.diagonal().mean())` line in the implementation below, and the `dead_weights` check handles the extreme case of a feature dimension that's exactly zero across all calibration data (which would otherwise leave a zero row/column in `H` no amount of small dampening could safely fix without also zeroing the corresponding weights).

### The OBQ Update Formula

When you quantize weight `w_q` (round it to INT4), the induced error in that weight is:

```
error_q = w_q - quantize(w_q)
```

The update to compensate for this error in the remaining weights is:

```
w_update = -(error_q / H[q,q]) × H[q, :]

Applied only to the remaining unquantized weights.
```

This is the **Optimal Brain Surgeon (OBS)** update applied to each weight as it is quantized — the same delete-and-compensate mechanism from the Background section, just retargeted at "round to a grid value" instead of "delete."

---

## GPTQ Algorithm: Step by Step

OBQ, as just described, has to run its greedy loop — pick the cheapest weight to quantize next, quantize it, update the Hessian inverse, repeat — **separately for every row** of every weight matrix, re-deriving the inverse-Hessian trajectory each time. That gives roughly `O(d_row · d_col³)` complexity, which explodes for a 175B-parameter model. GPTQ keeps OBQ's exact mathematical idea (Hessian-guided rounding + compensation) but adds **three** distinct engineering tricks that make it fast. None of them change the underlying math — they change how cheaply it's computed.

**Trick 1 — Fixed, arbitrary order, shared across all rows.** Instead of greedily picking "whichever weight is cheapest to fix right now" (which can differ row to row), GPTQ found that for large layers, a simple fixed order — e.g., left to right, column by column — performs about as well as the greedy order; with thousands of weights per row, the gains from picking a perfect per-row order are small and mostly average out. The payoff is computational: if every row uses the *same* column order, the sequence of Hessian-inverse updates is identical across rows, so it's computed **once for the whole layer** instead of once per row. This alone drops the complexity to roughly `O(max(d_row · d_col², d_col³))`.

**Trick 2 — Lazy batched updates.** Even with a shared order, naively updating every remaining weight immediately after each single weight is quantized means doing many small, memory-bandwidth-bound operations — expensive on a GPU, which is much better at fewer, larger operations. GPTQ processes columns in **blocks** (e.g., 128 at a time): updates *within* the current block are applied normally, but updates to everything *outside* the block are deferred and applied all at once, as a single large batched operation, once the block finishes. Same math, applied in bigger and less frequent chunks.

**Trick 3 — Cholesky reformulation.** Manually updating `H⁻¹` step by step (as OBQ does) is both expensive and numerically fragile — floating-point error accumulates over thousands of sequential updates and can eventually break the positive-definiteness guarantee from the dampening step above. GPTQ instead computes a single, one-time **Cholesky decomposition** of `H⁻¹` up front, using standard, numerically stable linear algebra routines. Everything the quantization loop needs at each step is then just a cheap read from that precomputed factor, rather than a live matrix update — faster, and numerically robust.

| | OBQ | GPTQ |
|---|---|---|
| Quantization order | Greedy, different per row | Fixed, arbitrary, shared across all rows |
| `H⁻¹` computation | Re-derived per row, per step | Cholesky-decomposed once per layer |
| Weight updates | Applied immediately, one at a time | Deferred and applied in large batches |
| Complexity | `O(d_row · d_col³)` | `O(max(d_row · d_col², d_col³))` |
| Numerical stability | Can degrade over many sequential updates | Stable (standard Cholesky routine) |
| Practical result (175B model) | ~weeks | ~4 GPU-hours |

### High-Level Algorithm

```
For each linear layer in the model:
    1. Collect input activations from calibration data → X (shape: [hidden_dim × n_samples])
    2. Compute Hessian: H = 2XX^T / n_samples
    3. Apply Cholesky decomposition for numerical stability
    4. For each column j of W (block by block, e.g., 128 columns at a time):
        a. Quantize all weights W[:, j] to INT4
        b. Compute quantization error: error = W[:, j] - Quant(W[:, j])
        c. Update all remaining columns W[:, j+1:] to compensate:
           W[:, j+1:] -= error × (H[j, j+1:] / H[j, j])
    5. Store the INT4 quantized weights
```

### Implementation (Simplified)

```python
import torch
import numpy as np

def gptq_quantize_layer(
    weight: torch.Tensor,     # Shape: [out_features, in_features], FP16
    H: torch.Tensor,          # Hessian, Shape: [in_features, in_features]
    n_bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01
) -> dict:
    """
    Simplified GPTQ quantization for one layer.
    
    Args:
        weight: FP16 weight matrix
        H: Hessian matrix computed from calibration data
        n_bits: Target bit width (4 for INT4)
        group_size: Number of weights per quantization group
        damp_percent: Damping for numerical stability
    """
    
    W = weight.clone().float()   # Work in FP32 for numerical stability
    out_features, in_features = W.shape
    
    # Damping: add small value to Hessian diagonal for stability
    dead_weights = H.diag() == 0
    H[dead_weights, dead_weights] = 1
    W[:, dead_weights] = 0
    
    H.diagonal().add_(damp_percent * H.diagonal().mean())
    
    # Cholesky decomposition (numerically stable Hessian inversion)
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    H_inv = torch.linalg.cholesky(H_inv, upper=True)
    
    # Quantized weight storage
    Q = torch.zeros_like(W)
    
    # Process in blocks of group_size columns
    Losses = torch.zeros(out_features, in_features // group_size)
    
    for col_start in range(0, in_features, group_size):
        col_end = min(col_start + group_size, in_features)
        
        W_block = W[:, col_start:col_end]
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]
        
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        
        for j in range(col_end - col_start):
            w = W_block[:, j]               # Current column to quantize
            d = H_inv_block[j, j]           # Hessian diagonal element
            
            # Quantize this column to INT4
            q = quantize_to_nbit(w, n_bits)  # Round to nearest INT4
            Q_block[:, j] = q
            
            # Compute error
            err = (w - q) / d
            Err_block[:, j] = err
            
            # Compensate remaining columns in this block
            W_block[:, j+1:] -= err.unsqueeze(1) @ H_inv_block[j:j+1, j+1:]
        
        Q[:, col_start:col_end] = Q_block
        
        # Propagate error to subsequent blocks
        W[:, col_end:] -= Err_block @ H_inv[col_start:col_end, col_end:]
    
    return {
        "quantized_weight": Q.to(torch.int8),  # Actually INT4 packed into INT8
        "scales": compute_group_scales(weight, group_size, n_bits),
        "zero_points": compute_group_zeros(weight, group_size, n_bits)
    }


def quantize_to_nbit(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Quantize tensor to n-bit integers."""
    max_int = 2**(n_bits - 1) - 1
    scale = x.abs().max() / max_int
    return (x / scale).round().clamp(-max_int, max_int) * scale
```

### The Calibration Dataset

GPTQ needs real input data to compute the Hessian. A calibration dataset of 128-512 samples is typically sufficient:

```python
# Standard practice: use 128 samples from the training distribution
# For general-purpose models: C4 or wikitext-2 are common choices
# For domain-specific models: use samples from your domain

calibration_texts = [
    "The quick brown fox jumps over the lazy dog...",
    # 127 more samples
]

# Tokenize
inputs = tokenizer(
    calibration_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048
)

# Run forward pass to collect Hessian for each layer
# (GPTQ library handles this automatically)
```

Note that collecting `X` for the Hessian only requires **forward passes** through the calibration data — no loss function, no backward pass, no gradient descent. This is what makes GPTQ a "backpropagation-free" method: the only role the calibration data plays is producing realistic activation statistics to compute `H = X·Xᵀ` from, not training anything via gradient updates.

---

## Using GPTQ in Practice

The `auto-gptq` library handles all the complexity:

```python
# Quantizing a model with GPTQ
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name = "meta-llama/Llama-2-7b-hf"

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,                  # INT4
    group_size=128,          # Group size for per-group scales
    desc_act=True,           # Act-order: quantize in order of activation magnitude
                             # Better quality but slightly slower
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare calibration dataset
calibration_dataset = [
    tokenizer(text, return_tensors="pt")["input_ids"]
    for text in calibration_texts[:128]
]

# Run GPTQ quantization (takes 10 minutes to 2 hours depending on model size)
model.quantize(calibration_dataset)

# Save quantized model
model.save_quantized("./llama-2-7b-gptq-int4")

# Later: load quantized model (fast — no re-quantization needed)
model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-int4",
    device="cuda:0"
)
```

### Loading Pre-Quantized GPTQ Models from HuggingFace Hub

Many popular models already have GPTQ-quantized versions on the Hub (TheBloke uploads many):

```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# Pre-quantized model — no quantization step needed
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
    use_triton=True  # Faster inference via Triton kernels
)
```

---

## GPTQ Hyperparameters and Their Effects

**`bits` (2, 3, 4, 8):**
- INT4 (bits=4): Best quality-memory tradeoff. Standard choice.
- INT3 (bits=3): ~25% less memory than INT4, noticeable quality drop.
- INT2 (bits=2): Extreme compression, severe quality degradation on most models.

**`group_size` (32, 64, 128, -1):**
- Smaller group → more scale factors → better quality → more memory overhead
- 128: Standard choice. Excellent quality-efficiency balance.
- 64: Slightly better quality, slightly more memory.
- -1: Per-column quantization (one scale per output neuron). Maximum memory efficiency, lower quality.
- **Rule:** For INT4, use group_size=128. For INT3, use group_size=64 or 32.

**`desc_act` (True/False) — Activation Order:**
- True: Quantize weights in order of their activation magnitude (most important weights quantized first). Better quality, especially for INT4.
- False: Quantize in natural order. Faster GPTQ execution, slightly lower quality.

```python
# Quality comparison (approximate perplexity on wikitext-2 for LLaMA-7B)
# FP16 baseline:           5.68
# GPTQ INT4, g128:         5.85  (+3% degradation)
# GPTQ INT4, g128, desc_act: 5.78  (+2% degradation)
# GPTQ INT3, g128:         6.20  (+9% degradation)
# GPTQ INT2, g128:         8.90  (+57% degradation)
```

---

## Memory Savings: What GPTQ Achieves

For a 7B parameter model:

```
FP16:       7B × 2 bytes = 14.0 GB
INT8:       7B × 1 byte  =  7.0 GB   (2×  reduction)
GPTQ INT4:  7B × 0.5B   =  3.5 GB   (4×  reduction)
            + scales overhead ≈ +0.5 GB (group_size=128)
            Total ≈ 4.0 GB            (3.5× reduction net)
```

For a 70B parameter model:
```
FP16:       140 GB  (requires 2× A100 80GB)
GPTQ INT4:   35 GB + ~5 GB scales ≈ 40 GB  (fits on single A100 80GB!)
```

GPTQ makes it possible to run 70B models on a single 80GB GPU — a transformational improvement for deployment cost.

---

## Limitations of GPTQ

**Quantization time:** GPTQ takes time to run — 10-30 minutes for 7B models, 2-4 hours for 70B models. Not fast enough for on-the-fly quantization.

**Calibration data dependency:** Quality depends on the calibration dataset being representative. Quantizing a coding model with general text calibration data may not be optimal.

**Inference speed:** GPTQ INT4 is not always faster than FP16 on GPU. The matrix multiplications happen with dequantization overhead. Actual speedup depends heavily on batch size and hardware. At batch size 1 (typical for generation), it may be slightly slower than FP16 but use far less memory.

**Not suitable for fine-tuning:** GPTQ-quantized models cannot be directly fine-tuned (the quantization grid is fixed). For fine-tuning quantized models, you need NF4 + QLoRA (Lesson 7).

---

## GPTQ vs. LLM.int8()

| Aspect | LLM.int8() | GPTQ INT4 |
|---|---|---|
| Memory reduction | ~2× | ~3.5-4× |
| Quality | Near-lossless | Very good (small degradation) |
| Quantization time | Seconds (no calibration) | Minutes to hours |
| Inference speed | Similar to FP16 | Can be slower at batch=1 |
| Fine-tunable | No (practically) | No |
| Calibration data | Not needed | 128 samples needed |
| Bits | 8 | 4 |
| Best use case | When 2× reduction is enough | Running 70B on single GPU |

One more distinction worth having ready for an interview: LLM.int8() avoids error almost entirely by isolating outliers into an exact FP16 path — there's very little error left to manage. GPTQ instead accepts that error at 4-bit is unavoidable for *every* weight, and manages it explicitly through Hessian-guided compensation. That's a genuinely different strategy, not just "the same idea at a lower bit-width."

---

## Summary

- GPTQ uses **uniform, asymmetric (min-max, zero-point) quantization** — evenly spaced levels across each group's actual range — unlike NF4's quantile-based, distribution-matched levels. This is a separate design choice from the Hessian-based error compensation and is what the rest of this lesson corrects *around*.
- Because GPTQ's levels are uniform, `code → value` is a simple formula (`scale × (code − zero_point)`), so GPTQ does **not** need a lookup table the way NF4 does. Any per-group table seen in GPTQ kernels is an optional cache of that formula's outputs, not a required, model-wide artifact like NF4's Gaussian-quantile table.
- Like NF4, GPTQ separates storage (4-bit codes + per-group scale/zero-point) from compute (dequantized to FP16/BF16 right before the matmul) — production kernels fuse this dequantization directly into the matmul for speed.
- GPTQ achieves practical INT4 quantization by compensating for quantization errors as they accumulate, using second-order information (the Hessian) to update remaining weights when each weight is quantized.
- This traces back to classic pruning literature (Optimal Brain Damage → Optimal Brain Surgeon → Optimal Brain Quantization): curvature (the Hessian) is used instead of the gradient because, at a trained minimum, the gradient is ≈0 and tells you nothing — curvature is the first term that actually predicts the damage from perturbing a weight.
- `H = X·Xᵀ` (up to a constant factor) falls directly out of taking the second derivative of the layer's squared reconstruction error `||δ·X||²` with respect to the weight perturbation `δ` — this derivation is exact, since the reconstruction-error objective is a true quadratic form, unlike the Taylor-approximated network loss it stands in for.
- `H` is a Gram matrix, so it's guaranteed positive semi-definite but *not* automatically invertible — dampening (`H ← H + λI`) is added specifically to guarantee an inverse exists before the algorithm proceeds.
- The key formula: when quantizing weight `w_q`, update remaining weights by `-(error / H[q,q]) × H[q, :]` to minimize the change in layer output.
- GPTQ speeds up OBQ with three distinct tricks: a fixed quantization order shared across all rows (lets the Hessian-inverse trajectory be computed once per layer instead of once per row), lazy batched updates (fewer, larger GPU operations instead of many small ones), and a one-time Cholesky decomposition (faster and numerically stable, replacing repeated manual matrix-inverse updates).
- Calibration data (128 samples) is needed to compute the Hessian — but only via forward passes. No gradients, no loss function, no backpropagation are involved anywhere in GPTQ.
- Standard configuration: bits=4, group_size=128, desc_act=True. Gives approximately 3.5× memory reduction with < 3% quality degradation on perplexity benchmarks.
- GPTQ is the standard for serving large models at INT4. TheBloke and other contributors have pre-quantized most popular models on HuggingFace Hub.
- Limitation: GPTQ takes time to run and cannot be used for fine-tuning (fixed quantization grid). For fine-tuning, use QLoRA with NF4 (Lesson 7).

---

## What's Next

Lesson 6 covers AWQ — Activation-Aware Weight Quantization, which takes a different approach to INT4: instead of compensating for error after quantization, it protects the most important weights before quantization by scaling them up, so they are not rounded away.