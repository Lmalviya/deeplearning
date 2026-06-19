# Quantization Lesson 5 — GPTQ: GPU-Accelerated Post-Training Quantization

---

## The Problem GPTQ Solves

INT8 with LLM.int8() gives you 2× memory reduction. But if you want to run a 70B model on a 2× A100 (80GB each) setup, you need 4× compression or better — you need INT4.

Naive INT4 quantization (per-channel absmax) is not practical for LLMs. With only 16 distinct values across a weight distribution that spans several standard deviations, the rounding errors are so large that model quality degrades severely.

GPTQ (Frantar et al., 2022) is the breakthrough that made INT4 quantization viable for large language models. It achieves INT4 quantization with quality close to FP16 by using a smart error compensation strategy.

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

### The Role of the Hessian

The Hessian of the quantization error with respect to the weights is:

```
H = 2 × X × X^T
```

This is computable! You run a calibration dataset through the model and collect the input activations X for each layer. Then compute H = 2XX^T.

The Hessian tells you: **how sensitive is the layer's output to changes in each weight?** Large Hessian diagonal entry for `w_ij` → that weight is sensitive and needs to be quantized carefully. Small entry → that weight can absorb quantization error without much impact.

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

This is the **Optimal Brain Surgeon (OBS)** update applied to each weight as it is quantized.

---

## GPTQ Algorithm: Step by Step

GPTQ makes OBQ practical for large models by making two approximations that reduce complexity while preserving quality:

1. **Quantize in column order** (all rows simultaneously), not by selecting the optimal weight to quantize next.
2. **Use a blocked Cholesky decomposition** of the Hessian to compute updates efficiently.

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

---

## Summary

- GPTQ achieves practical INT4 quantization by compensating for quantization errors as they accumulate, using second-order information (the Hessian) to update remaining weights when each weight is quantized.
- The key formula: when quantizing weight `w_q`, update remaining weights by `-(error / H[q,q]) × H[q, :]` to minimize the change in layer output.
- Calibration data (128 samples) is needed to compute the Hessian. Any representative text works for general-purpose models.
- Standard configuration: bits=4, group_size=128, desc_act=True. Gives approximately 3.5× memory reduction with < 3% quality degradation on perplexity benchmarks.
- GPTQ is the standard for serving large models at INT4. TheBloke and other contributors have pre-quantized most popular models on HuggingFace Hub.
- Limitation: GPTQ takes time to run and cannot be used for fine-tuning (fixed quantization grid). For fine-tuning, use QLoRA with NF4 (Lesson 7).

---

## What's Next

Lesson 6 covers AWQ — Activation-Aware Weight Quantization, which takes a different approach to INT4: instead of compensating for error after quantization, it protects the most important weights before quantization by scaling them up, so they are not rounded away.