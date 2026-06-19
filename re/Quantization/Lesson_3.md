# Quantization Lesson 3 — Absmax and Zero-Point Quantization

---

## The Core Problem: Mapping Floats to Integers

You have a weight matrix with values like:
```
[-1.82, 0.43, -0.07, 2.15, -0.93, 0.11, 1.67, -2.38]
```

You want to represent these as INT8 (values from -128 to 127). The question is: **what is the best mapping?**

This is not trivial. You need to:
1. Preserve the relative differences between values as much as possible.
2. Use the full range of INT8 (don't waste representable values).
3. Be able to approximately recover the original floats later (dequantization).

The two fundamental approaches are **absmax quantization** (symmetric) and **zero-point quantization** (asymmetric).

---

## Absmax Quantization (Symmetric)

### The Idea

Find the maximum absolute value in the tensor. Use it to scale all values to fill the INT8 range symmetrically around zero.

### The Math

Given a tensor **X** (floating point) and target integer range [-127, 127]:

```
scale = max(|X|) / 127

X_quantized = round(X / scale)

X_dequantized = X_quantized × scale
```

### Worked Example

```python
import numpy as np

weights = np.array([-1.82, 0.43, -0.07, 2.15, -0.93, 0.11, 1.67, -2.38])

# Step 1: Find scale
max_abs = np.max(np.abs(weights))   # = 2.38
scale = max_abs / 127               # = 2.38 / 127 = 0.01874

# Step 2: Quantize
quantized = np.round(weights / scale).astype(np.int8)
print(quantized)
# [-97, 23, -4, 115, -50, 6, 89, -127]

# Step 3: Dequantize (approximately recover originals)
dequantized = quantized.astype(np.float32) * scale
print(dequantized)
# [-1.818, 0.431, -0.075, 2.155, -0.937, 0.112, 1.668, -2.380]

# Quantization error
error = weights - dequantized
print(np.max(np.abs(error)))
# Max error: ~0.022 (less than 1% of the max value)
```

### Visual Intuition

```
Float range:     [-2.38 ........ 0 ........ 2.38]
INT8 range:      [-127  ........ 0 ........ 127]

Each INT8 step represents:  2.38 / 127 = 0.01874 in float space
```

The mapping is symmetric: zero maps to zero exactly. Positive values map to positive integers, negative to negative.

### The Scale Factor Is Everything

Notice that the scale factor (`0.01874` above) is the key artifact. You must store this alongside the quantized weights to recover the original values. At INT8, storing one FP32 scale per tensor adds negligible memory overhead.

```python
# What you store in memory
{
    "quantized_weights": np.array([-97, 23, -4, 115, -50, 6, 89, -127], dtype=np.int8),
    "scale": np.float32(0.01874)
}
# Total: 8 bytes (int8) + 4 bytes (scale) = 12 bytes
# vs original: 8 × 4 = 32 bytes (float32)
# Compression ratio: 2.67× (not 4× because of the scale factor overhead — but per-tensor it's negligible)
```

### The Outlier Problem

Absmax has a critical weakness: **outliers**.

```python
weights_with_outlier = np.array([-0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.01, 100.0])
#                                 ^ all values are tiny                           ^ huge outlier

max_abs = np.max(np.abs(weights_with_outlier))  # = 100.0
scale = 100.0 / 127  # = 0.787

quantized = np.round(weights_with_outlier / scale).astype(np.int8)
# [-0, 0, -0, 0, -0, 0, -0, 127]
# The small values all quantize to 0 or ±1 — complete precision loss!
```

One outlier forces the scale to be large, and all the small values collapse to zero. This is not a theoretical problem — large language models are known to develop **activation outliers** of magnitude 1000× larger than typical values. Handling outliers is the central challenge of quantizing LLMs.

---

## Zero-Point Quantization (Asymmetric)

### The Idea

Absmax assumes the float distribution is symmetric around zero. Many real distributions are **not** symmetric — they might be concentrated in [0.2, 1.8] with nothing below zero.

Zero-point quantization maps the actual min and max of the distribution to the full integer range, allowing asymmetric distributions to use the full precision of INT8.

### The Math

Given a tensor **X** and target INT8 range [-128, 127]:

```
scale = (max(X) - min(X)) / (127 - (-128))
      = (max(X) - min(X)) / 255

zero_point = round(-min(X) / scale) - 128

X_quantized = clamp(round(X / scale) + zero_point, -128, 127)

X_dequantized = (X_quantized - zero_point) × scale
```

### Worked Example

```python
weights = np.array([0.2, 0.5, 0.8, 1.1, 0.3, 0.9, 0.6, 1.4])
# These are all positive — absmax would waste half the INT8 range

# Step 1: Compute scale and zero_point
min_val = weights.min()  # = 0.2
max_val = weights.max()  # = 1.4

scale = (max_val - min_val) / 255  # = 1.2 / 255 = 0.00471
zero_point = round(-min_val / scale) - 128
# zero_point = round(-0.2 / 0.00471) - 128
# zero_point = round(-42.5) - 128 = -43 - 128 = -171 → clamped to -128

# Step 2: Quantize
quantized = np.clip(np.round(weights / scale) + zero_point, -128, 127).astype(np.int8)

# Step 3: Dequantize
dequantized = (quantized.astype(np.float32) - zero_point) * scale
```

### What you store:
```python
{
    "quantized_weights": quantized,   # INT8
    "scale": scale,                   # FP32
    "zero_point": zero_point          # INT8 or INT32
}
```

### Absmax vs. Zero-Point: When to Use Each

| Scenario | Recommended Method | Reason |
|---|---|---|
| Weights (near-zero centered) | Absmax (symmetric) | Weights are typically symmetric around 0 |
| Activations (always positive after ReLU) | Zero-point (asymmetric) | Asymmetric distribution needs full range |
| Activation after GeLU/SiLU | Zero-point | Slightly asymmetric |
| Embedding layers | Zero-point | May not be centered at zero |

For LLM weights specifically, absmax is the most common choice because weight distributions are empirically near-Gaussian and approximately symmetric. Zero-point quantization adds computation overhead (the zero_point subtraction at inference time) without meaningful benefit for symmetric distributions.

---

## Per-Tensor vs. Per-Channel vs. Per-Token Quantization

The granularity of the quantization — how many values share one scale factor — dramatically affects quality.

### Per-Tensor Quantization

One scale factor for the entire weight matrix. Least memory overhead. Most affected by outliers.

```
Weight matrix W (rows × cols):
┌─────────────────────────────────┐
│ -0.3  0.8  -1.2  2.1  -0.1 ... │
│  0.5 -0.2   0.9  0.4  -0.7 ... │
│ ...                             │
└─────────────────────────────────┘
        │
        ▼ One scale for ALL values
   scale = max(|W|) / 127
```

**Problem:** If one row or column has unusually large values, the scale is dominated by that outlier and all other values lose precision.

### Per-Channel (Per-Row or Per-Column) Quantization

One scale factor per row (or per column). Each row is quantized independently.

```python
def per_channel_quantize(weight_matrix: np.ndarray) -> dict:
    """
    Quantize each row of a weight matrix independently.
    Each row gets its own scale factor.
    """
    rows, cols = weight_matrix.shape
    quantized = np.zeros_like(weight_matrix, dtype=np.int8)
    scales = np.zeros(rows, dtype=np.float32)
    
    for i in range(rows):
        row = weight_matrix[i]
        scale = np.max(np.abs(row)) / 127
        scales[i] = scale
        quantized[i] = np.round(row / scale).clip(-127, 127).astype(np.int8)
    
    return {"quantized": quantized, "scales": scales}


# Memory overhead of per-channel scales:
# For a 4096 × 4096 weight matrix:
# - Per-tensor: 1 scale (4 bytes) vs matrix data
# - Per-channel: 4096 scales (16,384 bytes = 16KB) vs matrix data (16MB INT8)
# Per-channel overhead: 16KB / 16MB = 0.1% — negligible!
```

Per-channel quantization is dramatically better than per-tensor for quality and has negligible memory overhead. It is the standard for modern quantization methods.

### Per-Token Quantization

For activation quantization (not weights), per-token means one scale per token in the sequence. Each row in the activation matrix (where each row is one token's activation vector) gets its own scale.

```
Activation matrix (seq_len × hidden_dim):
Row 0 (token 0): [0.2, -1.1, 0.5, ...] → scale_0 = 1.1/127
Row 1 (token 1): [3.2, -0.1, 1.8, ...] → scale_1 = 3.2/127
Row 2 (token 2): [0.1,  0.3, 0.1, ...] → scale_2 = 0.3/127
```

Token-level scale allows the system to handle the outlier token activations that large language models produce without affecting the precision of normal tokens.

### Per-Group Quantization

A compromise between per-channel and per-token: group adjacent values together and give each group its own scale. Common group sizes: 64, 128.

```python
def group_quantize(weights: np.ndarray, group_size: int = 128) -> dict:
    """
    Quantize weights in groups of group_size.
    Each group of 128 values shares one scale factor.
    """
    original_shape = weights.shape
    weights_flat = weights.reshape(-1)
    n_groups = len(weights_flat) // group_size
    
    quantized = np.zeros_like(weights_flat, dtype=np.int8)
    scales = np.zeros(n_groups, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = weights_flat[start:end]
        
        scale = np.max(np.abs(group)) / 127
        scales[g] = scale
        quantized[start:end] = np.round(group / scale).clip(-127, 127).astype(np.int8)
    
    return {
        "quantized": quantized.reshape(original_shape),
        "scales": scales,
        "group_size": group_size
    }
```

Per-group quantization with group_size=128 is used by GPTQ and AWQ. It gives much better quality than per-channel for INT4 (where per-channel is often insufficient).

---

## Quantization Error: What You Lose

Every quantization introduces error. The error is:

```
quantization_error = original_value - dequantized_value
```

For a given scale, the maximum possible error is:

```
max_error = scale / 2
```

Because rounding to the nearest integer can be off by at most 0.5 steps.

```python
def analyze_quantization_error(weights: np.ndarray, n_bits: int = 8) -> dict:
    """Analyze quantization error for a given weight tensor."""
    
    max_int = 2**(n_bits - 1) - 1
    
    scale = np.max(np.abs(weights)) / max_int
    
    quantized = np.round(weights / scale).clip(-max_int, max_int).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale
    
    error = weights - dequantized
    
    return {
        "scale": scale,
        "max_abs_error": np.max(np.abs(error)),
        "mean_abs_error": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "relative_error": np.mean(np.abs(error)) / np.mean(np.abs(weights))
    }

# Typical results for a 7B model weight matrix:
# INT8 absmax: RMSE ≈ 0.001-0.005 (< 0.1% relative error for most layers)
# INT4 absmax: RMSE ≈ 0.05-0.2   (noticeable quality loss without correction)
# INT2 absmax: RMSE ≈ 0.3-1.0    (severe quality degradation)
```

This is why INT8 is generally safe for most models but naive INT4 requires sophisticated correction methods (GPTQ, AWQ) to maintain quality.

---

## The Quantization-Dequantization Flow at Inference

At inference time, the pipeline is:

```
Stored (INT8 weight) + (FP32 scale)
         │
         │ Dequantize: weight_fp16 = weight_int8 × scale
         ▼
   FP16/BF16 weight
         │
         │ Matrix multiplication with FP16 activation
         ▼
   FP16 output activation
```

The key point: **computation still happens in FP16/BF16**. The INT8 is a storage format, not a compute format (for weight-only quantization). You dequantize just-in-time before the matrix multiply. This means you get the memory savings of INT8 but retain the numerical accuracy of FP16 computation.

Some methods (LLM.int8(), covered in Lesson 4) do the actual matrix multiply in INT8 using dedicated hardware units, which is faster but more complex.

---

## Summary

- **Absmax (symmetric) quantization:** maps [-max_abs, max_abs] to [-127, 127]. Simple, one scale per group. Works well for symmetric distributions (typical weights). Destroyed by outliers.
- **Zero-point (asymmetric) quantization:** maps [min, max] to [-128, 127]. Needs both scale and zero_point. Better for asymmetric distributions (activations, embeddings). More computation at inference.
- **Per-tensor:** one scale for entire tensor. Maximum memory efficiency, minimum quality. Impractical for INT4 and even problematic for INT8 with outliers.
- **Per-channel:** one scale per row/column. Negligible overhead (~0.1%). Significantly better quality. Standard for production quantization.
- **Per-group:** one scale per group of N values (typically 128). Best quality-efficiency tradeoff for INT4. Used by GPTQ and AWQ.
- **Quantization error:** bounded by `scale/2`. Smaller scale (more granular grouping) = less error. The methods in the next lessons all focus on minimizing this error intelligently.

---

## What's Next

Lesson 4 covers Post-Training Quantization (PTQ) — specifically LLM.int8(), which was the first method to make INT8 quantization practical for large language models by solving the activation outlier problem with mixed-precision decomposition.