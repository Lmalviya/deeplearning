# Quantization Lesson 2 — Number Formats: The Foundation

---

## Why Number Formats Are the Starting Point

Before you can understand what quantization does, you need to understand what it is replacing. Neural networks — including every LLM you have worked with — store their weights as floating point numbers. The choice of floating point format determines how much memory the model occupies, how fast arithmetic runs on GPU, and how much precision is available for representing weight values.

Quantization is fundamentally the act of moving from a high-precision format to a lower-precision format. To understand what you gain and lose, you need to understand what is inside each format.

---

## How Floating Point Numbers Work

A floating point number stores three components:

```
[sign bit] [exponent bits] [mantissa bits]
```

- **Sign:** 1 bit. Whether the number is positive or negative.
- **Exponent:** Encodes the magnitude (the power of 2). More exponent bits = wider range of representable values.
- **Mantissa (fraction/significand):** Encodes the precision. More mantissa bits = more decimal precision within a given range.

The value represented is:

```
value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa)
```

The key insight: floating point trades off **range** (exponent) against **precision** (mantissa). This is exactly the trade-off that different formats make differently.

---

## FP32 — Full Precision (32-bit float)

```
[1 sign] [8 exponent] [23 mantissa] = 32 bits = 4 bytes per value
```

FP32 is the standard format for neural network training. It has been the default since the beginning of deep learning.

**What FP32 gives you:**
- Exponent range: approximately 10^-38 to 10^38 (enormous range)
- Precision: ~7 significant decimal digits
- Can represent numbers as small as 1.4 × 10^-45 and as large as 3.4 × 10^38

**The memory cost:**
A 7 billion parameter model in FP32:
```
7,000,000,000 parameters × 4 bytes = 28 GB
```
This requires two A100 80GB GPUs just to hold the weights — before activations, optimizer states, or gradients.

**When FP32 is used:** Increasingly rare in practice. Some operations in mixed-precision training, master weight copies in optimizer. Almost never used for inference.

---

## FP16 — Half Precision (16-bit float)

```
[1 sign] [5 exponent] [10 mantissa] = 16 bits = 2 bytes per value
```

FP16 cuts FP32's memory in half by reducing both exponent and mantissa bits.

**What FP16 gives you:**
- Exponent range: approximately 6 × 10^-5 to 65,504 (dramatically narrower than FP32)
- Precision: ~3-4 significant decimal digits

**The critical problem with FP16: overflow and underflow.**

The maximum representable value in FP16 is **65,504**. If any value exceeds this during forward pass or backward pass, you get **overflow** (NaN or infinity). This happens frequently in practice — gradient accumulation, loss scaling, and LayerNorm can all produce values above 65,504.

```python
import torch

fp32_value = torch.tensor(70000.0, dtype=torch.float32)
fp16_value = fp32_value.to(torch.float16)

print(fp16_value)  # → inf  (overflow — 70000 > 65504)
```

The underflow problem: values smaller than ~6 × 10^-5 become zero. Small gradients vanish.

**Mixed precision training** addresses these problems: keep a master copy of weights in FP32 for updates, but run forward/backward pass in FP16. Loss scaling multiplies the loss before backward pass to prevent underflow.

**When FP16 is used:** Inference on NVIDIA GPUs (Tensor Cores accelerate FP16 matrix multiplication). Mixed-precision training. Model storage format.

---

## BF16 — Brain Float 16 (16-bit, Google's format)

```
[1 sign] [8 exponent] [7 mantissa] = 16 bits = 2 bytes per value
```

BF16 is the same bit width as FP16 but allocates bits completely differently: it keeps all 8 exponent bits from FP32 but truncates the mantissa from 23 bits to just 7.

**BF16 vs FP16 — the critical difference:**

| Format | Exponent bits | Mantissa bits | Max value | Min normal | Precision |
|---|---|---|---|---|---|
| FP32 | 8 | 23 | ~3.4 × 10^38 | ~1.2 × 10^-38 | ~7 decimal digits |
| FP16 | 5 | 10 | 65,504 | ~6 × 10^-5 | ~3-4 decimal digits |
| BF16 | 8 | 7 | ~3.4 × 10^38 | ~1.2 × 10^-38 | ~2-3 decimal digits |

BF16 has the same exponent range as FP32. This means **no overflow** and **no underflow** problems. It sacrifices precision (fewer mantissa bits) but preserves range.

For neural network training, range matters more than precision. Weight values and gradients can span a wide dynamic range during training, but the exact precision of each value matters less — SGD is noisy anyway.

**BF16 limitation:** Requires specific hardware support (A100, H100, TPUs). RTX 3090/4090 consumer GPUs support BF16 with reduced throughput. Older hardware does not support it.

**When BF16 is used:**
- Training on modern GPUs (A100+) — preferred over FP16 for stability
- Inference on A100+ — matches FP16 throughput, better precision than FP16 for some workloads
- Default format for LLaMA, Mistral, and most modern open-source LLM releases

```python
# Converting between formats
import torch

model_fp32 = model.float()        # FP32 (4 bytes/param)
model_fp16 = model.half()         # FP16 (2 bytes/param)
model_bf16 = model.bfloat16()    # BF16 (2 bytes/param)

# Memory comparison for 7B model
# FP32: 28 GB
# FP16: 14 GB
# BF16: 14 GB
```

---

## INT8 — 8-bit Integer

```
[8 integer bits] = 1 byte per value
```

INT8 is not a floating point format — it is a fixed-precision integer. Signed INT8 represents integers from -128 to 127. Unsigned INT8 represents 0 to 255.

The key difference from floating point: **no exponent**. There is no dynamic range. Every value is equally spaced between the minimum and maximum.

**This is the core challenge of quantization:** neural network weights are floating point numbers spread across a wide, non-uniform range. Mapping them to evenly-spaced integers loses precision for the values that are clustered closely together (most of them) and may lose the ability to represent extreme values at all.

**Why INT8 is valuable:**
- Cuts FP32 memory by 4× (1 byte vs 4 bytes per weight)
- Integer arithmetic is faster than float on many hardware units
- Modern GPUs have dedicated INT8 tensor core units

**7B model in INT8:**
```
7,000,000,000 × 1 byte = 7 GB
```
Fits comfortably on a single A10G (24GB VRAM).

---

## INT4 — 4-bit Integer

```
[4 integer bits] = 0.5 bytes per value
```

INT4 represents integers from -8 to 7 (signed) or 0 to 15 (unsigned). Only 16 distinct values.

With only 16 possible values to map the entire distribution of a weight matrix to, precision loss is severe if done naively. The methods that make INT4 practical (GPTQ, AWQ, NF4) are the subject of the next several lessons.

**7B model in INT4:**
```
7,000,000,000 × 0.5 bytes = 3.5 GB
```
Fits on a single 8GB consumer GPU (RTX 3070, Apple M1 Pro).

---

## The Precision-Memory-Speed Triangle

Every quantization decision lives in this triangle:

```
         Higher Precision
         (FP32, BF16)
               │
               │  More accurate, fewer artifacts
               │  Higher memory, slower on some hardware
               │
   ─────────────────────────────
               │
               │  Faster, less memory
               │  Potential accuracy loss
               │
         Lower Precision
     (INT8, INT4, INT2)
```

The formats in order of memory cost:

| Format | Bytes/param | 7B model size | 70B model size |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 (e.g. NF4) | 0.5 | 3.5 GB | 35 GB |
| INT2 | 0.25 | 1.75 GB | 17.5 GB |

INT2 exists (QuIP, QuIP#) but quality degradation is severe for most models. The practical floor for usable model quality is INT4 with a good quantization method.

---

## Why Weights and Activations Are Treated Differently

Models have two things to quantize: **weights** (stored between runs) and **activations** (computed at inference time).

**Weight quantization:**
- Weights are fixed after training — you can analyze their distribution in advance.
- They follow near-Gaussian distributions centered at zero.
- You have time to optimize the quantization mapping (GPTQ, AWQ take minutes to hours).

**Activation quantization:**
- Activations are dynamic — they depend on the input.
- They often have outliers (a small number of extremely large values) — this is the core challenge discovered by the LLM.int8() paper.
- Must be quantized on-the-fly during inference (cannot precompute).

Most practical quantization schemes today focus primarily on **weight-only quantization** (INT4 or INT8 weights, FP16 activations) because activation outliers make full INT8 quantization much harder.

---

## Summary

- **FP32:** 4 bytes, full range and precision. Standard for training master weights. Too large for LLM inference on single GPUs.
- **FP16:** 2 bytes, half precision. Narrow range (max 65,504) causes overflow. Requires loss scaling for stable training. Fast on GPU tensor cores.
- **BF16:** 2 bytes, same range as FP32 but less precision. No overflow/underflow problems. Preferred for modern LLM training and inference.
- **INT8:** 1 byte, fixed-precision integers. 4× smaller than FP32. Requires mapping the float distribution to the -128 to 127 range — this mapping is the quantization problem.
- **INT4:** 0.5 bytes, only 16 possible values. Extreme memory efficiency but requires sophisticated quantization methods to preserve quality.
- Weights and activations are quantized differently. Activation outliers (discovered in large models) make full INT8 quantization harder — modern methods typically use INT4/INT8 weights with FP16 activations.

---

## What's Next

Lesson 3 covers the mechanics of actually performing quantization: how you map a floating point range to an integer range using absmax (symmetric) and zero-point (asymmetric) quantization, and why the scale and granularity of this mapping matters so much for quality.