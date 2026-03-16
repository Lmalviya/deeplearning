# Quantization — Complete Reference Notes

> **Priority Legend used throughout this document:**
> - 🔴 **Must Know** — core concepts, asked in almost every interview
> - 🟡 **Intuition Level** — follow-up questions, senior roles, system design
> - 🟢 **Reference** — implementation detail, good-to-have, look up when needed
>
> **Prerequisite:** Basic familiarity with neural networks and floating point numbers helps but is not required.

---

## Table of Contents

1. [What Is Quantization? (The Universal Concept)](#1-what-is-quantization-the-universal-concept)
2. [How Numbers Are Quantized — The Math](#2-how-numbers-are-quantized--the-math)
3. [Number Representations in Computing](#3-number-representations-in-computing)
4. [Quantization in Machine Learning — Motivation](#4-quantization-in-machine-learning--motivation)
5. [Types of Quantization](#5-types-of-quantization)
6. [Quantization Granularity](#6-quantization-granularity)
7. [Symmetric vs Asymmetric Quantization](#7-symmetric-vs-asymmetric-quantization)
8. [Quantization Variants & Key Algorithms](#8-quantization-variants--key-algorithms)
9. [Calibration](#9-calibration)
10. [Mixed Precision Quantization](#10-mixed-precision-quantization)
11. [Quantization on the Hardware Side](#11-quantization-on-the-hardware-side)
12. [LoRA / QLoRA Connection](#12-lora--qlora-connection)
13. [Failure Modes & Limitations](#13-failure-modes--limitations)
14. [Practical Implementation](#14-practical-implementation)
15. [Summary & Key Takeaways](#15-summary--key-takeaways)

---

## 1. What Is Quantization? (The Universal Concept)

### 🔴 1.1 Definition

Quantization is the process of **mapping values from a large (often continuous or high-precision) space to a smaller (discrete or lower-precision) space**.

More formally:

```
Q : X → X_q

where:
  X   = original value space (e.g., all real numbers, FP32)
  X_q = quantized value space (e.g., integers -128 to 127, INT8)
  Q   = quantization function
```

The key idea: you accept some **information loss** in exchange for **efficiency gains** — less memory, faster compute, lower power consumption.

**One-line intuition:**
> Quantization is compression of the number line — fitting a wide range of values into a small set of discrete buckets.

---

### 🔴 1.2 Everyday Examples

Quantization is not a machine learning concept — it appears everywhere in daily life.

**Example 1: The Clock**
- Reality: time flows continuously (infinite precision)
- A clock face: maps continuous time → 12 discrete hour positions
- A digital clock: maps to minutes (60 values) or seconds (86,400 values per day)
- Quantization error: "It's 2:47" is close but not exactly 2:47:32.8
---

### 🔴 1.3 The Fundamental Trade-off: Precision vs Efficiency

```
HIGH PRECISION                         LOW PRECISION
────────────────────────────────────────────────────
FP64 ──── FP32 ──── FP16 ──── INT8 ──── INT4 ──── INT2
  │                                                  │
More accurate                              Faster & smaller
More memory                                Less memory
Slower compute                             More power-efficient
Harder to compress                         Easier to transmit
```

**The fundamental question quantization always asks:**
> How much precision can I give up before the quality degradation becomes unacceptable?

---

### 🔴 1.4 Key Terminology

| Term | Definition | Example |
|------|-----------|---------|
| **Resolution** | The smallest distinguishable difference between two quantized values | 8-bit audio: 1/256 of full range |
| **Range** | The span of values the quantization scheme can represent | INT8: [-128, 127] |
| **Precision** | How finely values within the range are represented | More bits → more precision |
| **Quantization Error** | Difference between original and quantized value: `e = x - Q(x)` | x=1.7, Q(x)=2 → e=-0.3 |
| **Clipping** | Values outside the representable range are clamped to min/max | x=300 clamped to INT8 max 127 |
| **Granularity** | The level at which quantization parameters are computed (per-tensor, per-channel) | See §6 |
| **Scale factor (s)** | Multiplier mapping real values to integer grid | s = range / (2^bits - 1) |
| **Zero-point (z)** | Offset ensuring zero maps to an integer exactly | See §7 |

---

## 2. How Numbers Are Quantized — The Math

### 🔴 2.1 Continuous → Discrete: The Quantization Function

The general quantization process has three steps:

```
Step 1: SCALE    → map real values to the integer grid
Step 2: ROUND    → snap to nearest integer
Step 3: CLAMP    → ensure values stay within representable range
```

**General quantization formula:**

```
Q(x) = clamp( round(x / s) + z,  q_min,  q_max )

where:
  x     = original real value (FP32)
  s     = scale factor (positive real number)
  z     = zero-point (integer)
  round = nearest integer rounding
  clamp = clip to [q_min, q_max]
  Q(x)  = quantized integer value
```

**Dequantization (reconstructing the real value):**

```
x̂ = s · (Q(x) - z)

where x̂ ≈ x  (with some error)
```

The **quantization-dequantization cycle** is:

```
x  ──[quantize]──►  Q(x)  ──[dequantize]──►  x̂  ≈  x
             (integer)                 (real, approximate)
```

---

### 🔴 2.2 Uniform Quantization — Equal Spacing

In **uniform quantization**, the quantization levels are **equally spaced** across the range.

```
Real line:  |─────────────────────────────────────|
            min                                  max

Levels:     |    |    |    |    |    |    |    |    |
            ▲    ▲    ▲    ▲    ▲    ▲    ▲    ▲    ▲
           -4   -3   -2   -1    0    1    2    3    4    ← 3-bit signed: 9 levels
```

**Scale factor for uniform quantization:**

```
s = (x_max - x_min) / (2^b - 1)

where:
  x_max, x_min = range of values to quantize
  b            = number of bits
  2^b - 1      = number of intervals
```

**Example — 8-bit uniform quantization of range [-1.0, 1.0]:**

```
b = 8  →  2^8 - 1 = 255 levels
s = (1.0 - (-1.0)) / 255 = 2.0 / 255 ≈ 0.00784

Value x = 0.753:
  Q(x) = round(0.753 / 0.00784) = round(96.04) = 96
  x̂    = 96 × 0.00784 = 0.7526
  Error = 0.753 - 0.7526 = 0.0004  ✓ small
```

**Uniform quantization is simple and hardware-friendly.** Most integer quantization (INT8, INT4) uses uniform spacing.

---

### 🟡 2.3 Non-Uniform Quantization — Unequal Spacing

In **non-uniform quantization**, levels are spaced **unevenly** — denser where values cluster, sparser where values are rare.

**Why it matters:** If your data follows a non-uniform distribution (e.g., many values near zero, few near the extremes), uniform spacing wastes levels on rare regions.

```
Uniform (INT4 example):
  ─┼──┼──┼──┼──┼──┼──┼──┼──►
  -8  -6  -4  -2  0   2   4   6   7

Non-uniform (more levels near 0):
  ─┼─┼─┼┼┼┼┼┼┼─┼─┼──►
  -8  -4  -1 0 1   4   7
       (denser near center)
```

**Two common non-uniform schemes:**

**1. Logarithmic quantization:**
```
Levels spaced on a log scale:
  ±{2⁰, 2⁻¹, 2⁻², 2⁻³, ...}
  Good for weights with exponential decay distribution
```

**2. Quantile quantization (used in NF4):**
```
Levels placed at quantile boundaries of the data distribution
Equal probability mass between each pair of adjacent levels
→ More levels where data is dense, fewer where sparse
```

*Note: NF4 (used in QLoRA) is a specific case of quantile quantization for normally distributed weights — covered in §8.3 and your LoRA/QLoRA notes.*

---

### 🔴 2.4 Quantization Error, Clipping Error, and Rounding Error

Total quantization error has two components:

```
Total Error = Rounding Error + Clipping Error
```

**Rounding Error:**
- Caused by snapping values to the nearest quantization level
- Unavoidable — exists even for values within range
- Bounded by ±s/2 (half the step size)
- Decreases as bit width increases

```
Rounding error for value x:
  e_round(x) = x - s · round(x/s)
  |e_round| ≤ s/2
```

**Clipping Error:**
- Caused by values outside [q_min · s, q_max · s] being clamped
- Can be large if range is set too tight
- Eliminated by widening the representable range
- But widening range increases step size → more rounding error

```
Example:
  INT8 range set to [-1.0, 1.0], but data has values up to 3.0
  x = 2.5  → clamped to 1.0 → clipping error = 1.5  ← large!

  Fix: widen range to [-3.0, 3.0]
  But now: s = 6/255 ≈ 0.0235  (vs 0.00784 before)
  Rounding error increases by 3×
```

**The clipping-rounding trade-off:**

```
Narrow range  → small rounding error + large clipping error (for outliers)
Wide range    → large rounding error + small clipping error

Optimal range: minimize total error = find the sweet spot
               (this is what calibration §9 does)
```

---

### 🟡 2.5 Signal-to-Quantization-Noise Ratio (SQNR)

**SQNR** measures quantization quality — higher is better.

```
SQNR = 10 · log₁₀(Signal Power / Quantization Noise Power)  [dB]

For uniform quantization of a full-scale sinusoidal signal:
  SQNR ≈ 6.02 · b + 1.76  dB

where b = number of bits
```

**SQNR per bit depth:**

| Bits | SQNR (approx) | Use case |
|------|--------------|----------|
| 4 | ~26 dB | Aggressive compression |
| 8 | ~50 dB | Production ML inference |
| 16 | ~98 dB | Training, high quality |
| 32 | ~194 dB | Reference / full precision |

**Key rule: every additional bit adds ~6 dB of SQNR.**

This is why going from FP32 → INT8 (a 4× memory reduction) reduces SQNR by ~144 dB — significant but often acceptable for inference.

---

### 🔴 2.6 Worked Numerical Example — Full Quantization Cycle

**Setup:** Quantize the value x = -0.38 using 4-bit signed uniform quantization over the range [-0.5, 0.5].

**Step 1: Determine parameters**
```
b     = 4 bits
range = [-0.5, 0.5]
q_min = -8   (signed 4-bit: -(2^3) = -8)
q_max =  7   (signed 4-bit: 2^3 - 1 = 7)

Scale: s = (0.5 - (-0.5)) / (7 - (-8))
          = 1.0 / 15
          ≈ 0.06667

Zero-point: z = 0  (symmetric, see §7)
```

**Step 2: Quantize**
```
Q(x) = clamp( round(x / s) + z,  -8,  7 )
      = clamp( round(-0.38 / 0.06667) + 0,  -8,  7 )
      = clamp( round(-5.7),  -8,  7 )
      = clamp( -6,  -8,  7 )
      = -6                              ← quantized integer
```

**Step 3: Dequantize**
```
x̂ = s · (Q(x) - z)
   = 0.06667 · (-6 - 0)
   = -0.4000
```

**Step 4: Compute error**
```
Rounding error = x - x̂ = -0.38 - (-0.40) = +0.02
Relative error = 0.02 / 0.38 ≈ 5.3%
```

**Interpretation:** The value -0.38 is stored as integer -6, and reconstructed as -0.40. A 5.3% error — acceptable for many use cases at only 4 bits.

---

## 3. Number Representations in Computing

### 🔴 3.1 Integer Formats

Integers store whole numbers only — no fractional part.

| Format | Bits | Signed Range | Unsigned Range | Memory |
|--------|------|-------------|----------------|--------|
| INT32 | 32 | -2.1B to 2.1B | 0 to 4.3B | 4 bytes |
| INT16 | 16 | -32,768 to 32,767 | 0 to 65,535 | 2 bytes |
| INT8 | 8 | -128 to 127 | 0 to 255 | 1 byte |
| INT4 | 4 | -8 to 7 | 0 to 15 | 0.5 bytes |
| INT2 | 2 | -2 to 1 | 0 to 3 | 0.25 bytes |
| INT1 | 1 | {-1, 1} (1-bit) | {0, 1} | 0.125 bytes |

**For ML:** INT8 is the workhorse of inference quantization. INT4 is used for aggressive compression (GPTQ, QLoRA). INT1/INT2 are research-level (binary/ternary networks).

---

### 🔴 3.2 Floating Point Formats

Floating point stores numbers in scientific notation form: **±mantissa × 2^exponent**

| Format | Total Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|-----------|------|----------|----------|-------|-----------|
| FP64 | 64 | 1 | 11 | 52 | ±1.8×10³⁰⁸ | ~15-16 decimal digits |
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits |
| FP16 | 16 | 1 | 5 | 10 | ±65,504 | ~3-4 decimal digits |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2-3 decimal digits |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | ±448 | ~1-2 decimal digits |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | ±57,344 | ~1 decimal digit |

---

### 🔴 3.3 Anatomy of a Float — Sign, Exponent, Mantissa

Every floating point number has three fields:

```
FP32 bit layout (32 bits total):
┌───┬────────────────────┬────────────────────────────────────────────────┐
│ S │      Exponent      │                   Mantissa                     │
│ 1 │       8 bits       │                   23 bits                      │
└───┴────────────────────┴────────────────────────────────────────────────┘
  ▲           ▲                                  ▲
Sign bit   Biased exp                      Fractional part
(0=+, 1=-)  (actual exp = stored - 127)    (1.mantissa in binary)
```

**Decoding formula:**

```
value = (-1)^S × 2^(E - bias) × (1 + M/2^23)

where:
  S    = sign bit (0 or 1)
  E    = stored exponent (8-bit integer)
  bias = 127 for FP32
  M    = mantissa bits (integer representation)
```

**Example — FP32 encoding of 0.15625:**

```
0.15625 = 1.25 × 2^(-3)

Sign: 0 (positive)
Exponent: -3 + 127 = 124 → binary: 01111100
Mantissa: 1.25 = 1 + 0.25 = 1 + 1/4 → .01000...0

FP32 bits: 0 | 01111100 | 01000000000000000000000
```

---

### 🔴 3.4 Dynamic Range vs Precision Trade-off

**This is the most critical concept for choosing between FP16 and BF16.**

- **Exponent bits** control **dynamic range** (how large/small values can be)
- **Mantissa bits** control **precision** (how finely values are represented)

```
FP16: 5 exponent bits,  10 mantissa bits → narrow range, high precision
BF16: 8 exponent bits,  7 mantissa bits  → wide range,  lower precision

FP16 max value:  65,504
BF16 max value:  ~3.4 × 10³⁸  (same as FP32!)
```

**Why BF16 is preferred for ML training:**

Neural networks can produce large intermediate values (gradients, activations). FP16's max of 65,504 causes **overflow** (values → inf) frequently during training. BF16 inherits FP32's exponent range, preventing overflow while halving memory.

```
Training stability comparison:
  FP32  →  stable, expensive
  BF16  →  stable (same range as FP32), half the memory  ← preferred
  FP16  →  unstable (overflow risk), requires loss scaling tricks
```

**Why FP16 is used at inference:**
Inference values are bounded (post-training, controlled). The narrow range is acceptable and hardware has fast FP16 kernels.

---

### 🔴 3.5 Comparison Table: All Formats Side by Side

| Format | Bits | Memory/param | Max Value | Precision | Best Use Case |
|--------|------|-------------|-----------|-----------|---------------|
| FP64 | 64 | 8 bytes | 1.8×10³⁰⁸ | Highest | Scientific computing |
| FP32 | 32 | 4 bytes | 3.4×10³⁸ | High | Training reference |
| BF16 | 16 | 2 bytes | 3.4×10³⁸ | Medium | Training (Ampere+) |
| FP16 | 16 | 2 bytes | 65,504 | Medium | Inference |
| FP8 E4M3 | 8 | 1 byte | 448 | Low | Forward pass |
| FP8 E5M2 | 8 | 1 byte | 57,344 | Very low | Backward pass |
| INT8 | 8 | 1 byte | 127 | Low | Inference (post-quant) |
| INT4 | 4 | 0.5 bytes | 7 | Very low | Aggressive compression |
| NF4 | 4 | 0.5 bytes | 1.0 (norm.) | Non-uniform | QLoRA base models |

---

## 4. Quantization in Machine Learning — Motivation

### 🔴 4.1 Why Neural Network Weights Suit Quantization

Neural network weights are surprisingly quantization-friendly for several reasons:

**Reason 1: Weights follow a smooth distribution**
Trained weights are not arbitrary — they cluster around zero and follow a roughly normal distribution. This means most weights occupy a narrow range, making low-bit representation efficient.

**Reason 2: Models are over-parameterized**
Large models have far more parameters than strictly necessary. Some redundancy can be sacrificed to compression with minimal accuracy impact.

**Reason 3: Robustness through depth**
Errors introduced at one layer are often compensated by subsequent layers. A small quantization error in layer 3 doesn't catastrophically affect the final output — the network "absorbs" small perturbations.

**Reason 4: Relative values matter more than absolute values**
In softmax operations, attention, and other functions, the relative ordering of values matters more than their exact magnitudes. Quantization preserves relative ordering well.

---

### 🔴 4.2 Memory, Compute, and Bandwidth Savings

**Memory savings — direct and immediate:**

```
LLaMA-2 7B parameter count: 7,000,000,000

Format     Bits/param    Memory
FP32       32            ~26.0 GB
BF16/FP16  16            ~13.0 GB
INT8       8             ~6.5 GB
INT4       4             ~3.25 GB
```

**Compute savings — via integer arithmetic:**

On modern hardware (NVIDIA GPUs):
```
Operation         Throughput (A100)
FP32 GEMM         312 TFLOPS
FP16/BF16 GEMM    624 TFLOPS   (2× FP32)
INT8 GEMM         1,248 TOPS   (4× FP32)
INT4 GEMM         2,496 TOPS   (8× FP32)  ← theoretical
```

**Bandwidth savings — often the real bottleneck:**

For large models, inference is **memory-bandwidth-bound**, not compute-bound. The GPU spends more time waiting for weights to load from VRAM than actually computing.

```
Memory bandwidth  =  bytes to transfer  /  time
                  ∝  model size

INT8 model: 4× less data to transfer → up to 4× faster inference
(in the memory-bound regime)
```

---

### 🟡 4.3 The Distribution of Weights — Why Normal Distribution Matters

After training, neural network weights in most layers follow an approximately **normal (Gaussian) distribution** centered at zero.

```
Weight distribution (typical trained layer):

    Count
      │         ████
      │       ████████
      │     ████████████
      │   ████████████████
      │ ████████████████████
      │─────────────────────── Weight value
           -σ   0   +σ

~ N(0, σ²)  with small σ (e.g., 0.01 - 0.1)
```

**Implications for quantization:**

1. Most weight values are near zero → quantization levels should be dense near zero
2. The tails contain very few values → can afford sparse levels there
3. INT8 with uniform spacing works well for normal distributions
4. This is exactly why **NF4** (non-uniform, quantile-based) outperforms INT4 — it respects the normal distribution

---

### 🔴 4.4 The Outlier Problem — The Silent Killer of Quantization Quality

**The single most important practical challenge in quantization.**

**What are outliers?**
In some transformer layers (especially attention projections in large models), a small number of weights or activations have **dramatically larger magnitudes** than the rest.

```
Typical layer (well-behaved):
  values in range [-0.5, 0.5]  →  INT8 works great

Layer with outliers:
  99% of values in [-0.5, 0.5]
  1% of values up to [-100, 100]  ← outliers!
```

**Why outliers destroy quantization:**

If you set your quantization range to cover outliers:
```
s = 200 / 255 ≈ 0.784   (for range [-100, 100] with INT8)

Now quantizing a normal value x = 0.3:
  Q(x) = round(0.3 / 0.784) = round(0.38) = 0
  x̂    = 0 × 0.784 = 0.0
  Error = 0.3 - 0.0 = 0.3  ← 100% error!
```

The outliers force a large step size, making normal values indistinguishable (all round to 0, 1, or -1).

**Where outliers appear in transformers:**
- Specific dimensions in attention key/query projections
- Certain hidden dimensions in FFN layers
- Consistent across inputs — same dimensions are always outliers

**Solutions to the outlier problem:**
- Percentile calibration (ignore top X% during range setting) — §9.4
- SmoothQuant (migrate outliers to weights) — §8.5
- Per-channel quantization (each channel has its own scale) — §6.2
- Mixed precision (keep outlier-heavy layers in FP16) — §10

---

## 5. Types of Quantization

### 🔴 5.1 Post-Training Quantization (PTQ)

**Definition:** Quantize a model **after training is complete**. No retraining or fine-tuning.

```
WORKFLOW:
  Trained model (FP32)
       │
       ▼
  [Calibration]  ← run a small dataset through to observe activations
       │
       ▼
  Compute scale factors (s) and zero-points (z) per layer
       │
       ▼
  Quantized model (INT8 / INT4)
       │
       ▼
  Deploy for inference
```

PTQ has two sub-types:

#### Static PTQ

Scale factors are computed **once** (during calibration) and **fixed** for inference.

```
Calibration phase (offline):
  Run N samples → observe activation ranges → compute s, z

Inference phase:
  Use fixed s, z for all inputs → fast!
```

**Pros:**
- No runtime overhead for computing scales
- Hardware-friendly (scales are constants)
- Fast inference

**Cons:**
- Calibration data must represent deployment distribution
- Poor if activation ranges vary wildly between inputs

**Best for:** Classification, NLP tasks with predictable input distributions.

#### Dynamic PTQ

Scale factors are computed **on the fly** for each input during inference.

```
Inference phase (per input):
  Step 1: Observe actual min/max of activations for this input
  Step 2: Compute s = (max - min) / 255
  Step 3: Quantize with this dynamic scale
  Step 4: Compute output
```

**Pros:**
- No calibration data needed
- Adapts to each input's actual distribution
- Better accuracy for variable-range inputs

**Cons:**
- Runtime overhead of computing scales per input
- Less hardware-friendly (scales change each step)

**Best for:** Sequence models with variable-length inputs, generative models.

---

### 🔴 5.2 Quantization-Aware Training (QAT)

**Definition:** Simulate quantization **during training** so the model learns to be robust to quantization noise.

```
WORKFLOW:
  Training loop:
    Forward pass:  FP32 weights → fake-quantize → FP32 activations
    Backward pass: use Straight-Through Estimator (STE) for gradients
    Weight update: update FP32 weights (not quantized)

  Deploy:
    Quantize final FP32 weights → INT8 model
```

**The key challenge — rounding is not differentiable:**

```
round(x) has gradient = 0 everywhere except at discontinuities

Problem: gradient descent needs gradients to update weights
Solution: Straight-Through Estimator (STE)
```

**Straight-Through Estimator (STE):**

```
Forward:   Q(x) = round(x/s) · s     ← actual quantization
Backward:  ∂Q/∂x ≈ 1                 ← pretend gradient passes through

(If x is outside clipping range, gradient = 0)
```

The STE is an approximation — it ignores the true (zero) gradient of rounding. But empirically it works well because the model learns to keep weights in ranges where rounding error is small.

**QAT Training diagram:**

```
FP32 weights
     │
     ▼
┌─────────────────┐
│  Fake Quantize  │  ← simulates INT8 precision
│  (round + STE)  │
└────────┬────────┘
         │
         ▼ (quantization-aware activations)
   [Forward pass]
         │
         ▼
   [Loss computation]
         │
         ▼
   [Backward pass] ← STE lets gradients flow through fake-quant
         │
         ▼
   Update FP32 weights
```

---

### 🔴 5.3 Comparison: PTQ vs QAT

| Dimension | PTQ | QAT |
|-----------|-----|-----|
| **Training required?** | No | Yes (fine-tuning or full retraining) |
| **Compute cost** | Very low | High (training cost) |
| **Accuracy** | Good (INT8), degraded (INT4) | Better, close to FP32 |
| **Time to deploy** | Fast (hours) | Slow (days to weeks) |
| **Calibration data** | Small dataset needed | Full training data |
| **Best bit-width** | INT8 | INT4 and below |
| **Model size** | Works on any size | Expensive for large models |
| **Use case** | Production inference, LLMs | Edge deployment, mobile, INT4 |

**Rule of thumb:**
- INT8: PTQ almost always sufficient
- INT4 or below: QAT often needed for quality
- LLMs (7B+): PTQ with calibration (GPTQ, AWQ) is standard

---

### 🟡 5.4 Weight-Only vs Weight + Activation Quantization

**Weight-only quantization:**
- Only the model weights (W) are quantized
- Activations remain in FP16/BF16
- Simpler: no calibration for activations needed
- Less memory reduction (activations also consume memory)
- **Most common for LLM inference (GPTQ, AWQ, QLoRA)**

```
Weight-only:
  y = Dequant(W_INT4) · x_FP16
      ────────────────────────
      dequant happens per layer, compute in FP16
```

**Weight + Activation quantization (W8A8, W4A8):**
- Both weights AND activations quantized to INT
- Full integer arithmetic (no dequantization during compute)
- Maximum hardware speedup (uses INT8 tensor cores)
- Harder: activations vary per input → calibration critical
- **Used in TensorRT, ONNX runtime, edge deployment**

```
W8A8:
  y = INT8_GEMM(W_INT8, x_INT8)  ← pure integer compute!
      ─────────────────────────
      true integer matrix multiply → maximum speedup
```

**Naming convention: W{w}A{a}**
- W8A8 = 8-bit weights, 8-bit activations
- W4A16 = 4-bit weights, 16-bit activations (weight-only)
- W4A8 = 4-bit weights, 8-bit activations

---

## 6. Quantization Granularity

### 🔴 6.1 Per-Tensor Quantization

**One scale factor and zero-point for the entire weight tensor.**

```
W = [[0.1, -0.5,  0.3],
     [0.8, -0.2,  0.6],
     [-0.4, 0.9, -0.7]]

Single scale: s = max(|W|) / 127 = 0.9 / 127 ≈ 0.00709
Single zero:  z = 0

All 9 values quantized with the same s
```

**Pros:** Simplest, smallest metadata overhead, most hardware-friendly.
**Cons:** One outlier in the entire tensor degrades all values' precision.

**Used in:** Basic PTQ, older TensorFlow Lite models.

---

### 🔴 6.2 Per-Channel Quantization

**One scale factor per output channel (row or column of the weight matrix).**

```
W = [[0.1, -0.5,  0.3],    → s₁ = 0.5/127
     [0.8, -0.2,  0.6],    → s₂ = 0.8/127
     [-0.4, 0.9, -0.7]]    → s₃ = 0.9/127

Each row gets its own scale based on its own max value
```

**Why this helps:**
Different channels in a neural network have different value ranges. Channel 2 might have weights in [-0.8, 0.8] while channel 5 has weights in [-0.02, 0.02]. Per-tensor quantization wastes precision on channel 5 (step size determined by channel 2's range).

**Pros:** Better accuracy than per-tensor, handles channel-wise distribution differences.
**Cons:** Slightly more metadata, requires per-channel dequantization.

**Used in:** Almost all modern INT8 inference (TensorRT, PyTorch quantization).

---

### 🔴 6.3 Per-Group / Per-Block Quantization

**One scale factor per small group of weights (e.g., every 64 or 128 consecutive weights).**

```
Weight vector: [w₁, w₂, ..., w₅₁₂]

Group size = 64:
  Group 1: [w₁  ... w₆₄]   → s₁
  Group 2: [w₆₅ ... w₁₂₈]  → s₂
  ...
  Group 8: [w₄₄₉... w₅₁₂]  → s₈
```

**Why this is powerful:**
Within a single weight matrix row, different segments can have very different ranges. Per-group quantization adapts to local structure within the tensor.

**The extreme case:** Group size = 1 → every weight has its own scale = full precision (defeats the purpose).
**The other extreme:** Group size = entire tensor → per-tensor quantization.

**Group size is a precision-efficiency knob:**
```
Smaller group → better precision → more scale metadata → more memory overhead
Larger group  → worse precision  → less scale metadata → less overhead

Common choices: 64, 128 (balance point)
```

**Used in:** GPTQ, AWQ, QLoRA (NF4 uses group size 64), llama.cpp GGUF formats.

---

### 🔴 6.4 Comparison Table

| Granularity | # Scale factors | Accuracy | Overhead | Hardware support |
|-------------|----------------|----------|----------|-----------------|
| Per-tensor | 1 per tensor | Lowest | None | Excellent |
| Per-channel | 1 per channel | Good | Small | Good |
| Per-group (g=128) | n/128 per tensor | Better | Moderate | Moderate |
| Per-group (g=64) | n/64 per tensor | Best | Higher | Moderate |
| Per-element | 1 per weight | Perfect | Huge | Poor (defeats purpose) |

**Interview rule:** Per-channel for activations is rare (activations change per input); per-channel for weights is standard. Per-group is the go-to for INT4 weight-only quantization of LLMs.

---

## 7. Symmetric vs Asymmetric Quantization

### 🔴 7.1 Symmetric Quantization

**The quantization range is symmetric around zero: [-α, +α]**

Zero-point z = 0 always. The quantization formula simplifies to:

```
Q(x) = clamp( round(x / s),  -2^(b-1),  2^(b-1) - 1 )

s = max(|x|) / (2^(b-1) - 1)    ← scale = max absolute value / max int

Dequant: x̂ = s · Q(x)           ← no zero-point term
```

**Example — 8-bit symmetric:**
```
Weights in range [-0.9, 0.7]
max(|w|) = 0.9

s = 0.9 / 127 ≈ 0.00709
z = 0

Quantize w = 0.7:
  Q(0.7) = round(0.7 / 0.00709) = round(98.7) = 99
  x̂     = 99 × 0.00709 = 0.702  ← small error ✓

Quantize w = -0.9:
  Q(-0.9) = round(-0.9 / 0.00709) = -127  ← uses full negative range

Note: positive max only reaches 0.9/127×127 = 0.9 ✓
      but negative range [-128 to -127] → -128 is unused
      (slight asymmetry — "signed symmetric")
```

**Pros:** Simple arithmetic, no zero-point addition needed, hardware-friendly.
**Cons:** Wastes range when data is not centered at zero (e.g., activations after ReLU are always ≥ 0).

**Best for:** Weights (approximately zero-centered), symmetric activation distributions.

---

### 🔴 7.2 Asymmetric Quantization

**The quantization range is [α, β] where α ≠ -β. Zero-point z ≠ 0.**

```
s = (β - α) / (2^b - 1)
z = round(-α / s)               ← zero-point shifts the range

Q(x) = clamp( round(x/s) + z,  0,  2^b - 1 )   ← unsigned typically

Dequant: x̂ = s · (Q(x) - z)
```

**Example — 8-bit asymmetric for ReLU output:**
```
Activations after ReLU: range [0, 3.6]  (always non-negative!)

Symmetric would use range [-3.6, 3.6] → wastes ALL negative codes!
Asymmetric:
  s = 3.6 / 255 ≈ 0.01412
  z = round(-0 / 0.01412) = 0    (min is 0, so z=0 here)

  Actually for range [0.5, 3.6]:
  s = (3.6 - 0.5) / 255 = 3.1/255 ≈ 0.01216
  z = round(-0.5 / 0.01216) = round(-41.1) ≈ -41
  → zero maps to integer 41

  Now ALL 256 codes are used for [0.5, 3.6] ← efficient!
```

**Pros:** Uses full integer range for any distribution, better accuracy for non-zero-centered data.
**Cons:** Zero-point subtraction adds one operation per multiply-accumulate, slightly more complex hardware.

**Best for:** Activations (especially after ReLU/GELU which have non-symmetric outputs).

---

### 🔴 7.3 Math Formulation Summary

**Symmetric:**
```
Forward:  Q(x) = clamp(round(x / s), q_min, q_max)
Backward: x̂   = s · Q(x)
Scale:    s    = max(|x|) / q_max
Zero-pt:  z    = 0
```

**Asymmetric:**
```
Forward:  Q(x) = clamp(round(x / s) + z, q_min, q_max)
Backward: x̂   = s · (Q(x) - z)
Scale:    s    = (x_max - x_min) / (q_max - q_min)
Zero-pt:  z    = round(q_min - x_min / s)
```

---

### 🟡 7.4 When Asymmetric Wins

| Layer/Tensor | Distribution | Better Choice |
|---|---|---|
| Weights (linear layers) | ~N(0, σ) | Symmetric |
| Activations after ReLU | [0, +∞) truncated | Asymmetric |
| Activations after GELU | Mostly positive, small negative tail | Asymmetric |
| Softmax outputs | [0, 1] | Asymmetric |
| Embedding tables | ~centered at 0 | Symmetric |
| Biases | Various | Asymmetric |

**Practical standard:**
- Weights → symmetric INT8
- Activations → asymmetric INT8
- This combination gives the best accuracy-efficiency balance for W8A8

---

## 8. Quantization Variants & Key Algorithms

### 🟡 8.1 GPTQ — One-Shot Weight Quantization

**Paper:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)

**Problem GPTQ solves:**
Naive round-to-nearest quantization accumulates error across rows of a weight matrix. GPTQ finds a better quantization by using second-order information (Hessian) to compensate for each quantized weight.

**Core Intuition:**
When you quantize weight w_i (introduce error δ_i), compensate by adjusting the remaining unquantized weights w_{i+1}, ..., w_n to minimize the total output error.

```
Standard quantization:
  Round each weight independently → errors accumulate

GPTQ:
  Round w₁ → adjust w₂, w₃, ... to compensate
  Round w₂ → adjust w₃, w₄, ... to compensate
  ...
  Each quantization step is locally optimal
```

**The math (key idea):**

The optimal weight update after quantizing w_q is derived from the inverse Hessian H⁻¹:

```
Error from quantizing w_q: eq = quant(w_q) - w_q

Compensation for remaining weights:
  δW = -eq · H⁻¹[q, F] / H⁻¹[q, q]

where:
  H    = Hessian of the layer's output error w.r.t. weights
  F    = set of remaining (not yet quantized) weight indices
  q    = index of currently quantized weight
```

**GPTQ uses the Cholesky decomposition** of H⁻¹ for efficient computation and processes weights in blocks (not one at a time) for GPU efficiency.

**Key properties:**
- Post-training: no gradient updates, just clever rounding
- Calibration data: ~128 sequences from training data
- Achieves INT4 quality close to FP16 on most LLMs
- Per-group (g=128) quantization recommended

**Typical results:**
```
LLaMA-2 7B perplexity (WikiText-2):
  FP16:          5.47
  GPTQ INT4:     5.61  (+0.14, ~2.5% degradation)
  Naive INT4:    6.80  (+1.33, much worse)
```

---

### 🟡 8.2 AWQ — Activation-Aware Weight Quantization

**Paper:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)

**Key insight (different from GPTQ):**
Not all weights are equally important. Weights that are multiplied by **large activations** matter more — their quantization error gets amplified.

```
Output error from weight error δw:
  δy = δw · x

If x is large (activation outlier), even small δw causes large δy
→ Protect weights corresponding to large-activation channels
```

**AWQ's approach: per-channel scaling**

Instead of quantizing all weights equally, AWQ scales important weights to increase their effective precision:

```
Step 1: Find important channels
  importance[i] = mean(|x[i]|)   ← activation magnitude per channel

Step 2: Scale weights in important channels
  W̃[:,i] = W[:,i] / s[i]        ← shrink important weights
  x̃[i]   = x[i] · s[i]         ← correspondingly scale activation

  (The product W·x = W̃·x̃ is unchanged — mathematically equivalent)

Step 3: Quantize W̃
  Now important channels have smaller magnitude → better precision after quant

Step 4: At inference
  y = Quant(W̃) · (s · x)       ← apply scale to input
```

**Why shrinking important weights helps:**
After division by s[i], important weights fit within a narrower range → quantization step size is smaller → less rounding error on these critical weights.

**AWQ advantages over GPTQ:**
- No Hessian computation (lighter calibration)
- Hardware-friendly (only per-channel scales needed)
- Designed for efficient kernel implementation
- Better on smaller calibration sets

**AWQ vs GPTQ:**

| Dimension | GPTQ | AWQ |
|-----------|------|-----|
| Calibration method | Hessian (second-order) | Activation magnitudes (first-order) |
| Compute cost | Higher | Lower |
| Quality | Slightly better | Close |
| Hardware efficiency | Good | Better |
| Best use case | Maximum accuracy | Fast deployment |

---

### 🟡 8.3 NF4 — NormalFloat 4-bit

*Full coverage in your LoRA/QLoRA notes. Brief recap here for completeness.*

**Key idea:** 4-bit quantization levels placed at **quantile midpoints of N(0,1)** — equal probability mass between adjacent levels.

```
NF4 levels (16 values for 4 bits):
{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0}

Notice: denser near 0 (where weight density is highest)
        sparser at extremes (where weight density is lowest)
```

**Position in the quantization landscape:**
NF4 is a non-uniform, data-type-level design choice — it bakes the normal distribution assumption directly into the quantization codec. Most other methods use uniform quantization with calibration to set the range.

---

### 🟢 8.4 GGUF / llama.cpp Quantization Formats

**GGUF** (formerly GGML) is the file format used by llama.cpp — the CPU-first LLM inference library.

**GGUF quantization types:**

| Format | Bits | Description |
|--------|------|-------------|
| Q2_K | ~2.6 | 2-bit with 4-bit scales per group of 16 |
| Q3_K | ~3.4 | 3-bit with 6-bit scales |
| Q4_0 | 4.0 | Simple 4-bit, per-group of 32 |
| Q4_K | ~4.5 | 4-bit with 6-bit scales per block |
| Q5_K | ~5.5 | 5-bit with better scales |
| Q6_K | ~6.6 | 6-bit, near-lossless |
| Q8_0 | 8.0 | 8-bit, minimal quality loss |

**Key distinction from GPU formats:**
GGUF targets **CPU inference** with ARM NEON / AVX2 SIMD instructions. The quantization schemes are co-designed with CPU vector instruction widths. This is why Q4_0 uses groups of 32 (fits in AVX2 registers).

---

### 🟡 8.5 SmoothQuant — Migrating Outliers from Activations to Weights

**Paper:** "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022)

**Problem:** W8A8 (quantizing both weights AND activations) is hard because activation outliers destroy quantization quality. Weights are easy to quantize; activations are hard.

```
Weights:     smooth distribution → easy to quantize
Activations: outliers in fixed channels → hard to quantize

Observation: outliers appear in CONSISTENT channels across different inputs
→ We can handle them systematically!
```

**SmoothQuant's key insight:**
The problematic channels are known (from calibration). We can mathematically migrate the quantization difficulty from activations to weights, which are easier to handle.

**The smoothing operation:**

```
y = (x · diag(s)⁻¹) · (diag(s) · W)
  = x̃ · W̃

where:
  x̃ = x / s    ← smooth activations (divide by per-channel scale)
  W̃ = s · W    ← absorb scale into weights

The math is equivalent: y = xW = x̃W̃
But now x̃ has smaller outliers, and W̃ is slightly harder to quantize
```

**Choosing the smoothing factor s:**

```
s[i] = max(|x[:,i]|)^α / max(|W[i,:]|)^(1-α)

where α ∈ [0, 1] controls the migration strength:
  α = 0: no migration (original)
  α = 0.5: balanced (recommended default)
  α = 1: full migration to weights
```

**Result:** Enables true W8A8 quantization on LLMs that previously required W8A16 or FP16 activations.

---

### 🔴 8.6 Comparison Table: All Quantization Algorithms

| Algorithm | Type | Precision | Key Innovation | Best For |
|-----------|------|-----------|---------------|----------|
| Round-to-nearest | PTQ | INT8 | Baseline | Simple INT8 |
| GPTQ | PTQ | INT4 | Hessian-based error compensation | Max accuracy INT4 |
| AWQ | PTQ | INT4 | Activation-aware channel scaling | Fast INT4 deployment |
| NF4 | PTQ | 4-bit | Quantile-based non-uniform levels | Normally distributed weights |
| SmoothQuant | PTQ | W8A8 | Activation-to-weight outlier migration | True integer inference |
| QAT | Training | INT4/INT8 | Simulate quant during training | Edge/mobile INT4 |
| GGUF/Q4_K | PTQ | ~4.5 bit | CPU-optimized mixed precision | CPU inference |

---

## 9. Calibration

### 🔴 9.1 What Calibration Is and Why It's Needed

Calibration is the process of **determining optimal quantization parameters** (scale s, zero-point z) for each layer by observing how actual data flows through the network.

**Why not just use the theoretical weight range?**
- Weights are static: you can compute their range exactly from the trained model
- **Activations are dynamic:** they change with every input. You need representative data to estimate their ranges.

```
Without calibration (for activations):
  Use min/max over all possible inputs → impossible
  Use min/max of first batch → may not represent full distribution

With calibration:
  Run ~100-1000 representative samples
  Track activation statistics per layer
  Compute s, z from observed statistics
```

---

### 🟡 9.2 Calibration Dataset Selection

**Key requirements:**
- Must represent the **deployment distribution** (not just training data)
- Size: typically 128–1024 samples is sufficient
- Diversity: cover different topics, lengths, languages if multilingual

**Common mistake:** Using only easy/clean samples. The quantization range must accommodate outlier inputs too.

---

### 🔴 9.3 Min-Max Calibration

**Method:** Set range to the absolute min and max observed values across all calibration samples.

```
For each layer l:
  Run all calibration samples through
  x_min[l] = min of all activation values seen
  x_max[l] = max of all activation values seen
  s[l]     = (x_max - x_min) / (2^b - 1)
```

**Pros:** Simple, guaranteed no clipping on calibration data.
**Cons:** Single outlier sets the range for all values → wastes precision if outliers exist.

---

### 🟡 9.4 Percentile Calibration

**Method:** Use the p-th and (100-p)-th percentile instead of absolute min/max.

```
For each layer l:
  Collect all activation values from calibration
  x_min[l] = p-th percentile     (e.g., 0.1th percentile)
  x_max[l] = (100-p)-th percentile (e.g., 99.9th percentile)

Common choices: 99.9%, 99.99%, 99.999%
```

**Effect:** Clips the top/bottom 0.1% of values, allowing much tighter range for the remaining 99.9%.

```
Example:
  99.9% of activations in [-2.0, 2.0]
  0.1% are outliers up to [-50, 50]

Min-max:      s = 100/255 = 0.392  → coarse precision for normal values
Percentile:   s = 4/255  = 0.0157  → 25× finer precision, clips 0.1% outliers
```

**Trade-off:** You accept clipping error for the outlier values in exchange for much better precision for the majority.

---

### 🟡 9.5 KL-Divergence Calibration

**Method:** Find the quantization range that minimizes the KL divergence between the original floating-point distribution and the quantized distribution.

```
For a set of candidate ranges [α, β]:
  Quantize activations to INT8 using this range
  Compute KL(P_fp32 || P_int8)   ← divergence between distributions

Choose the range that minimizes KL divergence
```

**Intuition:** Instead of minimizing raw numerical error, minimize the statistical difference between original and quantized distributions. This often gives better task performance even if raw MSE is slightly higher.

**Used in:** NVIDIA TensorRT calibration, Intel Neural Compressor.

**Pros:** Often best accuracy for classification tasks.
**Cons:** More expensive (requires searching over candidate ranges).

---

### 🟡 9.6 Comparison of Calibration Strategies

| Strategy | Cost | Accuracy | Sensitivity to outliers | Best for |
|----------|------|----------|------------------------|----------|
| Min-Max | Very low | Lowest | Very high | Quick baseline |
| Percentile (99.9%) | Low | Good | Low | General purpose |
| Percentile (99.99%) | Low | Better | Medium | When some outliers OK |
| KL-Divergence | Medium | Best | Low | Classification, vision |
| MSE minimization | Medium | Best | Low | Regression tasks |

---

## 10. Mixed Precision Quantization

### 🟡 10.1 What It Is and Why Uniform Bit-Width Is Suboptimal

**Uniform quantization:** all layers get the same bit-width (e.g., all INT8).

**Reality:** Different layers have very different sensitivities to quantization error.

```
Layer sensitivity varies:
  First embedding layer     → very sensitive (input representation)
  Last output layer         → very sensitive (logit magnitudes matter)
  Middle attention layers   → moderately sensitive
  Some FFN layers           → less sensitive

Using INT4 everywhere = bad accuracy in sensitive layers
Using INT8 everywhere = wasteful in insensitive layers
```

**Mixed precision:** assign bit-width per layer based on sensitivity.

```
Mixed precision example:
  Layer 1 (embedding):    FP16    ← sensitive, keep high precision
  Layers 2-10 (FFN):      INT8
  Layers 11-20 (attn):    INT4    ← insensitive to compression
  Layer 21 (LM head):     FP16    ← sensitive, keep high precision

Result: better accuracy than uniform INT4, less memory than uniform INT8
```

---

### 🟡 10.2 Sensitivity Analysis

**How to determine which layers to quantize aggressively:**

**Method 1: Perturbation analysis**
```
For each layer l:
  Quantize ONLY layer l to INT4
  Measure accuracy drop on validation set
  Sensitivity[l] = accuracy drop

High sensitivity → use INT8 or FP16
Low sensitivity  → safe to use INT4
```

**Method 2: Hessian-based sensitivity (used in GPTQ)**
```
Sensitivity[l] ∝ trace(H_l)

where H_l is the Hessian of loss w.r.t. layer l weights
High trace → output changes a lot with small weight perturbation → sensitive
```

**Method 3: Output MSE**
```
Sensitivity[l] = ||W_l · x - Quant(W_l) · x||²

Measure how much quantizing layer l changes its output
```

---

### 🟡 10.3 Common Mixed Precision Patterns

**LLM inference standard pattern:**

```
Component               Precision       Reason
─────────────────────────────────────────────────────
Token embedding         FP16            Input quality
First 2 transformer     FP16 / INT8     Sensitive early layers
Middle transformers     INT8 / INT4     Main compression target
Last 2 transformers     INT8            Sensitive final layers
LM head (vocab proj)    FP16            Logit precision critical
LayerNorm / RMSNorm     FP16/FP32       Tiny, numerically sensitive
```

**GGUF's Q4_K_M pattern (llama.cpp default):**
- Most layers: 4-bit
- Attention K/Q matrices: 6-bit (more sensitive)
- Embedding: 16-bit or 8-bit
- Output: 8-bit

**Why the first and last layers stay in FP16:**
- First layer: the entire model's representations depend on clean input embeddings. Quantization error here propagates everywhere.
- Last layer: small errors in logits can flip the top-1 prediction (e.g., the difference between predicting "yes" and "no").

---

## 11. Quantization on the Hardware Side

### 🟡 11.1 How INT8 Operations Are Faster Than FP32

**Floating point vs integer arithmetic:**

```
FP32 multiplication:
  1. Align exponents
  2. Multiply mantissas (23-bit × 23-bit)
  3. Normalize result
  4. Handle special cases (inf, NaN)
  → Complex, requires FPU (Floating Point Unit)

INT8 multiplication:
  1. 8-bit × 8-bit integer multiply
  → Simple, cheap, pipelined easily
```

**SIMD (Single Instruction Multiple Data) advantage:**

Modern CPUs and GPUs can pack multiple integers into wide registers:
```
AVX2 (256-bit):
  4 × FP32 operations per instruction
  8 × INT32 operations per instruction
  16 × INT16 operations per instruction
  32 × INT8 operations per instruction   ← 8× throughput vs FP32!
```

---

### 🟡 11.2 CUDA Tensor Cores and Quantization

NVIDIA Tensor Cores are specialized matrix multiply units with quantization-aware designs:

| GPU Generation | FP16 (TFLOPS) | INT8 (TOPS) | INT4 (TOPS) |
|---------------|--------------|-------------|-------------|
| V100 | 125 | 62 INT8 | N/A |
| A100 | 312 | 624 | 1,248 |
| H100 | 989 | 1,979 | 3,958 |
| RTX 4090 | 82.6 | 165.2 | 330.3 |

**Key observation:** INT8 throughput is exactly 2× FP16 on A100/H100, and INT4 is 4× FP16.

**But: memory bandwidth often limits actual speedup**

```
Roofline model:
  If model is memory-bandwidth bound (most LLM inference):
    Actual speedup ≈ memory reduction (4× for INT8 vs FP32)

  If model is compute-bound (large batch training):
    Actual speedup ≈ TOPS ratio (2× or 4×)

  LLM autoregressive inference: batch=1 → strongly memory-bound
  → INT8 gives ~4× speedup from bandwidth savings alone
```

---

### 🟡 11.3 Memory Bandwidth vs Compute — Which Is the Bottleneck?

**Arithmetic Intensity:** Operations per byte loaded from memory.

```
Arithmetic Intensity = FLOPs / Bytes

Matrix multiply W (M×N) × x (N×K):
  FLOPs = 2 × M × N × K
  Bytes = M×N × dtype_size + N×K × dtype_size

For LLM inference (batch=1, K=1, single token):
  FLOPs = 2 × M × N           (one matrix-vector product)
  Bytes = M × N × 4 (FP32)    (entire weight matrix loaded)
  Intensity = 2 / 4 = 0.5 FLOP/byte  ← very low!

A100 can do: 2,000 GFLOPS / 2,000 GB/s = 1 FLOP/byte (roofline)
Our intensity: 0.5 < 1 → MEMORY BOUND
```

**This is why quantization helps so much for LLM inference:**
- Halving weight size → halving memory bandwidth needed → near 2× speedup
- Even without faster compute instructions

---

### 🟢 11.4 Practical Speedup Numbers

**LLaMA-2 7B inference benchmarks (approximate):**

| Setup | Tokens/sec | Memory | vs FP16 baseline |
|-------|-----------|--------|-----------------|
| FP16 (A100 40GB) | 50 tok/s | 14 GB | 1.0× |
| INT8 bitsandbytes | 40 tok/s | 7 GB | 0.8× (memory savings, slight speed reduction) |
| INT4 GPTQ | 80 tok/s | 4 GB | 1.6× |
| INT4 AWQ | 90 tok/s | 4 GB | 1.8× |
| NF4 QLoRA | 35 tok/s | 5 GB | 0.7× (dequant overhead) |

*Note: Speed depends heavily on batch size, sequence length, and hardware. These are single-GPU, batch=1 estimates.*

---

## 12. LoRA / QLoRA Connection

*You have full coverage in your LoRA/QLoRA notes. This section positions those concepts within the broader quantization landscape.*

### 🟡 12.1 How NF4 Fits Into the Broader Quantization Landscape

NF4 is a **non-uniform, data-type-level quantization** scheme — it's not just a calibration technique but a new 4-bit floating point format designed specifically for normally distributed neural network weights.

```
Quantization landscape:
  Uniform INT quantization  → GPTQ, AWQ, SmoothQuant
  Uniform FP quantization   → FP8 (E4M3, E5M2)
  Non-uniform quantization  → NF4 ← here
  Learned non-uniform       → QAT with custom codebooks
```

NF4's placement: it sits between INT4 (uniform) and FP8 (floating point). It achieves near-FP16 quality for weight-only quantization because the normal distribution assumption about weights is almost universally correct for transformer models.

### 🟡 12.2 Double Quantization as Meta-Quantization

Double quantization (from QLoRA) is an elegant recursive application of quantization:

```
Level 1: Quantize weights W → W_NF4 using scale c₁ (FP32)
Level 2: Quantize the scales c₁ → c₁_FP8 using scale c₂ (FP32)

Memory:
  W_NF4:   4 bits/param
  c₁:      32 bits per 64 params = 0.5 bits/param  ← quantize this!
  c₁_FP8:  8 bits per 256 c₁ values
           = 8 / (256 × 64) bits/param ≈ 0.0005 bits/param  ← tiny!
  c₂:      32 bits per 256 c₁ values  ← negligible

Net: reduces overhead from 0.5 bits to ~0.127 bits per parameter
```

This is a general idea applicable beyond QLoRA: **quantize your quantization constants** whenever they contribute non-trivially to memory overhead.

---

## 13. Failure Modes & Limitations

### 🔴 13.1 When PTQ Fails

**Failure 1: Outlier-dominated layers**

*Symptom:* Large accuracy drop after INT8 or INT4 PTQ despite calibration.
*Cause:* Activation outliers in specific channels force a wide quantization range, making normal values indistinguishable.
*Diagnosis:* Plot per-channel activation distributions. Look for channels with max(|x|) >> mean(|x|).
*Fix:* Use per-channel quantization, percentile calibration, SmoothQuant, or keep problematic layers in FP16 (mixed precision).

---

**Failure 2: Calibration distribution mismatch**

*Symptom:* Model works well on calibration data but degrades on deployment data.
*Cause:* Calibration dataset doesn't represent real inputs. Scale factors are optimized for the wrong distribution.
*Fix:* Use diverse, representative calibration data. Include edge cases and domain-specific vocabulary.

---

**Failure 3: Quantizing sensitive layers aggressively**

*Symptom:* Perplexity increase much larger than expected from bit-width reduction.
*Cause:* First/last layers, embedding layers, or normalization layers quantized too aggressively.
*Fix:* Always keep embedding, LM head, and LayerNorm in FP16. Use sensitivity analysis before applying INT4.

---

**Failure 4: Small model + aggressive quantization**

*Symptom:* INT4 works for 13B model but fails for 1B model.
*Cause:* Smaller models are less redundant — each weight carries more information. Less capacity to absorb quantization error.
*Fix:* Use INT8 for smaller models, reserve INT4 for 7B+. Consider QAT for small models that need INT4.

---

### 🔴 13.2 When QAT Fails

**Failure 1: Training instability with very low bit-width**

*Symptom:* QAT loss oscillates or diverges for INT4.
*Cause:* The STE gradient approximation becomes very inaccurate at low bit-widths. Gradients that should be zero (clipped regions) are passed through.
*Fix:* Use lower learning rate during QAT fine-tuning. Gradually reduce bit-width (start INT8, then INT4).

---

**Failure 2: Catastrophic forgetting during QAT**

*Symptom:* Model adapts to quantization but loses general capabilities.
*Cause:* QAT fine-tunes on a narrow dataset. Without careful data mixing, model forgets pre-training knowledge.
*Fix:* Mix QAT training data with diverse general data. Use knowledge distillation: match outputs of FP32 teacher model.

---

**Failure 3: Prohibitive compute cost for large models**

*Symptom:* QAT on a 7B model requires weeks of GPU time.
*Cause:* QAT requires full backward passes with fake-quantization, adding ~2-3× overhead vs standard training.
*Fix:* Use GPTQ or AWQ instead (PTQ). For LLMs, PTQ with calibration is usually close enough to QAT quality. QAT for LLMs is generally not practical without significant infrastructure.

---

### 🟡 13.3 Activation Quantization Challenges

Activations are fundamentally harder to quantize than weights because:

1. **They are dynamic** — change with every input
2. **They have outliers** — specific dimensions consistently produce large values (LLM.int8() paper showed some dimensions 100× larger than others)
3. **They are not normally distributed** — post-ReLU activations are non-negative; post-attention activations can be highly skewed
4. **Inter-layer dependencies** — quantization error in layer N directly affects layer N+1 inputs

**This is why W4A16 (weight-only INT4) is more common than W4A8 (both INT4) for LLMs.**

---

### 🔴 13.4 Inherent Limitations of Quantization

These cannot be fixed by better algorithms — they are fundamental:

**1. Lossy compression — irreducible error floor**
Quantization always introduces error. At 4 bits, you have 16 possible values per weight. No calibration or algorithm can make 16 levels represent infinite precision exactly. There is a quality floor below which you cannot go at a given bit-width.

**2. Distribution assumption brittleness**
All quantization schemes make assumptions about data distribution (normal for NF4, uniform for INT8). When weights violate these assumptions (after unusual training procedures, sparse models, etc.), quantization quality degrades unexpectedly.

**3. Task-specific degradation is unpredictable**
A model might quantize well on perplexity (a general metric) but fail on specific downstream tasks. Quantization can disproportionately hurt rare capabilities that rely on precise weight values in a small number of neurons.

**4. Compounding error across layers**
In deep networks, quantization error accumulates. The last layer sees the accumulated errors from all previous layers. This is why deep networks and very aggressive bit-widths (INT2, INT1) are especially hard to quantize.

**5. No free lunch on quality-compression trade-off**
At some point, reducing bits will always hurt model quality. The frontier is pushed by better algorithms (GPTQ → AWQ → future methods), but the trade-off curve cannot be eliminated — only shifted.

---

## 14. Practical Implementation

### 🟢 14.1 Libraries

| Library | Platform | Key features | Best for |
|---------|----------|-------------|----------|
| **bitsandbytes** | NVIDIA GPU | INT8/NF4 PTQ, paged optimizers | QLoRA training, quick INT8 inference |
| **AutoGPTQ** | NVIDIA GPU | GPTQ INT4, fast kernels | INT4 inference on GPUs |
| **AutoAWQ** | NVIDIA GPU | AWQ INT4, GEMM kernels | Fast INT4, production |
| **llama.cpp** | CPU + GPU | GGUF, all quantization levels | CPU inference, edge devices |
| **TensorRT** | NVIDIA GPU | W8A8, W4A8, engine optimization | Maximum throughput production |
| **ONNX Runtime** | CPU + GPU | INT8 PTQ, cross-platform | Cross-platform deployment |
| **Intel Neural Compressor** | CPU | INT8, mixed precision | Intel CPU deployment |

---

### 🟢 14.2 Code: INT8 PTQ with bitsandbytes

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# INT8 quantization config (LLM.int8())
bnb_config_int8 = BitsAndBytesConfig(
    load_in_8bit=True,                          # Enable INT8 quantization
    llm_int8_threshold=6.0,                     # Outlier threshold (default 6.0)
    llm_int8_has_fp16_weight=False,             # Keep weights in INT8
    llm_int8_skip_modules=["lm_head"],          # Keep LM head in FP16
)

# Load model with INT8 quantization (PTQ, no calibration needed)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config_int8,
    device_map="auto",
)

# Model is ready — no further steps needed
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# Check quantization was applied
for name, module in model.named_modules():
    if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
        print(f"{name}: {module.weight.dtype}")
# Output: many layers show torch.int8
```

---

### 🟢 14.3 Code: GPTQ Quantization with AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Step 1: Define quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,                    # INT4 quantization
    group_size=128,            # Per-group of 128 weights
    damp_percent=0.01,         # Hessian damping factor
    desc_act=False,            # Whether to quantize in activation order
)

# Step 2: Load model for quantization
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Step 3: Prepare calibration data (128 samples recommended)
calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=512, truncation=True)["input_ids"]
    for text in calibration_texts  # list of representative strings
]

# Step 4: Run GPTQ quantization (this takes 10-30 min for 7B)
model.quantize(calibration_data)

# Step 5: Save quantized model
model.save_quantized("./llama-2-7b-gptq-int4")

# Step 6: Load for inference (fast)
quantized_model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-int4",
    use_triton=False,           # Use CUDA kernels (triton for more speed)
    device="cuda:0",
)
```

---

### 🟢 14.4 Choosing the Right Quantization Strategy

```
START: Deploy a quantized LLM
          │
          ▼
What is your primary constraint?
  ├── Memory (must fit on X GB GPU)
  │       │
  │       ▼
  │   How much can you compromise on quality?
  │     ├── Minimal: INT8 (bitsandbytes load_in_8bit)
  │     ├── Moderate: INT4 GPTQ (AutoGPTQ, group=128)
  │     └── Aggressive: INT4 AWQ or NF4 (QLoRA)
  │
  ├── Throughput (maximize tokens/sec)
  │       │
  │       ▼
  │   Have GPU with Tensor Cores?
  │     ├── Yes: AWQ INT4 + vLLM or TensorRT
  │     └── No (CPU): llama.cpp GGUF Q4_K_M
  │
  ├── Quality (minimize accuracy drop)
  │       │
  │       ▼
  │   Model size?
  │     ├── < 7B: INT8 PTQ or QAT
  │     └── ≥ 7B: GPTQ INT4 (quality near INT8)
  │
  └── Fine-tuning on quantized model
          │
          ▼
      Use QLoRA (NF4 base + LoRA adapters)
```

---

### 🟢 14.5 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Quantizing embedding + LM head | Large perplexity spike | Always skip first/last layers |
| Wrong calibration distribution | Good lab results, bad production | Use diverse, real-world calibration data |
| Per-tensor for activation outliers | Accuracy drop on specific tasks | Use per-channel or SmoothQuant |
| INT4 on small models (<3B) | Catastrophic quality loss | Use INT8 for models < 7B |
| Missing bitsandbytes CUDA install | Runtime CUDA errors | `pip install bitsandbytes --upgrade` + verify GPU |
| FP16 vs BF16 mismatch | Overflow NaN during inference | Check GPU supports BF16 (Ampere+), else use FP16 |
| Forgetting to eval after quantization | Silently degraded model | Always run benchmark before deploy |

---

## 15. Summary & Key Takeaways

### 🔴 15.1 Decision Tree: Choosing Quantization Strategy

```
TRAINING or INFERENCE?
  ├── Training:
  │     → Use BF16 (mixed precision with FP32 master weights)
  │     → Add QAT only if deploying to INT4 edge devices
  │
  └── Inference:
        │
        ├── Target: GPU server
        │     ├── 7B model, quality priority  → INT8 PTQ (bitsandbytes)
        │     ├── 7B+ model, speed priority   → INT4 AWQ (AutoAWQ + vLLM)
        │     ├── 7B+ model, quality + small  → INT4 GPTQ (group=128)
        │     └── Fine-tune + serve           → NF4 QLoRA → merge → deploy
        │
        └── Target: CPU / Edge
              ├── Max quality       → GGUF Q8_0 or Q6_K
              ├── Balanced          → GGUF Q4_K_M  (default recommendation)
              └── Max compression   → GGUF Q2_K (significant quality loss)
```

---

### 🔴 15.2 Master Comparison Table

| Method | Type | Bit-width | Hardware | Quality | Speed | Memory | Use case |
|--------|------|-----------|----------|---------|-------|--------|----------|
| FP32 | — | 32 | All | Baseline | 1× | 1× | Training |
| BF16 | — | 16 | Ampere+ | ≈FP32 | 2× | 0.5× | Training/Inference |
| FP16 | — | 16 | All GPU | ≈FP32 | 2× | 0.5× | Inference |
| INT8 PTQ | PTQ | 8 | All GPU | ~99% | 2-4× | 0.25× | Standard inference |
| GPTQ | PTQ | 4 | NVIDIA | ~97% | 3-4× | 0.125× | LLM inference |
| AWQ | PTQ | 4 | NVIDIA | ~97% | 3-5× | 0.125× | Fast LLM inference |
| NF4 | PTQ | 4 | NVIDIA | ~97% | 2-3× | ~0.13× | QLoRA training |
| QAT | Training | 4-8 | All | ~98-99% | 3-4× | 0.125× | Edge deployment |
| GGUF Q4_K | PTQ | ~4.5 | CPU | ~97% | varies | ~0.14× | CPU inference |
| SmoothQuant | PTQ | W8A8 | NVIDIA | ~99% | 4× | 0.25× | True INT8 inference |

---

### 🔴 15.3 Top 10 Interview Questions & Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | What is quantization? | Mapping values from a high-precision space (FP32) to a lower-precision space (INT8/INT4) to save memory and speed up compute, at the cost of some accuracy. |
| 2 | What is the difference between PTQ and QAT? | PTQ quantizes after training (fast, no retraining, slight quality loss); QAT simulates quantization during training (slower, better quality, needed for INT4 on small models). |
| 3 | Why is BF16 preferred over FP16 for training? | BF16 has the same 8-bit exponent as FP32 (same dynamic range), avoiding overflow. FP16's 5-bit exponent can overflow on large gradient values. |
| 4 | What is the outlier problem in quantization? | A few large-valued weights/activations force a wide quantization range, making the step size large and causing most normal values to quantize poorly. |
| 5 | What is per-channel vs per-tensor quantization? | Per-tensor uses one scale for the whole matrix; per-channel uses one scale per row/column. Per-channel is more accurate as it adapts to each channel's distribution. |
| 6 | What is symmetric vs asymmetric quantization? | Symmetric centers the range at zero (zero-point=0); asymmetric allows any range [α,β] with a non-zero zero-point. Asymmetric is better for non-zero-centered data like ReLU activations. |
| 7 | What does GPTQ do differently from naive rounding? | GPTQ uses Hessian-based error compensation: after quantizing each weight, it adjusts remaining weights to minimize output error, yielding much better INT4 quality. |
| 8 | What is calibration and why is it needed? | Calibration runs representative data through the model to determine the activation ranges (scale factors) for PTQ. Needed because activations change with input and can't be determined from weights alone. |
| 9 | Why does quantization help inference speed? | Two ways: (1) reduced memory → faster memory bandwidth for memory-bound ops; (2) integer arithmetic is faster than float on modern hardware (INT8 GEMM is 2× faster than FP16 on A100). |
| 10 | When would you choose INT8 over INT4? | For smaller models (<7B), quality-critical applications, or when model accuracy drop at INT4 is unacceptable. INT8 has minimal quality loss while INT4 can degrade significantly on sensitive layers or small models. |

---

### 🔴 15.4 Quick Reference Cheat Sheet

```
MEMORY FOOTPRINT (per parameter):
  FP32  = 4 bytes    BF16/FP16 = 2 bytes
  INT8  = 1 byte     INT4/NF4  = 0.5 bytes

QUANTIZATION FORMULA:
  Q(x) = clamp(round(x/s) + z,  q_min,  q_max)
  x̂   = s · (Q(x) - z)
  s    = (x_max - x_min) / (2^b - 1)

SQNR RULE: each bit ≈ +6 dB of quality
  4-bit ≈ 26 dB,  8-bit ≈ 50 dB,  16-bit ≈ 98 dB

WEIGHTS vs ACTIVATIONS:
  Weights    → static, ~N(0,σ), symmetric quantization
  Activations→ dynamic, outliers, asymmetric, harder to quantize

KEY FORMAT FACTS:
  BF16: same range as FP32, lower precision → training
  FP16: narrow range, higher precision → inference
  NF4:  quantile-based, optimal for normal distributions → QLoRA

ALGORITHM SELECTION:
  Quick INT8:  bitsandbytes load_in_8bit
  Best INT4:   AutoGPTQ (max quality) or AutoAWQ (max speed)
  CPU:         llama.cpp GGUF Q4_K_M
  Fine-tune:   QLoRA (NF4 + LoRA)
```
