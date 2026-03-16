# Day 1 — Phase 2 Addendum: Double Quantization

> **Add this to your existing QLoRA notes**
> Revision time: 5 minutes

---

## Why Quantization Constants Exist

Quantization does not compress the entire model at once.
It works in **blocks** of weights (typically 64 weights per block).

**Why blocks and not the entire layer?**

```
If you quantize an entire layer together:
  One outlier weight (e.g. 100.0) forces
  a huge scale factor → all other weights
  lose precision because their range is
  tiny compared to the outlier.

Block-wise quantization:
  Each block of 64 weights gets its OWN
  scale and zero_point constants.
  Outliers only damage their local block.
  Rest of model keeps full precision range.
```

To dequantize back during forward pass:

$$\text{actual\_value} = \text{quantized\_int} \times \text{scale} + \text{zero\_point}$$

These two numbers per block — **scale** and **zero\_point** — are the quantization constants.

---

## The Problem — Constants Take Significant Memory

```
7B model example:
  Total weights      = 7,000,000,000
  Block size         = 64
  Number of blocks   = 7B ÷ 64 = ~109M blocks

  Each block needs:
    scale      → fp32 = 4 bytes
    zero_point → fp32 = 4 bytes

  Total constants memory:
    109M × 8 bytes = ~875 MB ≈ 0.5–1GB
```

Nearly **1GB just for decoding constants** —
before storing a single activation or gradient.

---

## Double Quantization — The Fix

Quantize the quantization constants themselves:

```
Regular QLoRA:
  Model weights     → 4-bit NF4   ✅ compressed
  Scale constants   → fp32        ❌ not compressed
  Zero points       → fp32        ❌ not compressed

Double Quantization:
  Model weights     → 4-bit NF4   ✅ compressed
  Scale constants   → 8-bit       ✅ also compressed
  Zero points       → 8-bit       ✅ also compressed
```

Memory saving for 7B model:

$$875\text{ MB} \times \frac{4\text{-bit}}{32\text{-bit}} \approx 219\text{ MB}$$

$$\text{Saving} = 875 - 219 \approx \textbf{650 MB} \approx 0.5\text{–}0.7\text{ GB}$$

---

## Why 8-bit for Constants — Not 4-bit?

```
Model weights → 4-bit is safe
  LoRA adapters correct for quantization errors
  Errors stay isolated to individual weights

Constants → must use 8-bit, NOT 4-bit
  Constants are the DECODER for entire blocks
  Error in one constant = error amplified
  across ALL 64 weights in that block

  4-bit constants → cascading precision loss
  8-bit constants → sufficient precision
                    to decode weights accurately
```

---

## Why This Matters in Practice

```
7B model, QLoRA, single GPU:

Without double quantization:  ~5.7 GB → does not fit on 6GB GPU
With double quantization:     ~5.0 GB → fits on RTX 3060 / 3080

0.5–0.7 GB is the difference between
accessible consumer hardware and not.
At 70B scale this saving grows to ~7GB.
```

---

## Interview One-Liner

> *"Double quantization applies two compression levels in QLoRA.
> Model weights go to 4-bit NF4, and then the quantization
> constants themselves are compressed to 8-bit — not 4-bit,
> because errors in constants amplify across entire weight blocks
> rather than staying isolated. This saves 0.5–0.7GB on 7B models,
> which is often the difference between fitting on a consumer GPU or not."*
