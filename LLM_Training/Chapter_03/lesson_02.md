# Chapter 3 · Lesson 2 — Mixed Precision Training: FP16 vs. BF16, Loss Scaling

> **Where this fits:** Lesson 1 built a decoder block assuming full fp32 precision everywhere. Real training never runs in pure fp32 — it's too slow and too memory-hungry at scale. This lesson is about the precision tradeoffs that make large-scale training feasible at all.

---

## 1. Why Precision Is a Real Lever, Not Just a Speed Knob

Every tensor in the forward and backward pass is stored in some floating-point format. The format determines three things simultaneously: memory footprint, compute throughput (modern GPUs have dedicated fast paths for lower precision), and numerical range/precision. Training entirely in fp32 is numerically safest but roughly 2x the memory and meaningfully slower than using 16-bit formats on hardware built for them (Tensor Cores on NVIDIA GPUs, for example, get large throughput gains in fp16/bf16 over fp32).

The problem: naively running everything in 16-bit precision breaks training. Understanding *why* is the actual interview-grade content here.

---

## 2. FP16 — Where It Breaks, and Why

FP16 (IEEE half precision) has **10 mantissa bits and 5 exponent bits**. The narrow exponent range is the problem: the smallest representable positive normal number is around `6.1 × 10⁻⁵`. Gradients in deep networks routinely have values smaller than that, especially early in training or in deep layers — and anything smaller than the representable range simply becomes **zero**. This is called **underflow**, and it silently destroys gradient information.

```
FP16 representable range:  ~6.0 × 10⁻⁵  to  ~65,504
Typical small gradient:     ~1.0 × 10⁻⁶   → rounds to 0 in fp16, gradient lost
```

**The fix: loss scaling.** Multiply the loss by a large constant (e.g., 1024 or 65536) before calling `.backward()`. Because gradients scale linearly with the loss (chain rule), every gradient in the backward pass gets scaled up by the same factor — pushing small gradients back into fp16's representable range. After the backward pass, divide the gradients by that same scale factor before the optimizer step, so the actual parameter update is correct.

```python
scale = 1024.0
scaled_loss = loss * scale
scaled_loss.backward()  # gradients are now ~1024x larger, avoiding underflow

for p in model.parameters():
    if p.grad is not None:
        p.grad /= scale  # undo the scaling before the optimizer sees it
```

**Dynamic loss scaling** (what production code actually uses, e.g. PyTorch's `torch.cuda.amp.GradScaler`) adjusts the scale factor automatically: if gradients overflow to `inf`/`NaN` at a given scale, it halves the scale and skips that optimizer step; if training goes many steps without overflow, it increases the scale to use more of fp16's range. This handles the tuning automatically rather than requiring a hand-picked constant.

---

## 3. BF16 — Why It Sidesteps the Whole Problem

BF16 (bfloat16) has **7 mantissa bits and 8 exponent bits** — the same exponent range as fp32, just less mantissa precision.

```
                exponent bits    mantissa bits    dynamic range
FP32                8                 23           ~1e-38 to 1e38
FP16                5                 10           ~6e-5  to 65504
BF16                8                  7           ~1e-38 to 1e38  (same as fp32!)
```

Because BF16's exponent range matches fp32 exactly, the underflow problem from Section 2 essentially doesn't happen — small gradients stay representable. **The tradeoff:** BF16 has less mantissa precision than FP16 (7 bits vs. 10), so individual numbers are less precise — but in practice, this matters far less for deep learning training than range does, because neural network training is fairly tolerant of precision noise (it's already using stochastic gradient estimates) but *not* tolerant of gradients silently becoming zero.

**This is why BF16 dominates modern LLM training** (used by GPT-3 onward, LLaMA, PaLM, and essentially every major open training recipe): it gets almost all the speed/memory benefit of 16-bit training, without needing loss scaling at all, because the exponent range problem that necessitated loss scaling in fp16 doesn't exist in bf16.

---

## 4. Mixed Precision — Not Everything Runs in 16-bit

"Mixed precision" is the accurate term because certain operations still need higher precision to remain numerically stable, regardless of which 16-bit format you use for the bulk of computation:

| Component | Precision used | Why |
|---|---|---|
| Matrix multiplications (the bulk of FLOPs) | fp16 or bf16 | This is where the speed/memory win comes from — dominates compute cost |
| Master copy of weights (for the optimizer update) | fp32 | Small parameter updates (`lr * gradient`) can be smaller than what 16-bit precision can represent relative to the weight's magnitude; keeping a fp32 master copy avoids updates silently vanishing over many steps |
| Loss computation / reduction | fp32 | Directly from Chapter 2, Lesson 1 — summing many values in reduced precision compounds rounding error |
| LayerNorm / softmax internals | Often fp32 internally even in a "bf16" model | These involve operations (exponentials, small variance divisions) that are more numerically sensitive |

**Worked example of the master-weights problem:** a weight is `0.842531`, and an update step computes `lr * gradient = 0.0000012`. In fp16 or bf16, a number that small relative to `0.84` may round away to nothing when added — the update is silently lost. Keeping a separate fp32 master copy of the weights, updated in fp32 precision, and only *casting down* to bf16/fp16 for the forward/backward compute, avoids this — this is standard practice, not an edge-case optimization.

---

## 5. Code: The Practical Pattern

```python
import torch

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# BF16 path — no scaler needed, this is the common modern default
for batch in dataloader:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"])
        loss = causal_lm_loss(logits, batch["input_ids"])  # Chapter 2, Lesson 1
    loss.backward()   # gradients computed with bf16 forward, no scaling required
    optimizer.step()
    optimizer.zero_grad()

# FP16 path — needs a GradScaler because of Section 2's underflow problem
scaler = torch.cuda.amp.GradScaler()
for batch in dataloader:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(batch["input_ids"])
        loss = causal_lm_loss(logits, batch["input_ids"])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

`torch.autocast` handles the "which ops run in which precision" decision from Section 4 automatically — it keeps numerically sensitive ops in fp32 and casts the rest, so you don't have to manually annotate every operation.

---

## 6. Diagnosis: Precision-Related Training Symptoms

- **Loss suddenly becomes `NaN` partway through fp16 training** → almost always a gradient overflow that the scaler didn't catch in time, or the scale factor is too aggressive; check `GradScaler`'s skip-step count — frequent skips mean the scale is oscillating too high.
- **Training is stable in bf16 but the exact same code produces `NaN` in fp16** → strong evidence it's specifically the exponent-range underflow/overflow issue from Section 2, not a bug in the model code — a good diagnostic differentiator to state in an interview.
- **Loss looks fine but the model's final quality is subtly worse than an fp32 baseline** → check whether a fp32 master weight copy is actually being maintained (Section 4); its absence is a common, easy-to-miss implementation bug in from-scratch training loops.

---

## Key Takeaways

- FP16's narrow exponent range causes gradient underflow; loss scaling (static or dynamic) exists specifically to counteract this.
- BF16 matches fp32's exponent range, avoiding the underflow problem structurally — this, not marketing, is why it's the modern default for LLM training.
- "Mixed precision" means selectively keeping certain operations (loss reduction, master weights, LayerNorm internals) in fp32 even while the bulk of matmuls run in 16-bit.
- A fp32 master weight copy exists because small updates can vanish when applied directly in 16-bit precision relative to the weight's magnitude.

---

## Self-Check Before Moving to Lesson 3

1. Explain, using the actual exponent-bit numbers, why bf16 doesn't need loss scaling but fp16 does.
2. What specific numerical failure does keeping a fp32 master weight copy prevent?
3. Training in fp16 produces occasional `NaN` losses; training the identical code in bf16 is stable. What does that difference tell you about the likely cause?