# Lesson 2.2 — Gradient Descent & Backpropagation
### Chapter 2: What Fine-Tuning Actually Does to a Model

---

## The Problem Story

Deepa had fine-tuned a model twice before her interview. She could describe the `TrainingArguments` config she used. But then the interviewer asked: "You set learning_rate=2e-4. Walk me through what actually happens to the model weights between step 1 and step 2 of training."

She went blank. She knew learning rate was a number you set. She did not know what it actually did to the numbers inside the model.

The interviewer pushed: "When you had a loss spike at step 800, what mechanically happened to cause that? And how does gradient accumulation prevent OOM without changing the math?"

Two questions. Both fundamental. Both unanswerable without understanding the optimization loop at the mechanical level.

This lesson gives you that mechanical understanding.

---

## The Concept

### The Optimization Loop in Full

Fine-tuning is gradient descent applied to a pre-trained model. The loop has four steps that repeat for every batch:

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: FORWARD PASS                                       │
│  Input batch → model → logits → loss                       │
│                                                             │
│  STEP 2: BACKWARD PASS (Backpropagation)                   │
│  loss → compute gradient of loss w.r.t. every weight       │
│                                                             │
│  STEP 3: OPTIMIZER STEP                                     │
│  Use gradients to update every weight                       │
│                                                             │
│  STEP 4: ZERO GRADIENTS                                     │
│  Clear gradient buffers for next batch                      │
└─────────────────────────────────────────────────────────────┘
```

Let us go through each step precisely.

---

### Step 1: Forward Pass

You already know this from Chapter 1. Input tokens flow through the model and produce logits. The loss is computed by comparing logits to the correct labels.

```python
outputs = model(input_ids, labels=labels)
loss = outputs.loss
```

At this point, PyTorch has recorded every computation in a computational graph — it remembers how each intermediate value was computed from the inputs and weights. This graph is what makes backpropagation possible.

---

### Step 2: Backward Pass — Backpropagation

**The core question:** How much does each weight in the model affect the loss?

Backpropagation answers this by computing the **gradient** of the loss with respect to every weight — using the chain rule from calculus.

**The gradient of a weight tells you:**
- Sign: does increasing this weight increase (+) or decrease (-) the loss?
- Magnitude: how sensitive is the loss to a small change in this weight?

```python
loss.backward()
# After this call, every parameter p has p.grad filled with its gradient
```

**The chain rule makes this work for deep networks:**

In a network with many layers, the gradient of the loss with respect to a weight in layer 1 is computed by multiplying gradients through every layer between layer 1 and the output. This is the chain rule applied recursively.

```
∂L/∂W₁ = (∂L/∂out_N) × (∂out_N/∂out_{N-1}) × ... × (∂out_2/∂out_1) × (∂out_1/∂W₁)
```

PyTorch's autograd engine does this automatically. The computational graph recorded during the forward pass is traversed in reverse, accumulating gradients at each node.

**What backpropagation actually computes for a transformer:**

For each attention weight matrix (Q, K, V, O projections), for each FFN weight (gate, up, down projections), for each layer norm scale, for each position in the embedding table that was used — backpropagation computes a gradient value. For a 7B model, this is 7 billion gradient values.

Those gradients are stored in `parameter.grad` — same shape as the parameter itself.

---

### Step 3: Optimizer Step — Updating Weights

The optimizer uses the gradients to update the weights. The simplest possible optimizer is **vanilla SGD (Stochastic Gradient Descent)**:

```
W_new = W_old - lr × gradient
```

This says: move each weight in the direction that decreases the loss, by an amount proportional to the learning rate.

**Why this works:**

The gradient points in the direction of steepest increase of the loss. Subtracting the gradient therefore moves the weight in the direction of steepest decrease. This is gradient descent.

---

### AdamW: The Standard Optimizer for Fine-Tuning

Vanilla SGD has problems in practice — it treats all weights equally and uses the raw gradient as the step size. AdamW (Adam with decoupled weight decay) fixes this with two key additions:

**1. Momentum (first moment — mean of gradients):**

```
m_t = β₁ × m_{t-1} + (1 - β₁) × gradient_t
```

`m_t` is a running average of past gradients. Instead of using the raw gradient, use the smoothed average. This dampens oscillations in the gradient and accelerates convergence in consistent directions.

- `β₁ = 0.9` (the standard) means 90% of the update comes from past gradients, 10% from the current gradient.

Think of it as: if you have been consistently seeing a gradient pointing in one direction, trust that direction more. If the gradient is jumping around, smooth it out.

**2. Adaptive learning rate (second moment — mean of squared gradients):**

```
v_t = β₂ × v_{t-1} + (1 - β₂) × gradient_t²
```

`v_t` is a running average of the squared gradient. This tracks how large the gradient has been historically for each weight.

The weight update then divides by the square root of this:

```
W_new = W_old - lr × m_t / (√v_t + ε)
```

**What this does:** Weights with consistently large gradients get a smaller effective learning rate. Weights with small gradients get a larger effective learning rate. This is adaptive learning rate — each weight gets its own learning rate automatically calibrated to its historical gradient magnitude.

**Why this matters for transformers:** Different parts of the model receive very different gradient magnitudes. Embedding weights might see tiny gradients while certain attention weights see large ones. AdamW lets each weight adapt at its own pace.

**3. Weight decay (the "W" in AdamW):**

```
W_new = (1 - lr × λ) × W_old - lr × m_t / (√v_t + ε)
```

Weight decay shrinks weights toward zero at each step by a factor λ. This is a regularization technique — it prevents weights from growing very large, which can help with generalization.

The "W" distinguishes AdamW from original Adam, which applied weight decay incorrectly by mixing it with the gradient update. AdamW decouples them, which turns out to matter in practice for language model fine-tuning.

**Memory cost of AdamW:**

AdamW stores, for every parameter:
- The parameter itself (e.g., fp16 or bf16): 2 bytes per value
- The gradient: same size as parameter
- The first moment (m_t): same size as parameter, fp32
- The second moment (v_t): same size as parameter, fp32

Total: roughly 16 bytes per parameter for fp16/bf16 training with fp32 optimizer states.

For a 7B model: 7 × 10⁹ × 16 bytes = 112 GB. This is why full fine-tuning of 7B models requires multiple high-end GPUs, and why 8-bit Adam and PEFT methods exist.

---

### Learning Rate: The Most Important Hyperparameter

The learning rate determines how large each step is:

```
W_new = W_old - lr × gradient_direction
```

**Too high a learning rate:**
- Large weight updates
- The loss can overshoot the minimum and diverge
- In transformers: loss spikes, NaN gradients
- Catastrophic forgetting of pre-trained knowledge (huge updates overwrite learned representations)

**Too low a learning rate:**
- Very small weight updates
- Training converges extremely slowly
- May get stuck in flat regions and never converge meaningfully
- May not deviate enough from the pre-trained model to adapt to your task

**For fine-tuning specifically, learning rate must be lower than for pre-training:**

Pre-training starts from random weights and needs large steps to reach a useful solution quickly. Fine-tuning starts from a good solution (the pre-trained model) and needs small steps to nudge the model toward the task without destroying what it already knows.

```
Pre-training LR:       1e-4 to 1e-3
Full fine-tuning LR:   1e-5 to 5e-5
LoRA fine-tuning LR:   1e-4 to 5e-4  (can be higher because only small adapters are updated)
```

Why can LoRA use higher LR? Because the base model weights are frozen — only the small LoRA adapter weights change. The frozen base model cannot be damaged by a large LoRA update.

---

### Gradient Accumulation: Simulating Large Batches on Small GPUs

**The problem:**

Large batch sizes improve training stability and gradient quality. But the batch must fit in GPU memory. For a 7B model, even a batch size of 1 with long sequences can fill a 16GB GPU.

**The solution — gradient accumulation:**

Instead of doing one optimizer step per batch, accumulate gradients over several mini-batches before doing one optimizer step:

```python
accumulation_steps = 8  # accumulate over 8 mini-batches

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # normalize the loss
    loss.backward()  # accumulate gradients (do NOT zero them yet)

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()   # update weights once
        optimizer.zero_grad()  # clear gradients
```

**The math:**

With batch_size=4 and accumulation_steps=8:
- Effective batch size = 4 × 8 = 32
- The gradient after 8 mini-batches is the same as the gradient from one batch of 32 examples (because gradients are additive)

The model sees the same total gradient signal. The optimizer step happens once every 32 examples. This is mathematically equivalent to using a batch of 32.

**Why you divide loss by accumulation_steps:**

Without division, each mini-batch contributes a full loss value, and after 8 steps the accumulated gradient is 8× larger than it should be for one batch of 32. Dividing normalizes so that the gradient scale matches what you would get from a true batch of 32.

**What gradient accumulation does NOT give you:**

Batch normalization statistics (not relevant for transformer fine-tuning, which uses Layer Norm). That is the only thing gradient accumulation cannot simulate. For transformers, gradient accumulation is a perfect substitute for larger batch sizes.

---

### Mixed Precision Training: fp32, fp16, bf16

**Why this matters:**

By default, PyTorch uses 32-bit floating point (fp32) for all computations and stored values. Switching to 16-bit formats cuts memory in half and speeds up computation on modern GPUs.

**The three formats:**

| Format | Bits | Exponent | Mantissa | Max Value | Notes |
|--------|------|----------|----------|-----------|-------|
| fp32 | 32 | 8 bits | 23 bits | ~3.4 × 10³⁸ | Full precision, default |
| fp16 | 16 | 5 bits | 10 bits | 65,504 | Limited range — overflow risk |
| bf16 | 16 | 8 bits | 7 bits | ~3.4 × 10³⁸ | Same range as fp32, less precision |

**fp16 problem:**

fp16 has a maximum value of 65,504. Gradients and activations in large transformer models can exceed this, causing overflow to NaN — a training crash. This is called **gradient overflow** and is why fp16 training requires loss scaling (a technique that multiplies the loss before backward pass to keep gradients in range, then divides before the optimizer step).

**bf16 advantage:**

bf16 has the same exponent range as fp32 (8 bits) but less mantissa precision (7 bits vs 23). This means it can represent the same range of values as fp32 — no overflow risk. The reduced mantissa precision is generally acceptable for the kinds of computations in transformer forward/backward passes.

**Recommendation for modern training:**
- Use bf16 if your hardware supports it (A100, H100, Ampere-generation or newer GPUs, some Intel Arc GPUs)
- Use fp16 with loss scaling if bf16 is not available
- Intel GPUs (like yours) may have specific support considerations — check `torch.cuda.is_bf16_supported()`

**How mixed precision works:**

The word "mixed" refers to keeping some things in fp32 while computing in fp16/bf16:
- Model weights: stored in fp16/bf16 (memory savings)
- Gradient computation: in fp16/bf16 (speed improvement)
- Optimizer states (m_t, v_t in AdamW): kept in fp32 (precision for accumulation)
- Master copy of weights: fp32 for optimizer updates, then cast back to fp16/bf16

PyTorch handles this automatically with `torch.autocast`:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(**batch)
    loss = outputs.loss
loss.backward()
optimizer.step()
```

---

### Gradient Clipping: Preventing Explosions

Even with good learning rates, gradients can spike suddenly — caused by an unusual batch, a numerical instability, or early training instability. These spikes cause large weight updates that can destroy training.

**Gradient clipping** caps the gradient norm at a maximum value before the optimizer step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the gradient norm exceeds `max_norm`, all gradients are scaled down proportionally so the total norm equals `max_norm`. The direction of the gradient is preserved; only the magnitude is capped.

```
gradient_norm = √(Σ gradient²)   # L2 norm across all parameters

if gradient_norm > max_norm:
    scale = max_norm / gradient_norm
    for each gradient:
        gradient = gradient × scale
```

**Standard value:** `max_norm=1.0` is the most common default. Some implementations use `max_norm=0.5` for more conservative clipping.

Logging the gradient norm at each step is highly valuable — it tells you when training is unstable before the loss curve shows it. We cover this in Chapter 8.

---

### Step 4: Zero Gradients

PyTorch accumulates gradients by default — calling `loss.backward()` adds to existing `.grad` values rather than replacing them. You must explicitly clear gradients after each optimizer step:

```python
optimizer.zero_grad()
```

Forgetting this is a common bug. If you do not zero gradients, each step adds the gradient of the current batch to gradients from all previous batches. The optimizer step then uses a gradient that is the sum of many batches, causing wild and incorrect weight updates.

The only time you intentionally do not zero gradients between backward calls is during gradient accumulation (described above).

---

## The Intuition Bridge

**The loss landscape:**

Imagine the model's weights as a position in a very high-dimensional space (billions of dimensions, one per weight). The loss function defines a "landscape" over this space — hills are high loss, valleys are low loss.

Gradient descent is like hiking in this landscape in the dark, only able to feel the local slope. At each step, you measure the slope at your feet (the gradient) and take a step downhill (in the direction of negative gradient). The learning rate is the step size.

**Too large a step:** You overshoot valleys and bounce between walls.
**Too small a step:** You take forever to get anywhere meaningful.

For fine-tuning, you are not starting in random flat terrain (as in pre-training). You are starting in a valley that represents "good language model" and trying to walk to a nearby, slightly lower valley that represents "good language model for your task." You need small, careful steps so you do not accidentally walk out of the existing valley and lose your pre-trained starting point.

**Gradient accumulation as virtual batch:**

Imagine you want to estimate the average slope of a hill. You can either take 32 measurements all at once, or take 8 groups of 4 measurements and average all 32. The estimate of the slope is the same either way. Gradient accumulation is the second approach — same answer, less memory per measurement.

---

## Why This Matters for Fine-Tuning

**Your learning rate choice determines whether fine-tuning works at all.** Too high and you overwrite pre-trained knowledge. Too low and the model never adapts to your task. We cover systematic LR selection in Chapter 7.

**Gradient accumulation is not optional when your GPU is small.** With a 16GB GPU and a 7B model quantized to 4-bit, you might fit batch_size=1. Effective batch size 1 gives extremely noisy gradients. Gradient accumulation steps of 8–16 gives you effective batch size 8–16, which is far more stable.

**Mixed precision is almost always the right choice.** The memory and speed benefits are significant, and the quality difference is negligible for fine-tuning. Not using it on a GPU with bf16 support is leaving performance on the table.

**Gradient clipping is a safety net, not a crutch.** If your gradients are consistently hitting the clip threshold, that is a sign your learning rate is too high. Clipping prevents crashes but does not fix the underlying instability.

---

## The Code

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # fp32 for clarity in this lesson
    device_map="auto"
)

# ── 1. Manual training step ─────────────────────────────────────

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

text = "The capital of France is Paris."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Save weights BEFORE update (one layer as example)
weight_before = model.model.layers[0].self_attn.q_proj.weight.data.clone()

# Forward pass
model.train()
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
print(f"Loss before step: {loss.item():.6f}")

# Backward pass
optimizer.zero_grad()
loss.backward()

# Inspect gradients
q_proj_grad = model.model.layers[0].self_attn.q_proj.weight.grad
print(f"\nGradient stats for q_proj (layer 0):")
print(f"  Shape:    {q_proj_grad.shape}")
print(f"  Mean:     {q_proj_grad.mean().item():.8f}")
print(f"  Std:      {q_proj_grad.std().item():.8f}")
print(f"  Max abs:  {q_proj_grad.abs().max().item():.8f}")
print(f"  Nonzero:  {(q_proj_grad != 0).sum().item()} / {q_proj_grad.numel()}")

# Gradient norm before clipping
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"\nGradient norm before clip: {total_norm.item():.4f}")
print(f"(If > 1.0, gradients were clipped)")

# Optimizer step
optimizer.step()

# Compare weights after
weight_after = model.model.layers[0].self_attn.q_proj.weight.data.clone()
weight_diff = (weight_after - weight_before).abs()
print(f"\nWeight change stats (q_proj, layer 0):")
print(f"  Mean change:   {weight_diff.mean().item():.10f}")
print(f"  Max change:    {weight_diff.max().item():.10f}")
print(f"  Relative change: {(weight_diff.mean() / weight_before.abs().mean()).item():.8f}")

# ── 2. Gradient accumulation ────────────────────────────────────

print("\n── Gradient Accumulation ──")
accumulation_steps = 4

# Simulate 4 mini-batches
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
    "Transformers use self-attention mechanisms.",
    "Fine-tuning adapts pre-trained models to specific tasks.",
]

optimizer.zero_grad()
accumulated_loss = 0.0

for i, t in enumerate(texts):
    inp = tokenizer(t, return_tensors="pt",
                    padding=True, truncation=True).to(model.device)
    out = model(**inp, labels=inp["input_ids"])
    loss = out.loss / accumulation_steps  # normalize
    loss.backward()
    accumulated_loss += loss.item()
    print(f"  Mini-batch {i+1}: loss={out.loss.item():.4f} (normalized: {loss.item():.4f})")

print(f"  Accumulated loss sum: {accumulated_loss:.4f}")
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
print(f"  Optimizer step done after {accumulation_steps} mini-batches.")

# ── 3. Compare AdamW optimizer state ────────────────────────────

print("\n── AdamW Internal State ──")
# After at least one step, optimizer has state
for group in optimizer.param_groups:
    for p in list(group["params"])[:1]:  # just first param for demo
        if p in optimizer.state:
            state = optimizer.state[p]
            print(f"  step:           {state['step']}")
            print(f"  exp_avg (m_t) shape:    {state['exp_avg'].shape}")
            print(f"  exp_avg_sq (v_t) shape: {state['exp_avg_sq'].shape}")
            print(f"  exp_avg mean:   {state['exp_avg'].mean().item():.8f}")
            break

# ── 4. Mixed precision example ──────────────────────────────────

print("\n── Mixed Precision Training ──")
# Check bf16 support
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"BF16 supported: {bf16_supported}")

dtype = torch.bfloat16 if bf16_supported else torch.float16
print(f"Using dtype: {dtype}")

# Mixed precision forward pass
inp_mp = tokenizer("Testing mixed precision training.", return_tensors="pt").to(model.device)
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    out_mp = model(**inp_mp, labels=inp_mp["input_ids"])
    print(f"Loss (mixed precision): {out_mp.loss.item():.6f}")

# ── 5. Verify gradients exist on all parameters ─────────────────

print("\n── Parameter Gradient Check ──")
model.zero_grad()
inp_check = tokenizer("Verify gradients flow.", return_tensors="pt").to(model.device)
out_check = model(**inp_check, labels=inp_check["input_ids"])
out_check.loss.backward()

has_grad = 0
no_grad = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        has_grad += 1
    else:
        no_grad += 1

print(f"Parameters with gradients:    {has_grad}")
print(f"Parameters without gradients: {no_grad}")
print(f"(Parameters without grads are buffers or not in computational graph)")
```

---

## The Experiment

**EXP-2.2.A — Learning Rate Sensitivity**

Goal: Observe how different learning rates change the weight update magnitude and training stability.

Run 3 single gradient steps with the same batch, using lr = 1e-6, 1e-4, 1e-2. For each:
- Record the loss before the step
- Record the weight change magnitude (mean absolute difference in q_proj weights)
- Record the gradient norm

```
════════════════════════════════════════════════════════
EXPERIMENT LOG
════════════════════════════════════════════════════════
ID:       EXP-2.2.A
Lesson:   2.2 — Gradient Descent & Backpropagation
Goal:     Observe how learning rate affects weight update
          magnitude and what "too high" looks like in numbers

SETUP
Model: [your model]
Text: [one fixed sentence, same for all LR values]
LRs tested: 1e-6, 1e-4, 1e-2

RAW OBSERVATIONS
LR=1e-6:  weight_change_mean=___, gradient_norm=___
LR=1e-4:  weight_change_mean=___, gradient_norm=___
LR=1e-2:  weight_change_mean=___, gradient_norm=___

[Also: does the loss go down after the 1e-2 step? Run 10 more
 steps with 1e-2 and see what happens to the loss trajectory]

WHAT SURPRISED ME
[Fill: was the weight change at 1e-6 visible at all?]
[Fill: did 1e-2 cause the loss to spike or diverge?]

INTERPRETATION
[At what LR does the weight change become "too large"
 relative to the original weight magnitude?]
[What does the ratio weight_change/weight_magnitude tell you?]

IMPLICATIONS FOR FINE-TUNING
[Why does fine-tuning use LRs much lower than 1e-3?]
[At what LR would you expect catastrophic forgetting?]

OPEN QUESTIONS
[Fill]

NEXT STEP
[Fill]
════════════════════════════════════════════════════════
```

---

## Interview Checkpoint

**Q: Walk me through what happens to the model weights between step 1 and step 2 of training.**

> A: In step 1, we do a forward pass — input tokens flow through the model, producing logits. We compute cross-entropy loss between predicted and actual next tokens. Then the backward pass runs: PyTorch traverses the computational graph in reverse and computes the gradient of the loss with respect to every trainable weight using the chain rule. These gradients are stored in each parameter's `.grad` attribute. The optimizer then reads those gradients and applies the update rule — for AdamW, it updates running averages of the gradient and its square, then computes an adaptive step size for each weight, applies weight decay, and updates each weight. Finally we zero the gradients so they don't accumulate into step 2.

**Q: How does gradient accumulation work and why doesn't it change the math?**

> A: Gradient accumulation splits one large effective batch into several mini-batches. For each mini-batch, we do a forward and backward pass but divide the loss by the number of accumulation steps before calling backward. This means each mini-batch contributes a proportionally scaled gradient. Because gradients are additive in PyTorch (they accumulate in `.grad` until zeroed), after all accumulation steps the `.grad` contains the sum of all mini-batch gradients — which, due to the division, is mathematically equivalent to the gradient from one large batch. The optimizer step then runs once, just as it would for the large batch. The only approximation is that batch normalization statistics would differ, but transformers use layer norm which is computed per example, so there's no difference.

**Q: Why is bf16 preferred over fp16 for fine-tuning transformers?**

> A: fp16 has only 5 exponent bits, limiting its representable range to about ±65,504. Large transformer models can produce gradients and activations that overflow this range, causing NaN values and training crashes. bf16 has 8 exponent bits — the same as fp32 — giving it the same numerical range while using only 16 bits. The tradeoff is reduced mantissa precision (7 bits vs 23 in fp32), but for the types of computations in transformer training, this precision loss is acceptable and rarely causes problems. bf16 is therefore a drop-in replacement for fp32 that halves memory usage without the overflow risk of fp16.

---

## Common Mistakes & Misconceptions

❌ **"Gradient accumulation is a workaround, not real training."**
Gradient accumulation is mathematically equivalent to training with the larger batch size. It is not an approximation (for transformers). Many production training runs use it deliberately. It is a standard tool, not a compromise.

❌ **"Forgetting optimizer.zero_grad() is a minor bug."**
It is a catastrophic bug. Without zeroing gradients, each step accumulates the gradients of all previous steps. By step 10, your optimizer is using the sum of 10 batches' worth of gradients, scaled as if it were one batch's gradients. The weight updates will be wildly incorrect, typically causing loss divergence within a few steps.

❌ **"Lower learning rate is always safer."**
Very low learning rates (1e-8, 1e-7) may cause the model to never converge in a reasonable time. They can also cause the optimizer to get stuck in flat regions of the loss landscape. There is a minimum useful learning rate below which you are effectively doing nothing.

❌ **"Gradient clipping changes the direction of the gradient."**
Gradient clipping scales all gradients proportionally so the total norm equals max_norm. It changes the magnitude but not the direction. The optimizer still moves in the correct direction, just with a smaller step when gradients are very large.

❌ **"Mixed precision means fp16 everywhere."**
Mixed precision specifically means optimizer states (the running averages in AdamW) are kept in fp32 for precision, while forward and backward computation happens in fp16 or bf16. The "mixed" refers to this combination, not to some parameters being fp32 and others fp16.