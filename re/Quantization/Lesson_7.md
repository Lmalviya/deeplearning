# Quantization Lesson 7 — NF4: NormalFloat 4-bit and the QLoRA Format

---

## Recap: Why Uniform INT4 Wastes Precision on Neural Network Weights

Signed INT4 represents 16 evenly-spaced values:
```
-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
```
Every representable value sits the same "distance" from its neighbors — the grid doesn't care what the data actually looks like.

But pretrained neural network weights are not uniformly distributed. Sampling a real weight tensor shows a **near-Gaussian distribution centered at zero**: most values cluster tightly around zero, with only a small fraction taking on large magnitudes. This isn't a coincidence — weight initialization is Gaussian, and gradient-based training tends to preserve roughly that shape.

In a Gaussian distribution, ~68% of values sit within 1 standard deviation of zero, ~95% within 2 standard deviations, and only ~5% live out in the tails. INT4's *equal* spacing allocates **half its 16 codes to the sparse tails** and only half to the dense center where 95% of the actual data lives — a large fraction of the grid's resolution is spent on almost nothing.

This motivates the obvious fix: **don't space the codes evenly — space them to match where the data actually is.** That's quantile quantization, and it's where the real design problem of this lesson begins.

---

## The Real Problem: Why Not Just Use Each Tensor's Own Quantiles?

The "obviously correct" approach is: for each weight tensor, compute its own empirical quantiles, and place the 16 quantization levels there — each bin then holds an equal share of that tensor's actual values, which is provably the error-minimizing placement for a fixed bit budget.

Here's why this "obviously correct" approach is not what's actually done, and it's worth sitting with the problem before seeing the fix:

**Exact quantile estimation is expensive, and it has to be redone for every single tensor.** Finding the true quantiles of a tensor's values means something like sorting or rank-estimating millions of numbers. A large model has thousands of weight tensors, each with a *different* raw distribution (different scale, different exact shape of the histogram) — so this expensive computation can't be done once and reused; it would need to be repeated per tensor, which is computationally impractical at scale.

**So in practice, this cost is dodged with fast *approximate* quantile-estimation algorithms** (e.g. SRAM quantiles) instead of exact sorting. That solves the speed problem, but introduces a new one: approximation error. And critically, that error is **worst exactly at the tails** — the rare, large-magnitude values — because approximate algorithms are tuned to get the bulk of the distribution roughly right at the expense of precision in the sparse extremes.

**This is a real problem, not a minor one, because tail values (outliers) are often the most functionally important weights** in a network — they can encode the sharpest, most decisive parts of what a layer computes. So naive per-tensor quantile quantization ends up expensive *and* least accurate exactly where accuracy matters most. That combination is what NF4 is designed to avoid.

---

## The Key Trick: Fix the Distribution's Shape, Let Only a Scale Constant Vary

Here's the insight that resolves both problems at once, and it's the conceptual heart of NF4.

If every weight tensor had some *arbitrary, unpredictable* distribution shape, there would be no way around estimating quantiles per tensor. But the QLoRA paper makes (and empirically verifies, in its Appendix F) a much stronger claim: **pretrained weights are consistently zero-centered and Gaussian in shape — tensors differ from each other mainly by their standard deviation (how spread out they are), not by the fundamental shape of their distribution.**

If that's true — if every tensor is "the same distribution, just scaled differently" — then you never need to estimate quantiles from real data at all:

1. **Compute the quantiles of one canonical distribution — the standard normal N(0,1) — exactly, once, analytically**, using its known inverse CDF. This is not an approximation of anything: it's a closed-form calculation on a known theoretical distribution, so there's no sorting, no sampling, and no approximation error anywhere in this step — including at the tails.
2. **For any real weight block, normalize it by its own spread** (in practice, its `absmax` — the largest absolute value in that block, called the **quantization constant**) so its values line up with that same standard shape.
3. **Reuse the identical 16 quantile boundaries for every block, in every layer, forever.** The expensive part (finding good quantile boundaries) is done exactly once for the whole model — never again, no matter how many tensors or how large the model.

| | Arbitrary / unknown distribution per tensor | Fixed-shape distribution (up to a scale) |
|---|---|---|
| Quantile source | Estimated per tensor from real data | Computed once from theory (inverse CDF) |
| Cost | Expensive, repeated for every tensor | Cheap — computed a single time, ever |
| Accuracy at outliers | Degraded by approximation error | Exact — no approximation involved |
| What must be stored per tensor/block | Nothing extra beyond the estimation itself | One scalar: the quantization constant |

This is also exactly *why* the per-block quantization constant exists in NF4 — it isn't only there to rescale dequantized values back to their real magnitude. It's the single number that lets **one universal, precomputed table** correctly serve blocks with wildly different raw magnitudes, without ever having to look at each block's actual empirical distribution.

---

## Information Theory View: Quantile Quantization Is Optimal — Now That the Distribution Is Known

From information theory, the error-minimizing quantization for a *known* distribution is quantile quantization: place boundaries so each bin captures an equal share of the probability mass. For a Gaussian, this means levels packed densely near zero and sparse in the tails — the opposite layout from INT4's equal spacing.

```python
import scipy.stats as stats
import numpy as np

def compute_nf4_raw_quantiles() -> np.ndarray:
    """
    The 16 quantile levels of a standard normal distribution — the values
    that would form NF4's grid before the exact-zero adjustment below.
    """
    quantile_levels = np.array([(i + 0.5) / 16 for i in range(16)])
    values = stats.norm.ppf(quantile_levels)   # inverse CDF — closed form, no data needed
    return values

print(np.round(compute_nf4_raw_quantiles(), 4))
# [-1.9673 -1.4741 -1.1477 -0.9013 -0.6893 -0.4994 -0.3192 -0.1574
#   0.1574  0.3192  0.4994  0.6893  0.9013  1.1477  1.4741  1.9673]
```

```
INT4 (equal spacing):
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
-8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7

NF4 (quantile spacing - dense near zero, sparse in the tails):
|  |  |  | |  | | ||   ||  | |  |  |  |
-8        -4   -2  -1  0  1  2   4       8

Gaussian density (what the data actually looks like):
                    ████████████
                  ██████████████████
               ██████████████████████████
           ██████████████████████████████████
     ████████████████████████████████████████████
|--------------------------------------------------|
-8                     0                        8
```
INT4 wastes representable values in the sparsely populated tails; NF4 concentrates them exactly where the (known, assumed) distribution says the data actually lives.

---

## Constructing the Exact NF4 Table

The raw quantiles above have one issue for a *weight* format: none of them land exactly on `0.0`. That matters more than it might seem — a large fraction of trained weights are extremely close to zero, and having no code that represents exact zero would force every near-zero weight into a small, non-zero rounding error, introducing a small systematic bias across the entire model.

The fix used in NF4 (and implemented in `bitsandbytes`) is to build the table in two asymmetric halves instead of one symmetric set of 16 quantiles:
- Compute `2^(k-1)` quantiles for the **negative** half of N(0,1).
- Compute `2^(k-1) + 1` quantiles for the **positive** half.
- Merge the two halves and drop the one duplicate zero this produces, leaving exactly 16 unique values — 8 on one side (including a boundary exactly at 0) and 7 on the other, plus that shared zero.

This is why NF4's table looks asymmetric rather than a clean mirror image — it's a deliberate construction to guarantee an exact-zero code exists, not an approximation or a rounding artifact.

```python
def construct_nf4_lookup_table() -> np.ndarray:
    """
    The exact NF4 lookup table as implemented in bitsandbytes / used by QLoRA.
    Fixed for every NF4 tensor in every model - computed once, reused forever.
    """
    nf4_values = np.array([
        -1.0,                     # index 0  (most negative)
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,                      # index 7  (exact zero - guaranteed by construction)
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0                       # index 15 (most positive)
    ])
    return nf4_values

NF4_LOOKUP = construct_nf4_lookup_table()
```
These 16 numbers never change. They are not re-derived per model, per layer, or per block - they're a fixed constant of the format itself, exactly because of the "fix the distribution shape, vary only the scale" trick from earlier.

---

## Storage vs. Compute: Why NF4 Needs an Actual Lookup Table

It's worth being explicit about something easy to gloss over: an NF4 weight isn't a compressed float — it's a **4-bit index (0-15) into the table above**, plus one shared scale per block. The real floating-point value only exists transiently, reconstructed at the moment it's needed for a computation, then discarded.

This is different from uniform formats (like plain INT4 or the INT8 scheme from LLM.int8()), where evenly-spaced levels mean `code -> value` is a simple formula (`value = code x scale`) — no table required, just one multiply. NF4's levels are **irregularly spaced** (they're quantiles of a Gaussian, not arithmetic steps), so there is no formula connecting "code 6" to its value — the only way to recover it is to store and index into the 16 numbers directly. That's the direct, structural reason NF4 needs a lookup table where a uniform format doesn't.

And because there's no hardware that does arithmetic natively on an arbitrary 16-entry lookup table, NF4 is a **storage-only** datatype: weights sit in GPU memory as 4-bit codes, and right before any matrix multiplication, the relevant block is dequantized — table lookup, then multiply by the block's scale — into BF16, where the actual compute happens. The dequantized BF16 tile is discarded immediately after use; only the codes and the scale persist.

---

## NF4 Quantization and Dequantization Mechanics

```python
def nf4_quantize(weights: np.ndarray, group_size: int = 64) -> dict:
    """Quantize a weight tensor to NF4 codes, block by block."""
    weights_flat = weights.reshape(-1)
    n_groups = len(weights_flat) // group_size

    quantized_indices = np.zeros(len(weights_flat), dtype=np.uint8)
    absmax_per_group = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start, end = g * group_size, (g + 1) * group_size
        group = weights_flat[start:end]

        # The quantization constant for this block
        absmax = np.max(np.abs(group))
        absmax_per_group[g] = absmax

        # Normalize onto the same standard scale NF4_LOOKUP was built from
        normalized = group / absmax if absmax > 0 else group

        # Match each normalized value to its nearest fixed table entry
        for i, w_norm in enumerate(normalized):
            quantized_indices[start + i] = np.argmin(np.abs(NF4_LOOKUP - w_norm))

    return {"indices": quantized_indices.reshape(weights.shape),
            "absmax": absmax_per_group, "group_size": group_size}


def nf4_dequantize(quantized: dict) -> np.ndarray:
    """Reconstruct float values from NF4 codes - table lookup, then rescale."""
    indices_flat = quantized["indices"].reshape(-1)
    absmax, group_size = quantized["absmax"], quantized["group_size"]
    weights_flat = np.zeros(len(indices_flat), dtype=np.float32)

    for g in range(len(absmax)):
        start, end = g * group_size, (g + 1) * group_size
        weights_flat[start:end] = NF4_LOOKUP[indices_flat[start:end]] * absmax[g]

    return weights_flat.reshape(quantized["indices"].shape)
```

### A small worked example, by hand

Take a tiny block of 4 weights, `[0.42, -0.05, 0.91, -0.30]`:

```
absmax = max(|0.42|, |-0.05|, |0.91|, |-0.30|) = 0.91      <- the quantization constant

Normalized (divide by 0.91):
  0.42 / 0.91 = 0.462
 -0.05 / 0.91 = -0.055
  0.91 / 0.91 = 1.000
 -0.30 / 0.91 = -0.330

Match each to the nearest NF4_LOOKUP entry:
  0.462  -> nearest is 0.4407  -> code 12
 -0.055  -> nearest is -0.0911 or 0.0    -> 0.0 is closer -> code 7
  1.000  -> exact match             -> code 15
 -0.330  -> nearest is -0.2844 or -0.3949 -> -0.2844 is closer -> code 4

Stored on disk:  codes = [12, 7, 15, 4],  scale = 0.91   (that's it - no floats stored)

Dequantize:
  code 12 -> 0.4407 x 0.91 = 0.401   (original: 0.42, error 0.019)
  code 7  -> 0.0    x 0.91 = 0.000   (original: -0.05, error 0.05)
  code 15 -> 1.0    x 0.91 = 0.91    (original: 0.91, error 0)
  code 4  -> -0.2844 x 0.91 = -0.259 (original: -0.30, error 0.041)
```
Notice the largest weight in the block (0.91, the outlier of this tiny group) is reconstructed *exactly* — it always lands on the table's boundary value +-1.0 after normalization. This isn't a coincidence: it's the direct benefit of the whole "fix the distribution, vary the scale" design — the scale is defined by the block's own max, so the block's most extreme value is always representable, while the quantile spacing gives the best achievable precision to everything else, weighted by where the data is expected to concentrate.

### Quality of NF4 vs. INT4

On typical weight data, NF4 measurably beats plain uniform INT4 at the same bit-width:
```
NF4 RMSE:  0.00041
INT4 RMSE: 0.00058
-> NF4 is roughly 30% more accurate than INT4 for normally distributed weights
```
This gap is the direct, measurable payoff of matching the grid to the data's actual shape instead of spacing it uniformly.

---

## Double Quantization: Quantizing the Quantization Constants

NF4 alone still needs one FP32 `absmax` per block of 64 weights. For a 7B model with `group_size=64`, that's `7B / 64 ~= 110M` scale factors x 4 bytes ~= **440 MB** of overhead — not huge relative to the model, but not free either.

**Double quantization** compresses these scale factors themselves, by quantizing them a second time:
```
Level 1:  weights      -> NF4 codes         (using absmax_1, one per block of 64)
Level 2:  absmax_1     -> INT8              (using absmax_2, one per group of 256 absmax_1 values)
```

```python
def double_quantize(weights: np.ndarray, inner_group_size: int = 64, outer_group_size: int = 256) -> dict:
    level1 = nf4_quantize(weights, group_size=inner_group_size)
    absmax_1 = level1["absmax"]                      # FP32, one per 64 weights

    n_outer = len(absmax_1) // outer_group_size
    absmax_1_int8 = np.zeros_like(absmax_1, dtype=np.int8)
    absmax_2 = np.zeros(n_outer, dtype=np.float32)    # FP32, one per 256 absmax_1 values - much smaller

    for g in range(n_outer):
        start, end = g * outer_group_size, (g + 1) * outer_group_size
        group = absmax_1[start:end]
        max_val = np.max(np.abs(group))
        absmax_2[g] = max_val
        absmax_1_int8[start:end] = np.round(group / (max_val / 127)).clip(-127, 127)

    return {"nf4_indices": level1["indices"], "absmax_1_int8": absmax_1_int8,
            "absmax_2": absmax_2, "inner_group_size": inner_group_size, "outer_group_size": outer_group_size}
```

| Component | Without DQ | With DQ |
|---|---|---|
| NF4 codes (4-bit) | 3.5 GB | 3.5 GB |
| First-level scales | 440 MB (FP32) | 110 MB (INT8) |
| Second-level scales | -- | 1.7 MB (FP32) |
| **Total overhead** | **440 MB** | **~112 MB** |

For a 7B model this saves ~328 MB; for a 70B model (4.4 GB -> ~1.1 GB of overhead), the saving becomes substantial. The QLoRA paper reports this adds back roughly 0.5 bits/parameter on average — so NF4 + double quantization is effectively **~4.5 bits per parameter**, not a clean 4.

---

## QLoRA: Putting NF4 Into Practice for Fine-Tuning

NF4 wasn't built for inference — it exists specifically to make **fine-tuning** large models feasible on consumer hardware (Dettmers et al., 2023).

### The problem QLoRA solves

Full fine-tuning a 7B model in FP16 needs weights (14 GB) + gradients (14 GB) + Adam optimizer states (28 GB, two moments) + activations (2-4 GB) — **~60 GB**, requiring multiple A100s. QLoRA's fix: freeze the base model in NF4 (3.5 GB), add small trainable LoRA adapters in BF16, and only compute gradients/optimizer state for those tiny adapters — landing around **6-8 GB**, which fits on a single consumer 24 GB GPU.

### Why the base model has to stay frozen - not just "for memory"

There's a deeper reason NF4 weights are frozen during QLoRA training, beyond just saving memory: **they can't meaningfully be trained via ordinary backpropagation in the first place.**

Turning a continuous weight into an NF4 code is a round-to-nearest-table-entry operation — a **step function**. Nudge the underlying float slightly, and almost always the chosen code doesn't change at all; on the rare occasions it does, the output jumps discontinuously rather than shifting smoothly. The derivative of a step function is zero almost everywhere and undefined at the jumps — so `dLoss/dW` is simply not a usable training signal if `W` is NF4-quantized.

QLoRA's answer isn't a clever workaround for this (like a Straight-Through Estimator) — it sidesteps the problem entirely: **freeze `W` completely, and never compute a gradient of it.** Gradients still need to flow *through* the frozen, dequantized weights via the chain rule, to reach the LoRA adapters attached upstream — and that part is fine, because for a fixed set of codes, dequantization (table lookup x scale) behaves as a constant linear map with respect to the layer's input activations. The only thing that was ever non-differentiable is the *rounding* step used to produce the codes — and since QLoRA never updates the codes, it never needs to differentiate that operation at all. All trainable capacity instead lives in the separate LoRA path (`A`, `B`), which is never quantized and has no such problem.

```
                    +------------------------------------+
                    |     Frozen NF4 Base Model           |
Input -> Tokenizer -> |  W_NF4 (4-bit codes + scale)      | ->
                    |   -> dequantize (lookup x scale) -> BF16, transient, per matmul |
                    +--------------+---------------------+
                                   |
                    +--------------+-------------------+
                    |     LoRA Adapters (BF16, trainable) |
                    |   W_LoRA = W_base + alpha(BA)       |
                    |   A: [r x in_dim], B: [out_dim x r] |
                    +---------------------------------+
                                   |
                                Output
```
Only `A` and `B` ever receive a gradient update. `W`'s codes and scale constants never change across the entire training run.

### Paged optimizers

Long sequences combined with gradient checkpointing can cause transient GPU memory spikes large enough to trigger out-of-memory crashes, even when average memory usage looks fine. QLoRA uses NVIDIA's unified memory feature to automatically page optimizer states out to CPU RAM during these spikes and back when needed — transparent to the training loop:
```python
from bitsandbytes.optim import PagedAdamW8bit

optimizer = PagedAdamW8bit(
    model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01
)
```

---

## QLoRA Fine-Tuning in Practice

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

# 1. Load the base model in NF4 with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             # NormalFloat, not plain INT4
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # dequantize to BF16 for every matmul
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Prepare for k-bit training (enables gradient checkpointing, disables
#    incompatible ops around the frozen quantized layers)
model = prepare_model_for_kbit_training(model)

# 3. Attach LoRA adapters - on ALL linear layers, not just attention
#    (needed to match 16-bit performance)
lora_config = LoraConfig(
    r=64, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}% of total)")
# Trainable: 167,772,160 (2.39% of total) - only the LoRA adapters

# 4. Train
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llama-2-7b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=training_args, train_dataset=your_dataset, tokenizer=tokenizer)
trainer.train()

# 5. Save only the adapters - the frozen NF4 base is never modified
model.save_pretrained("./llama-2-7b-lora-adapters")

# 6. Reload for inference: base in NF4 + adapters on top
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./llama-2-7b-lora-adapters")
```

---

## Memory Comparison

```
Task: Fine-tune Llama-2 7B

Full fine-tuning (FP16):        ~60 GB   (weights 14 + grads 14 + Adam 28 + activ. 4) -> 2x A100 80GB
LoRA fine-tuning (FP16 base):   ~19 GB   (weights 14 + tiny adapter grads/optim + activ. 4) -> 1x A100 80GB
QLoRA (NF4 base + LoRA):        ~7.5 GB  (weights 3.5 + DQ scales 0.1 + adapter grads/optim 0.6 + activ. 3) -> 1x RTX 3090 24GB
```
This is the number that mattered: QLoRA made it possible to fine-tune a 7B model on a single consumer GPU — before this, fine-tuning anything above ~1B parameters generally required enterprise-grade hardware.

---

## Summary

- **The core problem NF4 solves isn't "how to compress weights" in the abstract — it's "how to get quantile-quality quantization without paying for expensive, outlier-inaccurate per-tensor quantile estimation."** Exact quantiles require sorting/rank-estimating each tensor separately (expensive, must repeat per tensor); fast approximations are cheap but inaccurate exactly at the tails, where the most important (outlier) weights live.
- **The fix**: assume every weight tensor shares the same underlying shape (zero-centered Gaussian) and differs only by a scale factor. This lets the 16 quantile boundaries be computed **once, exactly, analytically** from the standard normal distribution — no sorting, no per-tensor cost, no approximation error anywhere, including the tails. Each block then only needs to store one scalar (its **quantization constant**, e.g. absmax) to reuse that single fixed table.
- **NF4's table is asymmetric by construction** (8 negative quantiles + 7 positive + a shared exact zero) specifically to guarantee a dedicated code for zero, since near-zero values are extremely common in trained weights.
- **NF4 is a storage-only datatype.** Because its levels are irregularly spaced (quantiles, not arithmetic steps), there's no formula from code to value — only a lookup table. Compute always happens in BF16, dequantized block-by-block right before each matmul, and discarded immediately after.
- **Quantizing a block always represents its own extreme value exactly** (it lands on the table's +-1.0 boundary after normalization) — a direct consequence of scaling by the block's own absmax.
- **Double quantization** compresses the per-block scale constants themselves (FP32 -> INT8), cutting their overhead from ~440 MB to ~112 MB on a 7B model — meaningful at 70B+ scale.
- **QLoRA freezes the NF4 base entirely — not just for memory savings, but because quantized weights can't be usefully trained via backprop at all** (rounding is a non-differentiable step function). Gradients flow *through* the frozen dequantized weights to reach the separate, full-precision LoRA adapters, which are the only parameters ever updated.
- **Net result**: NF4 + double quantization + LoRA + paged optimizers together bring 7B fine-tuning from ~60 GB down to ~7.5 GB — fitting on a single consumer GPU.

---

## Chapter Summary: The Quantization Landscape

| Method | Bits | Memory (7B) | Quality | Use Case |
|---|---|---|---|---|
| FP16/BF16 | 16 | 14 GB | Baseline | Training, reference |
| LLM.int8() | 8 | 7 GB | ~=FP16 | Easy 2x reduction |
| GPTQ INT4 | 4 | ~4 GB | -1-2% | Large model inference |
| AWQ INT4 | 4 | ~4 GB | -1-2% | Fast GPU inference |
| NF4 + DQ | ~4.5 | ~4 GB | -1-2% | Fine-tuning on consumer GPU |
| GGUF Q4_K_M | ~4.5 | ~4 GB | -1-2% | CPU inference |

- **Serving inference at scale** -> AWQ (fastest GPU throughput, most calibration-robust)
- **Maximum quality at INT4** -> GPTQ with `desc_act`
- **Fine-tuning on limited hardware** -> QLoRA with NF4
- **CPU / edge deployment** -> GGUF
- **Just need 2x reduction, zero quality loss** -> LLM.int8()