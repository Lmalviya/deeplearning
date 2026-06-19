# Quantization Lesson 7 — NF4: NormalFloat 4-bit and the QLoRA Format

---

## Why INT4 Is Suboptimal for Neural Network Weights

Recall what INT4 represents: 16 evenly-spaced values. For signed INT4, these are:
```
-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
```

These 16 values are equally spaced from minimum to maximum. Every representable value is the same "distance" from its neighbors.

Now look at the actual distribution of weights in a pre-trained neural network:

```python
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Sample weights from a typical layer
weights = model.model.layers[0].self_attn.q_proj.weight.data.float()

print(f"Mean: {weights.mean():.4f}")
print(f"Std:  {weights.std():.4f}")
print(f"Min:  {weights.min():.4f}")
print(f"Max:  {weights.max():.4f}")

# Distribution characteristics:
# Mean: ~0.0
# Std:  ~0.02-0.04
# Shape: approximately Normal (Gaussian)

# Values near zero are very common
# Values in the tails (large positive or negative) are rare
```

Neural network weights follow a **near-Gaussian (Normal) distribution**, centered at zero. This is not accidental — weight initialization is Gaussian, and gradient-based learning tends to maintain this approximate normality.

The implication: **equal spacing is wrong for this distribution**. 

In a Gaussian distribution:
- ~68% of values fall within 1 standard deviation of zero (the middle)
- ~95% fall within 2 standard deviations
- Only ~5% fall in the tails beyond ±2σ

INT4's equal spacing means it allocates half of its representable values to the tails (±4 to ±8 in scaled units) where only 5% of the data lives, and only half to the dense center where 95% of data lives.

This is an enormous waste of representable values.

---

## Information Theory: Optimal Quantization for Normal Data

From information theory, the optimal quantization for a distribution is **quantile quantization** — placing quantization boundaries such that each bin contains an equal fraction of the probability mass.

For a normal distribution, this means:
- Quantization levels are **densely packed near zero** (where most data lives)
- Quantization levels are **sparsely placed in the tails** (where little data lives)

For a standard normal distribution N(0, 1), the 16 optimal quantization values (for 4-bit) are the values at the **quantiles** `i/16` for `i = 0.5, 1.5, 2.5, ..., 15.5`:

```python
import scipy.stats as stats
import numpy as np

def compute_nf4_values() -> np.ndarray:
    """
    Compute the 16 NF4 quantization levels using quantile quantization
    for a standard normal distribution.
    """
    
    # 16 quantiles, evenly spaced, centered
    # Using i = 0.5/16, 1.5/16, ..., 15.5/16 to get 16 values
    quantile_levels = np.array([(i + 0.5) / 16 for i in range(16)])
    
    # The quantile function (inverse CDF) of the standard normal
    values = stats.norm.ppf(quantile_levels)
    
    return values

nf4_raw = compute_nf4_values()
print("Raw NF4 quantile values:")
print(np.round(nf4_raw, 4))
```

Output:
```
[-1.9673, -1.4741, -1.1477, -0.9013, -0.6893, -0.4994, -0.3192, -0.1574,
  0.1574,  0.3192,  0.4994,  0.6893,  0.9013,  1.1477,  1.4741,  1.9673]
```

### Visualizing the Difference

```
INT4 (equal spacing):
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
-8  -7  -6  -5  -4  -3  -2  -1   0   1   2   3   4   5   6   7

NF4 (quantile spacing, scaled to same range):
|  |  |  | |  | | ||   ||  | |  |  |  |
-8        -4   -2  -1  0  1  2   4       8

Gaussian density:
                    ████████████
                  ██████████████████
               ██████████████████████████
           ██████████████████████████████████
     ████████████████████████████████████████████
│────────────────────────────────────────────────│
-8                     0                        8
```

INT4 wastes representable values in the sparsely populated tails. NF4 concentrates representable values where the data actually is.

---

## The NF4 Format: Exact Construction

The NF4 values used in the QLoRA paper are not the raw quantiles — they are normalized to [-1, 1] and have exactly zero included:

```python
def construct_nf4_lookup_table() -> np.ndarray:
    """
    Construct the exact NF4 lookup table as in the QLoRA paper.
    """
    
    # Step 1: Compute quantiles for the negative and positive halves separately
    # This ensures exactly 0.0 is in the table
    
    # 7 negative values: quantiles 1/16 through 7/16 of N(0,1)
    negative_quantiles = np.array([i / 16 for i in range(1, 8)])
    negative_values = stats.norm.ppf(negative_quantiles)
    
    # 7 positive values (mirror): quantiles 9/16 through 15/16
    positive_quantiles = np.array([i / 16 for i in range(9, 16)])
    positive_values = stats.norm.ppf(positive_quantiles)
    
    # Combine: 7 negatives + 0 + 7 positives + special = 16 values
    # Wait — standard NF4 is asymmetric (8 negatives, 7 positives + 0)
    # The exact construction in bitsandbytes:
    
    nf4_values = np.array([
        -1.0,       # Index 0 (most negative)
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,        # Index 7 (zero)
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0         # Index 15 (most positive)
    ])
    
    return nf4_values


NF4_LOOKUP = construct_nf4_lookup_table()
print("NF4 values:", NF4_LOOKUP)
```

These 16 values are stored as a fixed lookup table. They are the same in every NF4 quantization — they never change.

---

## NF4 Quantization and Dequantization

### Quantization (Encoding)

To quantize a weight `w` to NF4:

1. **Normalize:** compute `w_norm = w / absmax(weight_group)` so values lie in [-1, 1].
2. **Find nearest:** find the index `i` in the NF4 table where `NF4_LOOKUP[i]` is closest to `w_norm`.
3. **Store:** store the 4-bit index `i` (0-15).

```python
def nf4_quantize(weights: np.ndarray, group_size: int = 64) -> dict:
    """
    Quantize weights to NF4 format.
    """
    
    original_shape = weights.shape
    weights_flat = weights.reshape(-1)
    n_groups = len(weights_flat) // group_size
    
    # Output: 4-bit indices (stored as uint8 for simplicity; 
    # in practice, two 4-bit values packed per byte)
    quantized_indices = np.zeros(len(weights_flat), dtype=np.uint8)
    absmax_per_group = np.zeros(n_groups, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = weights_flat[start:end]
        
        # Step 1: Compute absmax for this group (the scale)
        absmax = np.max(np.abs(group))
        absmax_per_group[g] = absmax
        
        # Step 2: Normalize to [-1, 1]
        if absmax > 0:
            normalized = group / absmax
        else:
            normalized = group
        
        # Step 3: Find nearest NF4 value for each weight
        for i, w_norm in enumerate(normalized):
            # Find the index of the nearest value in the NF4 table
            distances = np.abs(NF4_LOOKUP - w_norm)
            nearest_idx = np.argmin(distances)
            quantized_indices[start + i] = nearest_idx
    
    return {
        "indices": quantized_indices.reshape(original_shape),
        "absmax": absmax_per_group,
        "group_size": group_size
    }


def nf4_dequantize(quantized: dict) -> np.ndarray:
    """
    Dequantize NF4 back to float.
    """
    indices_flat = quantized["indices"].reshape(-1)
    absmax = quantized["absmax"]
    group_size = quantized["group_size"]
    
    weights_flat = np.zeros(len(indices_flat), dtype=np.float32)
    
    for g in range(len(absmax)):
        start = g * group_size
        end = start + group_size
        group_indices = indices_flat[start:end]
        
        # Look up NF4 values
        group_nf4_values = NF4_LOOKUP[group_indices]
        
        # Denormalize
        weights_flat[start:end] = group_nf4_values * absmax[g]
    
    return weights_flat.reshape(quantized["indices"].shape)
```

### Quality of NF4 vs INT4

```python
def compare_quantization_quality(weights: np.ndarray) -> dict:
    """Compare NF4 vs INT4 quantization error on typical weight data."""
    
    # NF4
    nf4_q = nf4_quantize(weights)
    nf4_dq = nf4_dequantize(nf4_q)
    nf4_error = np.sqrt(np.mean((weights - nf4_dq)**2))
    
    # INT4 (per-group absmax, same group size)
    int4_q = int4_quantize(weights)
    int4_dq = int4_dequantize(int4_q)
    int4_error = np.sqrt(np.mean((weights - int4_dq)**2))
    
    return {
        "nf4_rmse": nf4_error,
        "int4_rmse": int4_error,
        "improvement": (int4_error - nf4_error) / int4_error
    }

# Typical result for a weight matrix from a large LLM:
# NF4 RMSE: 0.00041
# INT4 RMSE: 0.00058  
# NF4 is ~30% more accurate than INT4 for normally distributed weights
```

---

## Double Quantization: Quantizing the Quantization Constants

NF4 alone requires storing one `absmax` (FP32) per group of 64 weights:
- 7B model with group_size=64: 7B/64 = ~110M scale factors × 4 bytes = 440 MB overhead

The QLoRA paper introduces **double quantization**: quantize the scale factors themselves.

```
Level 1 quantization: weights → NF4 indices (using absmax_1 per group)
Level 2 quantization: absmax_1 values → INT8 (using absmax_2 per group-of-groups)
```

```python
def double_quantize(weights: np.ndarray, 
                    inner_group_size: int = 64,
                    outer_group_size: int = 256) -> dict:
    """
    Double quantization: quantize both weights AND their scale factors.
    """
    
    # Step 1: First-level NF4 quantization
    # Produces: indices (4-bit) + absmax_1 (FP32, one per inner_group_size)
    level1 = nf4_quantize(weights, group_size=inner_group_size)
    absmax_1 = level1["absmax"]  # FP32, one per 64 weights
    
    # Step 2: Second-level INT8 quantization of absmax_1
    # Produces: absmax_1_int8 (INT8) + absmax_2 (FP32, one per outer_group_size)
    n_outer_groups = len(absmax_1) // outer_group_size
    
    absmax_1_int8 = np.zeros_like(absmax_1, dtype=np.int8)
    absmax_2 = np.zeros(n_outer_groups, dtype=np.float32)
    
    for g in range(n_outer_groups):
        start = g * outer_group_size
        end = start + outer_group_size
        group = absmax_1[start:end]
        
        max_val = np.max(np.abs(group))
        absmax_2[g] = max_val
        absmax_1_int8[start:end] = np.round(group / (max_val / 127)).clip(-127, 127)
    
    return {
        "nf4_indices": level1["indices"],   # 4-bit, dominant storage
        "absmax_1_int8": absmax_1_int8,      # INT8 (was FP32)
        "absmax_2": absmax_2,                # FP32, much smaller
        "inner_group_size": inner_group_size,
        "outer_group_size": outer_group_size
    }
```

### Memory Savings from Double Quantization

For a 7B model:

| Component | Without DQ | With DQ |
|---|---|---|
| NF4 indices (4-bit) | 3.5 GB | 3.5 GB |
| First-level scales (FP32) | 440 MB | → |
| First-level scales (INT8) | — | 110 MB |
| Second-level scales (FP32) | — | 1.7 MB |
| **Total overhead** | **440 MB** | **~112 MB** |

Double quantization reduces the scale factor overhead from 440 MB to ~112 MB — saving ~328 MB. For a 7B model, this is modest. For a 70B model (4.4 GB scale overhead → 1.1 GB), it becomes significant.

The QLoRA paper reports double quantization adds ~0.5 bits per parameter to the effective bit-width (on top of the 4 bits for the weights themselves), i.e., NF4 with double quantization is effectively ~4.5 bits per parameter.

---

## QLoRA: Putting NF4 Into Practice for Fine-Tuning

NF4 quantization was not introduced for inference — it was introduced to enable **fine-tuning of large models on consumer hardware**. This is the QLoRA paper (Dettmers et al., 2023).

### The Problem QLoRA Solves

Fine-tuning a 7B model in FP16:
- Model weights: 14 GB
- Gradients: 14 GB (same size as weights)
- Adam optimizer states: 28 GB (two moments, each same size as weights)
- Activations: variable, ~2-4 GB for typical batch sizes
- **Total: ~60 GB minimum** — requires multiple A100s

QLoRA solution:
- Freeze base model in NF4 (4-bit) — only 3.5 GB for 7B
- Add small trainable LoRA adapters in BF16 — only the adapters get gradients
- Adapter parameters are tiny (0.1-1% of base model) — negligible gradient/optimizer cost
- **Total: ~6-8 GB for 7B** — fits on a single RTX 3090 (24 GB)

### QLoRA Architecture

```
                    ┌──────────────────────────────────┐
                    │     Frozen NF4 Base Model         │
                    │                                   │
Input → Tokenizer → │  W_NF4 (4-bit) → dequantize BF16 │ → 
                    │                                   │
                    │   Dequantized on-the-fly to BF16  │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │     LoRA Adapters (BF16)          │
                    │                                   │
                    │   W_LoRA = W_base + α(BA)         │
                    │   A: [r × in_dim]  (trainable)    │
                    │   B: [out_dim × r] (trainable)    │
                    │   r: rank (typically 16-64)       │
                    └─────────────────────────────────┘
                                   │
                                Output
```

**The key insight:** The base model weights (NF4) are frozen — no gradients flow through them. Only the LoRA adapters (A and B matrices) receive gradient updates. The NF4 model participates only in the forward pass, where its weights are dequantized to BF16 just-in-time for the matrix multiplication.

### Paged Optimizers

One more QLoRA innovation: **paged optimizers** for handling memory spikes.

During training, when a long sequence is processed, the gradient computation requires more memory than average. Without paging, this causes OOM errors. QLoRA uses NVIDIA's unified memory (GPU ↔ CPU memory paging) to handle these spikes:

```python
from bitsandbytes.optim import PagedAdamW8bit

# Standard AdamW: stores 2 FP32 moments = 8 bytes/parameter
# PagedAdamW8bit: stores INT8 moments = 2 bytes/parameter
# Additionally pages to CPU RAM during memory spikes

optimizer = PagedAdamW8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01
)
```

---

## QLoRA Fine-Tuning in Practice

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

# Step 1: Load model in NF4 (4-bit) with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit NF4
    bnb_4bit_quant_type="nf4",            # NormalFloat 4-bit (not INT4)
    bnb_4bit_use_double_quant=True,        # Double quantization for scale factors
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16 after dequantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Step 2: Prepare model for k-bit training
# This disables certain operations incompatible with backprop through quantized layers
# and enables gradient checkpointing
model = prepare_model_for_kbit_training(model)

# Step 3: Add LoRA adapters
lora_config = LoraConfig(
    r=64,                          # LoRA rank (higher = more capacity)
    lora_alpha=16,                 # Scaling factor (often r/4)
    target_modules=[               # Which weight matrices to add adapters to
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Check trainable parameters
def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}% of total)")

print_trainable_params(model)
# Trainable: 167,772,160 (2.39% of total)  ← Only LoRA adapters!

# Step 4: Train
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llama-2-7b-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,          # Use BF16 instead
    bf16=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    optim="paged_adamw_8bit",   # Paged optimizer for memory efficiency
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Step 5: Save only the LoRA adapters (not the base model)
model.save_pretrained("./llama-2-7b-lora-adapters")

# Step 6: Load for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./llama-2-7b-lora-adapters")
```

---

## Memory Comparison

```
Task: Fine-tune Llama-2 7B

Full fine-tuning (FP16):
  Model weights:       14 GB
  Gradients:           14 GB
  Optimizer (Adam):    28 GB
  Activations:          4 GB
  Total:              ~60 GB  (requires 2× A100 80GB)

LoRA fine-tuning (FP16 base):
  Model weights:       14 GB
  Adapter gradients:   0.3 GB (only 2.4% of params)
  Optimizer:           0.6 GB (only adapter params)
  Activations:          4 GB
  Total:              ~19 GB  (1× A100 80GB)

QLoRA (NF4 base + LoRA adapters):
  Model weights (NF4): 3.5 GB
  Double quant scales: 0.1 GB
  Adapter gradients:   0.3 GB
  Optimizer (paged):   0.3 GB
  Activations:          3 GB (gradient checkpointing reduces this)
  Total:               ~7.5 GB  (1× RTX 3090 24GB!)
```

QLoRA makes it possible to fine-tune a 7B model on a single consumer GPU. This democratized fine-tuning — before QLoRA, fine-tuning any LLM larger than ~1B required enterprise GPU access.

---

## Summary: NF4 and Why It Matters

- **INT4 wastes representable values** by spacing them equally across the range, even though neural network weights follow a near-Gaussian distribution concentrated near zero.
- **NF4 (NormalFloat 4-bit)** uses quantile quantization — derived from information theory — to place the 16 representable values at the quantiles of the standard normal distribution. This allocates precision proportionally to data density.
- **NF4 is ~30% more accurate than INT4** for normally distributed weights at the same bit-width.
- **Double quantization** further compresses the scale factors (one per group of 64) by quantizing them to INT8, saving ~400 MB for 7B models.
- **QLoRA** combines NF4 + double quantization + LoRA adapters + paged optimizers to enable fine-tuning of 7B models on a single 24GB consumer GPU (and 13B models on a 48GB workstation GPU).
- The base model is frozen in NF4 and dequantized to BF16 on-the-fly for computation. Only the LoRA adapter parameters (~2-3% of total) receive gradient updates and require optimizer states.

---

## Chapter Summary: The Quantization Landscape

You have now covered the full quantization stack:

| Method | Bits | Memory (7B) | Quality | Use Case |
|---|---|---|---|---|
| FP16/BF16 | 16 | 14 GB | Baseline | Training, reference |
| LLM.int8() | 8 | 7 GB | ≈FP16 | Easy 2× reduction |
| GPTQ INT4 | 4 | ~4 GB | -1-2% | Large model inference |
| AWQ INT4 | 4 | ~4 GB | -1-2% | Fast GPU inference |
| NF4 + DQ | ~4.5 | ~4 GB | -1-2% | Fine-tuning on consumer GPU |
| GGUF Q4_K_M | ~4.5 | ~4 GB | -1-2% | CPU inference |

The right choice depends on your use case:
- **Serving inference at scale** → AWQ (fastest GPU throughput)
- **Maximum quality at INT4** → GPTQ with desc_act
- **Fine-tuning on limited hardware** → QLoRA with NF4
- **CPU / edge deployment** → GGUF
- **Just need 2× reduction, zero quality loss** → LLM.int8()