# Quantization Lesson 6 — AWQ: Activation-Aware Weight Quantization

---

## The Problem with GPTQ's Approach

GPTQ achieves excellent INT4 quality by compensating for quantization errors after they occur — when you round a weight, you update its neighbors to absorb the error. This works well but has two limitations:

1. **Sequential dependency:** Each compensation depends on the previous, making parallelism difficult.
2. **Hardware-unfriendly weight ordering:** GPTQ's act-order (desc_act) mode reorders weights for quality, but this irregular memory access pattern hurts GPU throughput.

AWQ (Lin et al., 2023) takes a fundamentally different approach: **protect important weights before quantization, so they lose less precision to rounding.**

---

## The Core Observation: Not All Weights Are Equal

AWQ starts from an empirical observation that is easy to demonstrate:

```python
import torch
import numpy as np

def measure_weight_importance(weight_matrix: torch.Tensor, 
                               activations: torch.Tensor) -> torch.Tensor:
    """
    Measure which input channels (columns of the weight matrix) are most important.
    Importance = magnitude of activation × magnitude of weight.
    """
    
    # Average activation magnitude per input channel
    # activations shape: [n_samples, in_features]
    act_magnitude = activations.abs().mean(dim=0)  # Shape: [in_features]
    
    # Average weight magnitude per input channel
    # weight shape: [out_features, in_features]
    weight_magnitude = weight_matrix.abs().mean(dim=0)  # Shape: [in_features]
    
    # Importance combines both
    importance = act_magnitude * weight_magnitude
    
    return importance


# Demonstration: what happens when you protect the top 1% of channels
def demo_importance_of_salient_weights():
    
    weight = torch.randn(4096, 4096) * 0.02
    activations = torch.randn(128, 4096)  # 128 calibration samples
    
    importance = measure_weight_importance(weight, activations)
    
    # How many channels are "salient"?
    n_total = len(importance)
    n_salient = int(n_total * 0.01)  # Top 1%
    
    salient_idx = importance.topk(n_salient).indices
    normal_idx = torch.ones(n_total, dtype=torch.bool)
    normal_idx[salient_idx] = False
    
    print(f"Total channels: {n_total}")
    print(f"Salient channels (top 1%): {n_salient}")
    print(f"Importance of salient channels: {importance[salient_idx].sum() / importance.sum():.1%}")
    # Typically: "Importance of salient channels: 40-60%"
    # 1% of channels carry 40-60% of total importance!
```

The top 1% of input channels (by activation magnitude × weight magnitude) carry a disproportionate fraction of the total information throughput. If these channels are rounded aggressively, quality degrades. If they are protected, quality is preserved.

---

## AWQ's Solution: Scale the Salient Weights Up Before Quantization

Here is the elegant core idea:

If a weight value is `w = 0.012` and we quantize to INT4 with scale = 0.01 (so the INT4 range is [-8, 8] × 0.01 = [-0.08, 0.08]):

```
Quantized: round(0.012 / 0.01) = round(1.2) = 1
Dequantized: 1 × 0.01 = 0.010
Error: 0.012 - 0.010 = 0.002  (17% relative error)
```

Now, multiply this weight by a scale factor `s = 2.0` before quantization:

```
Scaled weight: 0.012 × 2.0 = 0.024
Quantized: round(0.024 / 0.01) = round(2.4) = 2
Dequantized: 2 × 0.01 = 0.020
After removing scale: 0.020 / 2.0 = 0.010  — same result
BUT: Error before scaling back: 0.024 - 0.020 = 0.004 (17% error on the scaled value)
After scaling back: 0.004 / 2.0 = 0.002 — same absolute error...
```

Wait — that gives the same error! The trick is more subtle. Let's look at it properly.

### The Real Mechanism

The quantization step size (the spacing between representable values) is `scale / 2^bits`. For INT4, if the scale is computed per-group:

**Without pre-scaling:**
```
Group: [0.012, 0.010, 0.011, 0.009, ...]
Max abs: 0.012
Scale: 0.012 / 8 = 0.0015
Step size: 0.0015

Weight 0.012 → quantizes to 8 (max) → exact
Weight 0.0075 → quantizes to round(0.0075/0.0015) = round(5) = 5 → 0.0075 exact
Weight 0.0078 → quantizes to round(0.0078/0.0015) = round(5.2) = 5 → 0.0075 (error: 0.0003)
```

**With pre-scaling salient weights by factor `s=2`:**
```
The salient weight 0.012 is pre-scaled to 0.024 before computing the group scale.
Group: [0.024, 0.010, 0.011, 0.009, ...]
Max abs: 0.024
Scale: 0.024 / 8 = 0.003
Step size: 0.003

Weight 0.024 → 8/8 scale → still exact (it's the max)
Weight 0.0078 → round(0.0078/0.003) = round(2.6) = 3 → 0.009 (error: 0.0012)
```

Hmm — that increased the error for the non-salient weights. This is the fundamental trade-off AWQ makes: **protect salient weights at the cost of slightly more error in non-salient weights.** Because salient weights carry far more importance, the net effect on model output is positive.

### The Compensation Step

But AWQ also needs to keep the actual computation correct. If you scale the weight by `s`, you must scale the corresponding activation by `1/s` to keep the output the same:

```
Original output:  y = W × x
After scaling:    y = (W × s) × (x / s) = W × x   ✓ Same output
```

The activation scaling by `1/s` happens through an additional per-channel multiplication on the input side — which can often be folded into the preceding LayerNorm or another linear layer, adding zero inference overhead.

---

## The AWQ Algorithm

```python
def awq_search_optimal_scales(
    weight: torch.Tensor,       # [out_features, in_features]
    activations: torch.Tensor,  # [n_samples, in_features]
    n_bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20            # Number of scale values to search over
) -> torch.Tensor:
    """
    Find optimal per-channel scales that minimize quantization error
    on the calibration data.
    """
    
    in_features = weight.shape[1]
    
    # Compute per-channel activation magnitude
    x_mean = activations.abs().mean(dim=0)  # [in_features]
    
    best_scales = torch.ones(in_features)
    best_error = float('inf')
    
    # Search over scale values
    # AWQ searches in the range [0.5, 1.0] relative to x_mean
    # (scale = x_mean^alpha, search over alpha in [0, 1])
    
    for alpha in torch.linspace(0, 1, n_grid):
        
        # Candidate scales (per channel)
        scales = x_mean.pow(alpha)
        scales = scales / scales.mean()  # Normalize to avoid changing overall magnitude
        
        # Scale weights by the candidate scales
        scaled_weight = weight * scales.unsqueeze(0)
        
        # Quantize the scaled weights
        q_weight = quantize_weight(scaled_weight, n_bits, group_size)
        
        # Dequantize
        dq_weight = dequantize_weight(q_weight, n_bits, group_size)
        
        # Remove the scaling
        dq_weight_unscaled = dq_weight / scales.unsqueeze(0)
        
        # Compute output error on calibration data
        # (compare W_original × x vs W_quantized × (x/scales) × scales)
        original_output = (weight @ activations.T)
        
        # The activation is scaled by 1/scales before the layer
        scaled_activations = activations / scales.unsqueeze(0)
        quantized_output = (dq_weight_unscaled @ scaled_activations.T)
        
        error = (original_output - quantized_output).pow(2).mean().item()
        
        if error < best_error:
            best_error = error
            best_scales = scales.clone()
    
    return best_scales


def quantize_weight(weight: torch.Tensor, n_bits: int, group_size: int) -> torch.Tensor:
    """Per-group absmax quantization."""
    W = weight.reshape(-1, group_size)
    
    scales = W.abs().max(dim=1, keepdim=True).values / (2**(n_bits-1) - 1)
    W_q = (W / scales).round().clamp(-(2**(n_bits-1)), 2**(n_bits-1) - 1)
    
    return W_q.reshape(weight.shape)
```

### Full AWQ Pipeline

```
1. For each linear layer:
   a. Collect input activations from calibration data
   b. Compute per-channel activation magnitude: x_mean[i] = mean(|x[:, i]|)
   c. Search for optimal scale factor per channel (minimize quantization error)
   d. Apply scales to weights: W_scaled[:, i] = W[:, i] × scale[i]
   e. Apply inverse scales to inputs (absorbed into previous layer)
   f. Quantize W_scaled to INT4

2. Result: INT4 weights + per-channel FP16 scales
   - The scale[i] for each input channel is absorbed into preceding operations
   - No inference overhead from the scaling step
```

---

## Using AWQ in Practice

The `autoawq` library provides a clean interface:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-int4"

quant_config = {
    "zero_point": True,   # Use zero-point for asymmetric quantization
    "q_group_size": 128,  # Per-group quantization
    "w_bit": 4,           # INT4
    "version": "GEMM"     # GEMM kernel (faster) vs GEMV (for batch size 1)
}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize (10-30 minutes)
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Load quantized model for inference
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

### Loading Pre-Quantized AWQ Models

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import torch

# Many models available with -AWQ suffix on HuggingFace
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,           # Fuse layers for faster inference
    trust_remote_code=False,
    safetensors=True
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")

# Generate
tokens = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
output = model.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

---

## AWQ vs. GPTQ: Detailed Comparison

### Quality

Both achieve similar quality at INT4. On standard benchmarks (MMLU, HellaSwag, ARC):

```
FP16 baseline:     100%  (reference)
GPTQ INT4 g128:    ~97-98% of FP16
AWQ INT4 g128:     ~97-98% of FP16
```

The quality difference between AWQ and GPTQ is generally within noise on most benchmarks. For specific domains, one may be slightly better than the other, but neither is consistently superior.

### Inference Speed

This is where AWQ has a clear advantage:

```python
# Benchmark: Llama-2-7B, batch size 1, single A100
# Tokens per second (higher is better):

# FP16:         ~2,100 tok/s
# GPTQ g128:    ~1,800 tok/s  (weights are in unusual order)
# AWQ GEMM:     ~2,400 tok/s  (faster than FP16 due to memory bandwidth savings)
```

AWQ's weight layout is hardware-friendly — weights stay in their natural order. The AWQ GEMM kernels (written in Triton/CUDA) are highly optimized and can actually exceed FP16 throughput because:
- Memory bandwidth is the bottleneck at small batch sizes
- INT4 weights require 4× less memory bandwidth to load
- The dequantization step is fast

### Quantization Speed

```
GPTQ:  10-30 min (7B), 2-4 hours (70B)  — needs Hessian computation
AWQ:   5-15 min (7B),  1-2 hours (70B)  — simpler grid search
```

### Memory Usage

Nearly identical:
```
Both GPTQ and AWQ INT4 g128: ~4 GB for 7B model
```

### Practical Recommendation

| Use case | Recommendation |
|---|---|
| Inference on GPU | AWQ (faster throughput) |
| Inference on CPU (llama.cpp) | GGUF (better CPU support) |
| Maximum quality | Test both, GPTQ with desc_act sometimes wins |
| Easiest setup | AWQ (simpler API, fewer hyperparameters) |
| Fine-tuning after quantization | NF4 + QLoRA (neither GPTQ nor AWQ support this) |

---

## Key Technical Insight: Why AWQ Works

The fundamental reason AWQ works is the **asymmetry of quantization damage**:

When you round a weight, the absolute rounding error is bounded by `scale/2`. But the **impact** of that rounding error on the model output is proportional to `error × activation`.

- For a salient channel with activation magnitude 10× the average: the impact of rounding error is 10× larger.
- For an unimportant channel with activation magnitude 0.1× the average: the impact is 10× smaller.

By scaling up salient channels before quantization, you force the quantization grid to be finer (smaller step size) in the channels that matter most. The slight coarsening of the grid in unimportant channels is a good trade.

AWQ is saying: **"Be precise where precision matters. Be imprecise where imprecision is cheap."**

This is the same philosophy behind NF4 (next lesson) and most good quantization schemes — the goal is not uniform precision but precision proportional to importance.

---

## Summary

- AWQ identifies "salient" input channels — the ~1% of dimensions that carry disproportionate importance (measured by activation magnitude × weight magnitude).
- Rather than compensating for errors after quantization (like GPTQ), AWQ prevents precision loss by scaling salient weights up before quantization, forcing a finer quantization grid in important channels.
- The corresponding activation scaling (by 1/scale) is absorbed into preceding operations — zero inference overhead.
- Scale factors are found by grid search on calibration data, optimizing the output error.
- AWQ and GPTQ achieve similar quality but AWQ has a hardware-friendly weight layout that enables faster GPU inference — often exceeding FP16 throughput at batch size 1.
- Use AWQ for production inference deployments. Use NF4 (next lesson) when fine-tuning is needed.

---

## What's Next

Lesson 7 covers NF4 — the NormalFloat 4-bit format introduced with QLoRA. This is the most theoretically elegant quantization format: derived from information theory to be the optimal quantization for normally distributed data. It is the format used to run fine-tuning on consumer GPUs.