# Quantization Lesson 6 — AWQ: Activation-Aware Weight Quantization

---

## The Problem with GPTQ's Approach

GPTQ achieves excellent INT4 quality by compensating for quantization errors after they occur: it computes a Hessian from calibration data, then when it rounds a weight, it uses that Hessian to update the remaining unquantized weights to absorb the error. This works well, but it has real costs:

1. **Calibration overfitting risk.** Because GPTQ explicitly reconstructs each layer's output against its specific calibration activations, it can distort features that only matter for inputs unlike the calibration set — a real problem for LLMs, which are meant to generalize broadly, not specialize to whatever text happened to be used for calibration.
2. **Hardware-unfriendly weight ordering.** GPTQ's act-order (`desc_act`) mode reorders weights for better quality, but this irregular memory access pattern can hurt GPU throughput unless a specially optimized kernel (like ExLlama) is used to work around it.
3. **Expensive to compute.** The Hessian (`H = XXᵀ`) and its Cholesky-based inversion, even with GPTQ's speed tricks, are real per-layer computations.

AWQ (Lin et al., 2023) takes a fundamentally different approach: **protect important weights before quantization, so they lose less precision to rounding — using nothing more sophisticated than an average activation magnitude.**

## The Core Idea, Upfront

Before the details, here's the shape of the whole method: AWQ stores every weight as a plain INT4 code, just like GPTQ — same storage format, same eventual dequantize-before-matmul mechanics. What's different is entirely in *how those codes get chosen*. There is **no Hessian anywhere in AWQ** — it never needs second-order information, because it doesn't try to compensate for error after the fact the way GPTQ does. Instead, it identifies the small handful of input channels that matter most (using only a simple first-moment statistic — average activation magnitude), and **scales those channels up before quantizing**, so the rounding grid is naturally finer where it counts. Because this whole pipeline is cheaper and touches the calibration data far more lightly than GPTQ's reconstruction, AWQ also turns out to generalize better and needs less calibration data — a genuine result, not just a side effect.

---

## Step 1: Which Weights Actually Matter?

AWQ starts from an observation that's easy to demonstrate: not all weights in an LLM are equally important. Keeping a tiny fraction of weights — as little as 0.1–1% — at full FP16 precision, while quantizing everything else aggressively, can dramatically close the gap to full-precision quality.

The important question is *how* to find that critical 0.1–1%, and here the intuitive answer is wrong. The natural guess is to look at the weights themselves — keep whichever weights have the largest magnitude (or L2 norm), since large weights presumably contribute more to the output. The paper tested this directly, alongside two alternatives, by measuring perplexity when keeping different selections of channels at FP16:

| Selection method | Effect on quality |
|---|---|
| By weight magnitude | Barely better than random — does **not** meaningfully help |
| Random | Baseline (no real improvement) |
| **By activation magnitude** | Dramatically closes the gap to FP16, even keeping only 0.1% |

The insight: a weight's own size tells you little about its importance. What matters is the magnitude of the **feature** flowing through it — channels processing large, information-dense activations are the ones that matter, regardless of how large or small their weight values happen to be. This is the "activation-aware" half of the paper's name, and it's the entire reason the method looks at activations at all despite being a *weight-only* quantization scheme.

**Practical consequence for a "salience score":** it should be computed purely from calibration activations —
```
salience[i] = mean(|activation[:, i]|)     # per input channel i
```
— with no weight-magnitude term mixed in. This is also why AWQ needs no Hessian: a Hessian is second-order information used to figure out how weights *interact* so you can compensate for damage after the fact. AWQ never compensates for anything; it only needs to know, up front, which channels deserve a finer quantization grid — a much simpler question, answerable from a first-moment activation statistic alone.

---

## Step 2: Protecting Salient Channels Without Mixed Precision

Keeping salient channels in FP16 works, but a **mixed-precision** weight matrix (some values FP16, some INT4) is a nightmare for real hardware — irregular memory layout, no clean tensor-core support. AWQ's actual mechanism avoids this entirely: instead of changing a salient weight's *datatype*, it changes its *magnitude* before quantizing, then undoes that change on the activation side so the computed output is unaffected:

```
Original:      y = W · x
Scale up W:    y = (W · s) · (x / s) = W · x     ✓ identical output, s can be folded into the previous layer
```

**Why does scaling help?** For an INT4 group with scale `Δ = max(|w|) / 2^(N-1)`, the quantization error introduced by rounding one element scales with `Δ` itself — a coarser grid means bigger rounding errors. Scale a weight `w` up by `s > 1` before quantizing, and compare the resulting output error to the original:

```
Err(Q(w)x)         = Δ  · RoundErr(w/Δ)    · x
Err(Q(w·s)(x/s))    = Δ' · RoundErr(ws/Δ')  · x · (1/s)
```
The ratio of new error to old error is `(Δ'/Δ) · (1/s)`. Two things make this favorable:
- `RoundErr(·)` stays roughly constant regardless of scaling — rounding error is roughly uniform no matter the specific value.
- Scaling a *single* salient element barely moves the group's max, so `Δ' ≈ Δ`.

With `Δ' ≈ Δ`, the ratio collapses to about `1/s` — **the relative error for that one weight shrinks roughly in proportion to the scale factor.** That's the entire mathematical justification for the trick.

**The catch, and why `s` isn't pushed arbitrarily high:** this reasoning assumed scaling barely moves the group max. Scale *too many* channels, or by too large a factor, and `Δ'` does grow — which inflates the *absolute* error for every non-salient weight sharing that group, since their error is proportional to `Δ`. The paper measured this directly: scaling the top 1% of channels by `s=4` changed the group scale for over 21% of groups and hurt quality overall, while `s≈2` gave the best result. The rule of thumb: **be precise where precision matters, tolerate a little more coarseness where it's cheap — but don't overdo the precision gain at the expense of everything else.**

---

## Step 3: Finding the Right Scale — No Hessian, No Backprop, Just a Grid Search

Given the mechanism above, the remaining question is: what's the actual optimal scale per channel? Two paths are *not* taken, deliberately:

- **Not a Hessian-based reconstruction (GPTQ's approach)** — AWQ has no analogue to `H = XXᵀ` or per-weight compensation. It only ever tracks one number per channel: average activation magnitude.
- **Not gradient descent** — the rounding function isn't differentiable, so directly optimizing a per-channel scale with backprop isn't possible without an approximate-gradient trick (like a Straight-Through Estimator), and the paper found these converge unstably for this problem.

Instead, AWQ narrows the entire search to **one scalar `α` per layer**:
```
scale = (activation_magnitude) ^ α,     α searched over [0, 1] via a small grid (e.g. 20 candidates)
```
`α = 0` means no scaling at all; `α = 1` is the most aggressive scaling in the search space. For each candidate `α`, AWQ quantizes with that scale, measures the reconstruction error on calibration data, and keeps the `α` that minimizes it. This is cheap (a handful of quantize-and-measure passes, not an optimization problem), and touches the calibration data far more lightly than GPTQ's per-layer reconstruction — which is directly responsible for the generalization advantage covered below.

---

## The Full AWQ Algorithm

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
    on the calibration data. No Hessian, no gradients — just a grid
    search over a single scalar alpha per layer.
    """

    in_features = weight.shape[1]

    # Compute per-channel activation magnitude (the ONLY saliency signal used)
    x_mean = activations.abs().mean(dim=0)  # [in_features]

    best_scales = torch.ones(in_features)
    best_error = float('inf')

    # scale = x_mean^alpha, search over alpha in [0, 1]
    for alpha in torch.linspace(0, 1, n_grid):

        scales = x_mean.pow(alpha)
        scales = scales / scales.mean()  # normalize so overall magnitude is unchanged

        scaled_weight = weight * scales.unsqueeze(0)
        q_weight = quantize_weight(scaled_weight, n_bits, group_size)
        dq_weight = dequantize_weight(q_weight, n_bits, group_size)
        dq_weight_unscaled = dq_weight / scales.unsqueeze(0)

        original_output = (weight @ activations.T)
        scaled_activations = activations / scales.unsqueeze(0)
        quantized_output = (dq_weight_unscaled @ scaled_activations.T)

        error = (original_output - quantized_output).pow(2).mean().item()
        if error < best_error:
            best_error = error
            best_scales = scales.clone()

    return best_scales


def quantize_weight(weight: torch.Tensor, n_bits: int, group_size: int) -> torch.Tensor:
    """Per-group absmax quantization (symmetric, matching the paper's Eq. 1)."""
    W = weight.reshape(-1, group_size)
    scales = W.abs().max(dim=1, keepdim=True).values / (2**(n_bits-1) - 1)
    W_q = (W / scales).round().clamp(-(2**(n_bits-1)), 2**(n_bits-1) - 1)
    return W_q.reshape(weight.shape)
```

**Full pipeline, per layer:**
```
1. Collect calibration input activations
2. Compute per-channel activation magnitude (the salience signal)
3. Grid search a single alpha -> per-channel scale
4. Scale weights up by that factor, apply the inverse to activations (folded into the previous layer)
5. Quantize the scaled weights to INT4 (plain group-wise absmax, or asymmetric with a zero-point
   in production implementations — see the note on autoawq's config below)
```
Result: INT4 weight codes + per-group scales, with the extra per-channel scaling folded into neighboring operations at zero runtime cost.

---

## Storage vs. Compute, and Why AWQ Tends to Be Faster at Inference

Like every INT4 format covered in this course, there's no native INT4 matmul hardware, so AWQ stores plain INT4 codes and dequantizes to FP16 right before the matmul. What differs from GPTQ is entirely about *how friendly that storage layout is to hardware*:

- AWQ's weights stay in their **natural column order** — there's no reordering step analogous to GPTQ's `desc_act`, because AWQ never needed to reorder anything in the first place; its per-channel scales apply uniformly regardless of column order.
- The paper ships a dedicated inference system, **TinyChat**, that fuses dequantization directly into the matmul kernel (avoiding extra DRAM round-trips), plus platform-specific weight packing (e.g. SIMD-friendly layouts for ARM) and kernel fusion for attention/layernorm — squeezing out real, measured speedups (>3× vs. HuggingFace FP16 on both desktop and mobile GPUs), not just theoretical memory savings.

This is why AWQ is generally faster than GPTQ at small batch sizes: **not** because INT4 arithmetic itself differs between the two (it doesn't — both dequantize-then-matmul in FP16), but because AWQ's natural weight ordering and dedicated kernels avoid the memory-access irregularity that GPTQ's reordering can introduce. Against a GPTQ setup using an optimized kernel like ExLlama (no reordering penalty), the practical gap narrows — the fair statement is "AWQ is consistently hardware-friendly by construction," not "INT4 math is inherently faster under AWQ."

---

## Using AWQ in Practice

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-int4"

quant_config = {
    "zero_point": True,   # production AWQ typically uses asymmetric (zero-point) quantization,
                           # not the plain symmetric formula in quantize_weight() above
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"     # GEMM kernel (faster) vs GEMV (for batch size 1)
}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)   # 10-30 minutes for a 7B model

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

```python
# Loading a pre-quantized model from the Hub
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")

tokens = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
output = model.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

---

## AWQ vs. GPTQ

### Quality
Both reach similar quality at INT4 — generally within noise of each other on standard benchmarks (MMLU, HellaSwag, ARC), roughly 97–98% of FP16.

### Inference speed
AWQ has the edge at small batch sizes, but with the kernel caveat from above:
```
FP16:              ~2,100 tok/s
GPTQ (desc_act):   ~1,800 tok/s   (irregular memory access from reordering)
AWQ (TinyChat):    ~2,400 tok/s   (natural ordering + fused kernels)
```
GPTQ served through an optimized kernel without reordering (e.g. ExLlama) narrows this gap considerably — the comparison above reflects each paper's own reported setup, not a controlled apples-to-apples kernel comparison.

### Quantization speed
```
GPTQ:  10-30 min (7B), 2-4 hours (70B)  — Hessian computation + Cholesky decomposition
AWQ:   5-15 min (7B),  1-2 hours (70B)  — grid search over one scalar, no Hessian
```

### Memory usage
Nearly identical: both land around ~4 GB for a 7B model at INT4, group size 128.

### Generalization and data efficiency
This is where AWQ's design choice (simple statistic + grid search, instead of Hessian reconstruction) pays off measurably, not just theoretically:
- **~10× less calibration data needed** — the paper reaches good quality with as few as 16 calibration sequences, versus roughly 192 for GPTQ to reach comparable quality on the same setup.
- **More robust to calibration/deployment domain mismatch** — in a cross-domain test, AWQ's perplexity rose by only ~0.5–0.6 when calibration and evaluation text came from different domains, versus GPTQ's 2.3–4.9.
- This is part of why AWQ was shown to generalize well to instruction-tuned models and — for the first time in this line of work — multimodal vision-language models, both quite different from the generic calibration text typically used.

### Fine-tuning after quantization — a correction
It's a common claim (and one worth correcting explicitly) that "GPTQ/AWQ-quantized models can't be fine-tuned, only NF4+QLoRA can." **This is false as a blanket statement.** LoRA adapters can be attached to a frozen GPTQ-quantized (or AWQ-quantized) base model and trained normally — the same mechanism QLoRA uses (freeze the quantized base, train only the LoRA path in FP16, gradients flow *through* the frozen dequantized weights but never update them). This is a documented, working workflow (e.g. HuggingFace TRL + AutoGPTQ), and **QA-LoRA** (Xu et al., 2023) is an entire published method built specifically around fine-tuning INT4/GPTQ-style quantized models with LoRA — its adapters can even be merged back into the quantized base without extra precision loss, something standard NF4/QLoRA can't do as cleanly.

The real, narrower distinction: **no quantization method lets you update the quantized weights themselves** — GPTQ, AWQ, and NF4 all have this same limitation (fixed grid, no differentiable path to the codes). What NF4+QLoRA actually has going for it is mature, well-supported tooling (bitsandbytes + PEFT is the default path), not unique compatibility with fine-tuning.

### Practical recommendation

| Use case | Recommendation |
|---|---|
| Inference on GPU | AWQ (faster throughput, hardware-friendly by construction) |
| Inference on CPU (llama.cpp) | GGUF (better CPU support) |
| Maximum quality | Test both — GPTQ with `desc_act` sometimes wins |
| Easiest setup | AWQ (simpler API, fewer hyperparameters) |
| Least calibration data / most out-of-domain robustness | AWQ |
| Fine-tuning after quantization | Any of GPTQ, AWQ, or NF4 + LoRA can work in principle; NF4 + QLoRA remains the most mature, best-supported path |

---

## Key Technical Insight: Why AWQ Works

The core reason this all works is the **asymmetry of quantization damage**: the absolute rounding error from quantizing a weight is bounded by `scale/2`, but the *impact* of that error on the model's output is proportional to `error × activation`. A channel with 10× the average activation magnitude suffers 10× the output impact from the same rounding error; a channel with 0.1× the average suffers 10× less. Scaling salient channels up before quantization makes the effective grid finer exactly where that multiplier is largest, at the cost of a slightly coarser grid where it barely matters.

AWQ's philosophy in one line: **be precise where precision matters, be imprecise where imprecision is cheap** — the same principle behind NF4's quantile-matched grid (next lesson), just implemented through scaling rather than through the grid's shape.

---

## Summary

- **No Hessian anywhere in AWQ.** Unlike GPTQ, which computes `H = XXᵀ` and compensates for error after rounding, AWQ never compensates for anything — it only tracks one simple statistic (average activation magnitude per channel) before quantizing.
- **Salience is determined by activation magnitude, not weight magnitude.** The paper explicitly tested weight-magnitude-based selection and found it barely better than random; activation magnitude is what actually predicts which ~0.1–1% of channels matter.
- **Weights are stored as plain INT4 codes**, same as GPTQ — the difference is entirely in how the codes are chosen, not in the storage format itself. Salient channels are protected by scaling them up before quantizing (and scaling the corresponding activations down to cancel it out), avoiding a hardware-unfriendly mixed-precision format.
- **The math**: scaling a weight by `s` shrinks its relative quantization error by roughly `1/s`, because rounding error stays roughly constant and scaling one element barely shifts the group's scale — but scaling too aggressively or too broadly does raise the group scale, which is why a moderate factor (`s≈2` empirically) works best.
- **The optimal scale is found by a cheap grid search over one scalar `α` per layer** — not gradient descent (the rounding function isn't differentiable) and not Hessian-based reconstruction.
- This lighter-touch use of calibration data gives AWQ a **real, measured generalization advantage**: ~10× less calibration data needed, and much smaller quality loss when calibration and deployment domains differ, versus GPTQ.
- AWQ is generally **faster than GPTQ at small batch sizes**, mainly because its natural (unreordered) weight layout is more hardware-friendly — not because INT4 arithmetic itself is different. The gap narrows against optimized GPTQ kernels like ExLlama.
- **Correction:** it is *not* true that only NF4+QLoRA supports fine-tuning after quantization — LoRA adapters can be trained on top of a frozen GPTQ or AWQ base too (e.g. QA-LoRA). What differs is tooling maturity, not fundamental compatibility.
- Use AWQ for production GPU inference deployments where calibration data is scarce or may not match deployment data. Use NF4 (next lesson) specifically when fine-tuning support with mature tooling is the priority.

---

## What's Next

Lesson 7 covers NF4 — the NormalFloat 4-bit format introduced with QLoRA. This is the most theoretically elegant quantization format: derived from information theory to be the optimal quantization for normally distributed data. It is the format used to run fine-tuning on consumer GPUs.