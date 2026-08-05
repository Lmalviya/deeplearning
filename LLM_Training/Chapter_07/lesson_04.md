# Chapter 7 · Lesson 4 — LoRA: Math Derivation, Rank Selection, Code

> **Where this fits:** The centerpiece PEFT method, and the one most likely to come up in an interview by name. This lesson builds the math from first principles (not just "LoRA adds small matrices"), then implements it, directly justifying Lesson 2's cost comparison.

---

## 1. The Core Insight LoRA Is Built On

**The hypothesis:** the *change* in weights needed to adapt a pretrained model to a new task has a low "intrinsic rank" — meaning the update, even though it's applied to a huge weight matrix, can be well-approximated by a much lower-dimensional update. This is an empirical finding, not something derivable from first principles alone, but it's the load-bearing assumption behind the entire method.

**What this means concretely:** instead of learning a full update `ΔW` (same shape as the original weight matrix `W`, potentially enormous), LoRA learns `ΔW` as the **product of two much smaller matrices**, constraining its rank to be small by construction.

---

## 2. The Math, Derived Step by Step

For a pretrained weight matrix `W` of shape `(d_out, d_in)`, full fine-tuning would learn a full-rank update `ΔW`, also `(d_out, d_in)` — that's `d_out × d_in` new parameters, exactly matching Lesson 2's full fine-tuning cost.

LoRA instead parameterizes `ΔW` as:

```
ΔW = B @ A

where:  A has shape (r, d_in)
        B has shape (d_out, r)
        r << min(d_in, d_out)   — the "rank"
```

The forward pass becomes:

```
h = W @ x + ΔW @ x = W @ x + B @ (A @ x)
```

`W` stays **frozen** — no gradients, no optimizer state needed for it (directly explaining Lesson 2's cost comparison: no AdamW moment estimates needed for the base model's billions of parameters). Only `A` and `B` are trained.

**Parameter count comparison, worked concretely:** for a `d_out = d_in = 4096` weight matrix (a realistic attention projection size), full fine-tuning's `ΔW` has `4096 × 4096 ≈ 16.8M` parameters. LoRA with rank `r = 8`: `A` has `8 × 4096 = 32,768` parameters, `B` has `4096 × 8 = 32,768` parameters, total `65,536` — **roughly 256x fewer trainable parameters** for this one matrix, directly because rank `r=8` is so much smaller than `min(4096, 4096) = 4096`.

---

## 3. Initialization — Why It Matters and What's Standard

`A` is initialized with small random values (commonly Gaussian), and `B` is initialized to **all zeros**. This isn't arbitrary: with `B = 0`, `ΔW = B @ A = 0` at the very start of training — meaning the model's forward pass is **identical to the original pretrained model** at initialization, and the LoRA update only starts to have any effect as `B` moves away from zero during training. This is a deliberate, important design choice: it guarantees training starts from the exact pretrained behavior, rather than starting from some random perturbation that could immediately degrade the base model's capability before any useful learning has happened.

---

## 4. The Alpha Scaling Factor

LoRA's actual update is typically scaled: `ΔW = (alpha / r) * B @ A`, where `alpha` is a separate hyperparameter. **Why this scaling exists:** it decouples the *magnitude* of the update's effect from the *rank* choice — without this scaling, changing `r` (to explore different capacity/cost tradeoffs) would also inadvertently change how strongly the LoRA update affects the forward pass, conflating two things that are conceptually separate decisions. A common convention is setting `alpha = 2 * r` or `alpha = r` as a starting point, then tuning from there — Chapter 8 covers this hyperparameter's tuning in full depth.

---

## 5. Code: A Minimal LoRA Layer From Scratch

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False   # freeze the base weight — Section 2's key point

        d_out, d_in = base_linear.weight.shape
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # zero init, per Section 3

    def forward(self, x):
        base_output = self.base_linear(x)
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        return base_output + self.scaling * lora_output


def apply_lora_to_model(model, target_modules, rank=8, alpha=16):
    """
    target_modules: e.g. ["q_proj", "v_proj"] — which linear layers to wrap
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original_linear = getattr(module, target)
                setattr(module, target, LoRALinear(original_linear, rank, alpha))
    return model
```

---

## 6. Rank Selection and Target Module Choice — the Practical Decisions

**Rank (`r`):** directly controls the capacity/cost tradeoff established by Section 2's math. Lower rank (4-8) is sufficient for narrower behavioral adjustments (Lesson 2's "narrow, well-defined gap" case from Chapter 5's diagnosis); higher rank (32-64+) approaches something closer to full fine-tuning's expressiveness, at proportionally higher cost — Chapter 8, Lesson 1 covers concrete ranges in depth.

**Target modules — which weight matrices actually get a LoRA adapter:** commonly applied to the attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj` from Chapter 3, Lesson 1's attention code) and sometimes the feed-forward layers as well. **The tradeoff, worth being able to state:** applying LoRA to more modules increases capacity (and cost) but doesn't uniformly help — some published findings suggest applying LoRA broadly across more module types with a lower rank per module can outperform concentrating a higher rank on fewer module types, for the same total parameter budget — a genuine empirical finding worth citing as nuance rather than assuming "more modules is strictly better."

---

## Key Takeaways

- LoRA's core assumption is that task-adaptation weight updates have low intrinsic rank — an empirical hypothesis, not a first-principles guarantee, but a well-validated one in practice.
- The update is parameterized as `B @ A`, with rank `r` controlling parameter count directly — a concrete, computable savings (often 100x+ fewer trainable parameters for a given matrix) versus full fine-tuning.
- Zero-initializing `B` guarantees training starts from exactly the pretrained model's behavior, not a random perturbation.
- Alpha scaling decouples update magnitude from rank choice, letting the two be tuned independently rather than conflated.
- Target module choice (which weight matrices get adapters) is a real tradeoff, not obviously "more is better" — broader-but-lower-rank sometimes outperforms narrower-but-higher-rank at equal budget.

---

## Self-Check Before Moving to Lesson 5

1. Derive, from the parameter-count math in Section 2, why LoRA's savings become more dramatic as `d_in`/`d_out` grow larger relative to a fixed rank `r`.
2. Explain precisely why `B` is zero-initialized and what would go wrong if both `A` and `B` were randomly initialized instead.
3. What does the alpha scaling factor decouple, and why does that decoupling matter practically when experimenting with different ranks?