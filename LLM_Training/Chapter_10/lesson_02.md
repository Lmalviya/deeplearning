# Chapter 10 · Lesson 2 — The "Shrink the Search Space" Toolkit

> **Where this fits:** This is the central insight from the research behind this chapter: real large-model teams don't out-search the problem with a fancier algorithm — they make the search space small enough that a tiny grid suffices. This lesson collects the specific techniques that do the shrinking, several already covered elsewhere in this curriculum but never framed as a unified strategy before now.

---

## 1. Reframing the Whole Problem

Lesson 1's comparison table implicitly asks "which search algorithm is best." This lesson asks a different, more consequential question: **can you avoid needing to search a given hyperparameter at all** — by deriving its value from structure, transferring it from a cheaper proxy, or inheriting it from prior published work with high confidence it'll transfer? Every technique below is a way of answering "yes" to that question for at least one hyperparameter, which is what makes Lesson 1's simple grid search sufficient for whatever's left.

---

## 2. Technique 1: μP / MetaP-Style Transfer — Already Covered, Now in Its Real Context

Chapter 4, Lesson 4 covered μP as a theoretical technique for pretraining LR transfer. Worth stating plainly now: this is not a hypothetical academic tool — Llama 4's own published methodology describes a procedure (referred to as "MetaP") specifically built to select per-layer learning rates and initialization scales that generalize across different batch sizes, model depths, and token budgets. **The direct consequence for your original question:** a team using this doesn't run HPO on the full-scale model's LR at all — it's computed via the transfer procedure from smaller-scale experiments, collapsing what would be an expensive full-scale search into a formula applied once.

---

## 3. Technique 2: Hyperparameter Scaling Laws

Directly extending Chapter 3, Lesson 7's Chinchilla-style scaling laws — but applied to *hyperparameters* rather than to loss as a function of model/data size. Published work exists (fitting curves like "optimal peak LR as a function of model size and token budget") that lets a team **predict** the optimal LR for a new, larger scale by fitting a curve on several smaller-scale experiments, then extrapolating — structurally identical in spirit to Chinchilla's compute-optimal model-sizing curve, just with "optimal LR" as the predicted quantity instead of "optimal parameter count."

```python
import numpy as np

# Conceptual illustration: fit a power-law relationship between
# model scale and optimal LR, using results from several small-scale sweeps
small_scale_results = {
    "60M":  {"optimal_lr": 1e-3},
    "130M": {"optimal_lr": 8e-4},
    "350M": {"optimal_lr": 4e-4},
    "1B":   {"optimal_lr": 2e-4},
}

sizes = np.array([6e7, 1.3e8, 3.5e8, 1e9])
lrs = np.array([1e-3, 8e-4, 4e-4, 2e-4])

# Fit log(lr) = a * log(size) + b  (a power law in log-log space)
coeffs = np.polyfit(np.log(sizes), np.log(lrs), deg=1)

def predict_optimal_lr(model_size):
    log_lr = coeffs[0] * np.log(model_size) + coeffs[1]
    return np.exp(log_lr)

predicted_7b_lr = predict_optimal_lr(7e9)  # extrapolate to the real target scale —
                                             # NO search run at 7B scale at all
```

**Why this is worth knowing as a named, real technique rather than an improvised curve-fit:** it's the direct generalization of something you already learned deeply (Chapter 3, Lesson 7) to a new target quantity — recognizing that generalization is itself a strong interview signal, since it shows the scaling-laws concept was actually internalized rather than memorized as "a Chinchilla fact."

---

## 4. Technique 3: Known Ratios Between Training Stages

Directly formalizing something Chapter 8, Lesson 2 and Chapter 9, Lesson 6 both touched on separately: the field has accumulated enough empirical experience that certain **ratios between stages** are treated as reliable priors, not things to rediscover via search every time.

| Known ratio | Why it's trusted enough to skip searching |
|---|---|
| Fine-tuning LR ≈ 10-100x lower than pretraining LR (Chapter 8, Lesson 2) | Extremely consistently observed across published fine-tuning work, for the mechanistic reason Chapter 8, Lesson 2 covered (starting from an already-good checkpoint) |
| Alignment-stage (DPO/RLHF) LR lower again than SFT LR, commonly by another order of magnitude | Consistently reported in published DPO/RLHF configs — the alignment stage operates on an already-instruction-tuned model, an even smaller further adjustment |
| LoRA alpha ≈ fixed ratio to rank, e.g. 2x (Chapter 7, Lesson 4) | A convention that holds the update's effective magnitude roughly constant as rank is varied, derived from the math itself, not empirically discovered per-model |

**The practical upshot:** a team doesn't search "what fraction should the DPO LR be of the SFT LR" — they start from the known ~10-100x-lower prior directly and, at most, do a small local search *around* that anchor point (Technique 4, next section) rather than treating it as an unknown dimension.

---

## 5. Technique 4: Structural Priors That Eliminate a Dimension Entirely — Layer-Wise LR Decay

New content, from the vision/CLIP fine-tuning research behind this chapter: **Layer-wise Learning Rate Decay (LLRD)**, common when fine-tuning a pretrained backbone (ViT, CLIP, and originally popularized for BERT-style fine-tuning in NLP) — rather than searching for a separate optimal LR per layer (an enormous, intractable search space), assign the *top* layer a base LR, then multiply by a fixed decay factor (e.g., 0.9) going down each layer toward the input.

```python
def layerwise_lr_decay(base_lr, num_layers, decay_factor=0.9):
    return [base_lr * (decay_factor ** (num_layers - i)) for i in range(num_layers)]

# Layer 0 (closest to input, most general features) gets the smallest LR;
# the final layer (most task-specific) gets close to the full base_lr
lrs = layerwise_lr_decay(base_lr=1e-4, num_layers=12, decay_factor=0.9)
```

**Why this reduces the actual search problem to almost nothing:** instead of an intractable per-layer search, there are now only **two** hyperparameters to consider — the base LR and the decay factor — and the decay factor itself is commonly held at a small set of conventional values (0.8-0.95) rather than searched at all, leaving effectively one real dimension (base LR) to tune via Lesson 1's simple grid. **The underlying reasoning that justifies this structural shortcut, worth stating explicitly:** early layers of a pretrained backbone already encode general, broadly useful features (edges/textures in vision, or low-level syntax in language models) that shouldn't need much adjustment; later layers are closer to the original pretraining task's specifics and benefit from more adjustment toward the new task — this is a *reasoned* prior, not an arbitrary convention.

---

## 6. Bringing It Together: A Full Worked Example

Say you're fine-tuning a large open-weight vision-language model. Applying every technique from this lesson before running a single experiment:

1. **Technique 3:** anchor the fine-tuning LR at roughly 10-100x below whatever the base model's known pretraining LR was (available from the model's technical report).
2. **Technique 4:** apply layer-wise LR decay to the vision backbone specifically, since it's a pretrained component being adapted, collapsing what would be a per-layer search into one base-LR-plus-conventional-decay-factor decision.
3. **Technique 1/2, if this were large enough to warrant it:** for a genuinely new pretraining run rather than fine-tuning, use μP-style transfer or a fitted scaling curve from smaller proxy runs rather than guessing at full scale.
4. **What's actually left to search:** one real dimension — the base LR — searched via Lesson 1's simple grid over a handful of values, validated on a cheap proxy (a data subset, or a smaller version of the model if one from the same family exists) before committing to the full run.

**This is the complete answer to your original question** — "what's the next set of hyperparameters to try" is rarely answered by a sophisticated search algorithm in practice; it's answered by having already eliminated most of the dimensions via Techniques 1-4, leaving a small enough remaining space that Lesson 1's simplest method is sufficient.

---

## Key Takeaways

- The real skill in large-model hyperparameter tuning is shrinking the search space via transfer, scaling laws, known ratios, and structural priors — not applying a more sophisticated search algorithm to a large space.
- μP/MetaP-style transfer and hyperparameter scaling laws both let a team predict a full-scale hyperparameter from smaller/cheaper experiments, avoiding a search at the expensive scale entirely.
- Known ratios between training stages (fine-tuning vs. pretraining LR, alignment vs. SFT LR, LoRA alpha vs. rank) are trusted priors accumulated across the field, not things to rediscover per-project.
- Layer-wise LR decay is a structural prior that collapses an intractable per-layer search into a two-parameter (often effectively one-parameter) problem, grounded in a reasoned argument about what early vs. late layers need.

---

## Self-Check Before Moving to Lesson 3

1. Explain the difference between Technique 1 (μP/MetaP transfer) and Technique 2 (hyperparameter scaling laws) — both predict a full-scale hyperparameter from smaller experiments, so what's actually different about their approach?
2. Why does layer-wise LR decay reduce an intractable search problem to essentially one dimension, and what's the reasoning that justifies doing this rather than searching each layer's LR independently?
3. Walk through Section 6's full worked example for a different scenario — full fine-tuning a text-only decoder model — adapting which techniques apply.