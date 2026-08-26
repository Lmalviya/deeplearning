# Chapter 10 · Lesson 5 — Modality Reality Check: Text LLMs vs. Diffusion/Vision vs. VLMs

> **Where this fits:** Lesson 4 covered the training-stage axis. This lesson covers the modality axis you explicitly asked about — what's shared across text, image, and multimodal training, and what's structurally different, grounded in real diffusion/ViT/CLIP training practice.

---

## 1. What's Shared Across Every Modality — the Universal Pattern Confirmed Again

Worth stating plainly before the differences: diffusion model and CLIP/ViT fine-tuning papers reviewed for this chapter show **exactly the same shape** as Lesson 4's text-model findings — a small discrete grid over learning rate (and sometimes one task-specific regularization weight), selected by a validation metric, with batch size, optimizer choice, and schedule shape fixed by convention rather than searched. The "shrink the search space, then grid the remainder" strategy (Lessons 1-3) isn't a text-specific finding — it's a genuinely cross-modality pattern.

---

## 2. What's Structurally Different: Vision-Specific Priors

**Layer-wise LR decay (Chapter 10, Lesson 2, Section 5), revisited as a vision-specific default rather than a curiosity:** this technique is *routine* in ViT/CLIP fine-tuning specifically — multiple reviewed papers apply it by default when fine-tuning a pretrained vision backbone, treating it as close to a standard practice rather than an optional refinement. This has no equally standard counterpart in typical decoder-only LLM fine-tuning (Chapter 7-8), where a single global LR is far more commonly used without layer-wise decay — worth knowing as a genuine modality-specific difference, not an oversight in the LLM-fine-tuning chapters.

**EMA (Exponential Moving Average) of weights — a diffusion-training-specific practice with no direct LLM-fine-tuning equivalent covered so far in this curriculum:** diffusion model training commonly maintains a separate, slowly-updated EMA copy of the model's weights throughout training, used for final sampling/inference rather than the raw, most-recently-updated weights. The EMA decay rate (β, commonly 0.999-0.9999) is itself a hyperparameter, but is treated similarly to other diffusion "stability" hyperparameters — set from known-good conventional ranges based on training duration, rather than searched.

```python
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]

    def update(self, model):
        for shadow, param in zip(self.shadow_params, model.parameters()):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
    # After training, the shadow_params (not the raw trained weights)
    # are typically what's used for generating final samples/inference
```

**Why EMA doesn't need to be swept as carefully as LR:** it's fundamentally a smoothing/stability mechanism rather than a capability-determining choice — the reviewed literature treats it as "pick a reasonable value based on training length" (longer training → higher/closer-to-1 decay, giving more smoothing) rather than something whose exact value needs fine-grained search, similar in spirit to how Chapter 4, Lesson 1 flagged AdamW's ε as "rarely tuned."

**Gradient clipping's role in diffusion training specifically:** while gradient clipping exists in text-model training too (Chapter 3, Lesson 9), the diffusion literature treats it as a more central, actively-monitored stability lever, given diffusion training's own well-documented tendency toward instability — worth knowing this is a difference in *emphasis and monitoring practice*, not a fundamentally different mechanism from what Chapter 3 already covered.

---

## 3. VLMs (Vision-Language Models) — Combining Both Worlds, With a New Wrinkle

**The core practical finding, worth stating clearly:** VLM fine-tuning in the reviewed literature typically applies **different treatment to different components of the same model** — e.g., a lower or layer-wise-decayed LR for the vision backbone (Section 2's vision-specific convention) combined with a separate, typically higher LR or separate LoRA configuration for the language-model component (following Chapter 7-8's LLM-specific conventions) — rather than a single global LR applied uniformly across a model that's actually composed of two structurally different pretrained components.

```python
vlm_training_config = {
    "vision_backbone": {
        "base_lr": 1e-6,           # Section 2's vision-specific low-LR convention
        "layerwise_decay": 0.9,     # Section 2 — standard for pretrained vision backbones
    },
    "language_model": {
        "lora_rank": 16,            # Chapter 7, Lesson 4's LLM-specific convention
        "lora_alpha": 32,
        "lr": 1e-5,                 # Chapter 8, Lesson 2's fine-tuning LR range
    },
    "cross_modal_projection": {
        "lr": 1e-4,                 # Often the HIGHEST LR of the three components —
                                     # this connector layer is frequently trained
                                     # from scratch or near-scratch, unlike the two
                                     # pretrained backbones it connects
    },
}
```

**Why the cross-modal projection component often gets the highest LR of the three, worth understanding rather than memorizing:** unlike the vision backbone and language model, which both arrive with substantial pretrained knowledge (warranting small, careful adjustments, per Chapter 7-8's catastrophic-forgetting-avoidance reasoning), the projection/connector layer that maps vision representations into the language model's embedding space is frequently initialized fresh or only lightly pretrained — it has comparatively little to "forget" and needs to learn a substantial new mapping, justifying a learning rate closer to what a from-scratch or DAPT-style component would use (Chapter 7, Lesson 1) rather than a fine-tuning-scale LR.

**The direct connection this creates back to Chapter 5's diagnostic discipline:** a VLM training failure diagnosed as "the model doesn't ground language in the image well" could stem from any of these three components being mistuned — an LR too low on the vision backbone (it never adapts at all), too high on the language model (catastrophic forgetting of language capability), or too low on the projection layer (the connector never learns the mapping) — three different root causes requiring three different fixes, directly extending Chapter 5, Lesson 1's "one symptom, multiple structurally different causes" framework to this specific architecture.

---

## 4. Comparison Table: Modality-Specific Additions to the Universal Pattern

| Modality | Universal pattern (Lessons 1-4) applies? | Modality-specific addition |
|---|---|---|
| Text LLM (decoder/encoder/encoder-decoder) | Yes, fully | Nothing beyond what Chapters 3-4, 7-9 already cover |
| Vision (ViT/CLIP fine-tuning) | Yes | Layer-wise LR decay as a near-default convention |
| Diffusion (image generation) | Yes | EMA weight averaging as a standard practice; heightened emphasis on gradient clipping monitoring |
| VLM (multimodal) | Yes, but applied per-component | Different LR treatment per architectural component (vision backbone, language model, cross-modal connector), with the connector often warranting the highest LR of the three |

---

## Key Takeaways

- The core "shrink the search space, grid the remainder" pattern from Lessons 1-3 holds across text, vision, and diffusion training — it's not a text-LLM-specific finding.
- Layer-wise LR decay is a vision-specific near-default convention with no equally standard LLM counterpart; EMA weight averaging is a diffusion-specific stability practice with no direct equivalent covered elsewhere in this curriculum.
- VLM training requires treating different architectural components (vision backbone, language model, cross-modal connector) with genuinely different hyperparameter conventions, not a single global LR.
- The cross-modal connector's frequently-higher LR is explainable by the same catastrophic-forgetting-avoidance logic (Chapter 7, Lesson 2) applied in reverse — it has little pretrained knowledge to protect, unlike the two backbones it connects.

---

## Self-Check Before Moving to Lesson 6

1. Explain why layer-wise LR decay is common in vision fine-tuning but not typically used the same way for decoder-only LLM fine-tuning.
2. What does EMA weight averaging do, and why doesn't its decay rate need careful searching the way LR does?
3. For a VLM, explain why the cross-modal projection layer often warrants a different (typically higher) LR than either of the two pretrained backbones it connects.
4. A VLM fails to ground language in visual input well. Using Section 3's reasoning, name three structurally different root causes and which architectural component each points to.