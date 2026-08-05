# Chapter 7 · Lesson 2 — Full Fine-Tuning: When It's Justified, Catastrophic Forgetting, Cost Math

> **Where this fits:** With Lesson 1's foundation-layer interventions covered, this lesson opens the behavior-layer toolbox proper — starting with the most expensive, most powerful option, and the one whose main risk (catastrophic forgetting) recurs as a concern throughout the rest of this chapter.

---

## 1. What Full Fine-Tuning Actually Is, Precisely

Every parameter in the model is updated during training on the fine-tuning dataset — no parameters are frozen, no additional small modules are added (contrast with LoRA/PEFT, Lessons 4-6). Mechanically nearly identical to Lesson 1's DAPT code — the difference is the training objective and data (instruction-following pairs or task-specific labeled data, rather than raw next-token prediction on a domain corpus) and typically a smaller token budget than DAPT.

---

## 2. When It's Actually Justified — Not the Default Choice

Given Chapter 5's diagnostic discipline, full fine-tuning should be reached for specifically when:
- **A confirmed behavioral capability gap exists** (Chapter 5, Lessons 3-9) that cheaper interventions (prompting, schema fixes, RAG) have already been ruled out for.
- **The scale of behavioral change needed is large** — PEFT methods (Lessons 4-6) work by modifying a small fraction of the model's effective capacity; if the required behavior shift is substantial (e.g., a significant style or domain-behavior overhaul, not a narrow skill), full fine-tuning's larger effective capacity for change may be genuinely necessary.
- **Sufficient compute and high-quality training data are available** — full fine-tuning is the most compute- and data-hungry option on the menu (Chapter 5, Lesson 11), and applying it with insufficient data risks overfitting badly to a small dataset.

**When it's usually the wrong call:** for a narrow, well-defined behavioral adjustment (Chapter 5's specific capability gaps often fall here), or when training data is limited, PEFT methods (Lessons 4-6) typically achieve comparable results at a fraction of the cost and with meaningfully lower catastrophic forgetting risk (Section 3) — full fine-tuning being the "default" choice without this comparison is a common, costly overreach.

---

## 3. Catastrophic Forgetting — The Core Risk, Mechanistically

**What it is:** updating all parameters to fit a new, often narrower dataset can overwrite the distributed representations that encoded previously-learned general capabilities — the model gets better at the fine-tuning task while getting measurably worse at things it could previously do well, that weren't represented in the fine-tuning data at all.

**Why it happens, mechanistically:** neural network parameters don't have cleanly separated "slots" for different capabilities — a given weight often participates in many different learned behaviors simultaneously (a consequence of the distributed representations transformers learn). Gradient updates driven entirely by a narrow fine-tuning objective can shift those shared weights in a direction that helps the new objective while incidentally degrading unrelated behaviors that depended on the same weights being in their pretrained configuration.

**The direct connection to Chapter 6, Lesson 7's regression-check layer:** this is precisely why that lesson insisted on checking capabilities the fine-tune *wasn't* targeting, not just the targeted one — catastrophic forgetting is invisible to an eval that only measures the intended improvement.

---

## 4. Worked Example: Quantifying Forgetting Risk Factors

Not a precise formula (forgetting severity depends on many interacting factors), but the directionally reliable relationships worth knowing:

```
Forgetting risk increases with:
  - Higher learning rate (larger parameter shifts per step)
  - More training steps / epochs on the fine-tuning data
  - Narrower/less diverse fine-tuning data relative to the original pretraining distribution
  - Smaller fine-tuning dataset relative to model capacity (more epochs needed to "use" the data,
    compounding the epoch-count risk above)

Forgetting risk decreases with:
  - Lower learning rate (directly, Chapter 8 covers the typical 10-100x reduction vs. pretraining LR)
  - Fewer epochs, with early stopping (Chapter 8, Lesson 5) triggered on a held-out general-capability check
  - Mixing in a small amount of general-purpose data alongside the fine-tuning data
    (a common practical mitigation — replaying a sample of the original training
    distribution during fine-tuning to anchor previously-learned behavior)
  - PEFT methods instead of full fine-tuning (Lessons 4-6) — by construction, modifying
    fewer parameters leaves more of the original weight configuration untouched
```

---

## 5. Cost Math — A Concrete Comparison

Directly extending Chapter 3, Lesson 3's memory arithmetic to the fine-tuning context:

```
Full fine-tuning a 7B model:
  Same ~18 bytes/parameter for weights + AdamW optimizer state as pretraining
  (Chapter 4, Lesson 1) — because ALL parameters are being trained
  7e9 * 18 bytes ≈ 126 GB — requires the same multi-GPU memory infrastructure
  as pretraining a model this size, even though the TRAINING DATA VOLUME
  is vastly smaller than pretraining's

LoRA fine-tuning the same 7B model (previewed here, full treatment in Lesson 4):
  Base model weights: ~14 GB (bf16, frozen, no optimizer state needed for these)
  LoRA adapter parameters: often <1% of total parameters
  Optimizer state needed only for the small adapter parameters
  Total: often fits on a SINGLE consumer/prosumer-grade GPU,
  vs. full fine-tuning's multi-GPU requirement
```

**The practical takeaway this comparison sets up for the rest of the chapter:** full fine-tuning's cost isn't just "somewhat more expensive" than PEFT — it can be the difference between needing a multi-GPU cluster (Chapter 3, Lesson 3's distributed training machinery) versus a single GPU, which is precisely why Lesson 9's method-choice framework treats this as a first-order cost consideration, not a minor implementation detail.

---

## Key Takeaways

- Full fine-tuning updates every parameter — mechanically similar to DAPT, but with instruction/task-specific data and typically far less data volume.
- It's justified for large-scale behavioral change with sufficient data and compute, confirmed via Chapter 5's diagnostic process — not a default first choice.
- Catastrophic forgetting is a real, mechanistically-explainable risk: distributed representations mean fine-tuning updates can overwrite previously-learned, unrelated capabilities.
- Forgetting risk is directionally predictable from learning rate, epoch count, and data diversity/size — and is mitigated by lower LR, early stopping, data-mixing, or choosing PEFT instead.
- The memory/compute cost gap between full fine-tuning and PEFT (previewed here, detailed in Lesson 4) is often the difference between needing a multi-GPU cluster and a single GPU.

---

## Self-Check Before Moving to Lesson 3

1. Explain catastrophic forgetting mechanistically — why does updating parameters for a new task risk degrading unrelated capability?
2. Name three factors that increase forgetting risk and three that decrease it.
3. Using Chapter 3 Lesson 3's memory arithmetic, explain why full fine-tuning a 7B model requires roughly the same infrastructure as pretraining one, even with much less training data.