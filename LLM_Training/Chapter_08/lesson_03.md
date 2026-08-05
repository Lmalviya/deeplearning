# Chapter 8 · Lesson 3 — Epochs, Effective Batch Size, and Overfitting Risk on Small Fine-Tuning Sets

> **Where this fits:** Pretraining (Chapter 3, Lesson 6) rarely completes even one full epoch over its massive corpus. Fine-tuning inverts this completely — multiple epochs over a small, fixed dataset is the norm, and that inversion is exactly what makes overfitting a first-order concern here in a way it mostly wasn't for pretraining.

---

## 1. Why Fine-Tuning's Epoch Story Is Fundamentally Different From Pretraining's

Chapter 3, Lesson 7's scaling laws assumed roughly 20 tokens per parameter, drawn from a massive, diverse corpus — pretraining typically sees each piece of training data once, or a small fraction of an epoch's worth of repetition across the whole corpus. Fine-tuning datasets (Chapter 7, Lesson 7) are often thousands to low-millions of tokens, not billions — running only a fraction of an epoch would mean the model barely sees most of the curated examples at all. **Multiple epochs (commonly 2-5, sometimes more for very small, high-quality datasets) are standard for fine-tuning specifically because the dataset is small enough that repetition is necessary to extract a meaningful training signal from it.**

---

## 2. The Direct Tension This Creates With Overfitting

This is precisely the mechanism behind Chapter 7, Lesson 8's overfitting risk factor list, made explicit here: each additional epoch is another full pass of gradient updates driven by the *same* fixed set of examples — after enough passes, the model has enough "opportunities" to start fitting the specific, idiosyncratic details of individual training examples (their exact phrasing, incidental correlations) rather than the general behavioral pattern those examples were meant to teach.

**The diagnostic signature, directly reusing Chapter 7, Lesson 8's flowchart:** validation loss (Chapter 7, Lesson 7's held-out split) starts diverging upward from training loss at some epoch — that divergence point is a genuine, measurable signal for how many epochs is actually appropriate for this specific dataset, not a number that can be reliably guessed in advance.

---

## 3. Effective Batch Size — Revisiting Chapter 3, Lesson 6 in the Fine-Tuning Context

Chapter 3, Lesson 6 established the linear scaling rule (batch size and LR move together) and the tokens-per-step framing for pretraining. For fine-tuning, the practical batch-size story is usually simpler and smaller-scale:

**Typical fine-tuning batch sizes are much smaller in absolute terms** (often tens to low hundreds of examples per effective batch, rather than pretraining's millions of tokens per step) — directly a consequence of small dataset size: an effective batch size approaching or exceeding a meaningful fraction of the total dataset size doesn't provide the same "many independent gradient estimates per epoch" benefit that a large batch provides against a much larger corpus.

**Gradient accumulation (Chapter 3, Lesson 4) remains directly relevant here**, particularly for QLoRA/LoRA setups on constrained hardware (Chapter 7, Lesson 5) — the same accumulation-loss-scaling mechanics apply unchanged, just at fine-tuning's smaller absolute batch-size scale.

**A fine-tuning-specific batch-size consideration Chapter 3 didn't need to cover:** very small batch sizes on a very small dataset can produce noisy, high-variance gradient estimates (Chapter 3, Lesson 6, Section 1's variance argument) relative to the limited number of total gradient steps available across the whole fine-tuning run — unlike pretraining, where a noisy estimate at one step is one of many thousands, a noisy estimate during a short fine-tuning run represents a much larger fraction of the total learning signal, making batch-size-driven noise a comparatively bigger concern here.

---

## 4. Worked Example: Reasoning Through Epoch Count for a Real Dataset

Say a fine-tuning dataset has 2,000 curated, deduplicated (Chapter 7, Lesson 7) tool-use examples. Walking a reasoned approach rather than guessing a fixed epoch count upfront:

1. **Start with a moderate default** (e.g., 3 epochs) as a first run, given this dataset size falls within the "thousands of examples" range where 2-5 epochs is a common starting convention.
2. **Monitor the train/validation loss gap throughout, per-epoch, not just at the end** — checking after each epoch (or more frequently) rather than only at final convergence catches the overfitting divergence point (Section 2) as close to when it actually happens as possible.
3. **If validation loss is still improving at epoch 3**, consider extending — the moderate default was a starting point, not a hard ceiling, and this dataset may tolerate more epochs before overfitting sets in.
4. **If validation loss diverges noticeably by epoch 2**, the appropriate response (directly connecting to Lesson 5 of this chapter, on early stopping) is to use a checkpoint from before the divergence point, not to complete all 3 planned epochs regardless.

**This reasoning process — start with a convention-based default, then let the validation curve determine the actual answer — is the transferable skill**, not "always use exactly 3 epochs," which would ignore that the right number is fundamentally dataset-dependent.

---

## 5. Diagnosis & Mental Models: Epoch Count and Dataset Size Interact

A useful mental model connecting Sections 1-3 together: the *appropriate* epoch count is inversely related to dataset size and diversity, and directly related to how much unique training signal the model needs to actually learn the target behavior (Chapter 5's diagnosed capability, and Chapter 7, Lesson 9's scope-of-change assessment).

```
Small, narrow dataset (few hundred examples, single well-defined skill):
  → fewer unique examples per epoch, but the skill may be simple enough
    to learn without excessive repetition → moderate epoch count,
    watch validation closely for early divergence

Larger, more diverse dataset (many thousands of examples,
broader behavioral coverage):
  → more unique signal per epoch, generally more epoch tolerance
    before overfitting sets in → can often support the higher end
    of the typical epoch range
```

---

## Key Takeaways

- Fine-tuning's multi-epoch norm exists specifically because datasets are small enough that a single pass doesn't provide adequate training signal — the opposite situation from pretraining.
- This same smallness is exactly what makes overfitting a first-order fine-tuning concern, with the train/validation divergence point serving as the concrete, measurable signal for appropriate epoch count.
- Fine-tuning batch sizes are typically much smaller in absolute terms than pretraining's, and batch-size-driven gradient noise is proportionally a bigger concern given fine-tuning's much shorter total training run.
- The right epoch count is genuinely dataset-dependent — a reasoned starting point plus close validation-curve monitoring beats guessing a fixed number upfront.

---

## Self-Check Before Moving to Lesson 4

1. Explain why fine-tuning commonly uses multiple epochs while pretraining typically doesn't complete even one, tying the explanation to dataset size specifically.
2. Why is gradient noise from a small batch size proportionally more concerning during fine-tuning than during pretraining?
3. Walk through Section 4's worked reasoning process for a hypothetical dataset of a different size, adapting the epoch-count decision accordingly.