# Chapter 8 · Lesson 2 — Learning Rate for Fine-Tuning vs. Pretraining

> **Where this fits:** Chapter 4 covered pretraining learning rates in depth. This lesson is specifically about how and why fine-tuning's learning rate story differs — a frequently-asked follow-up to any fine-tuning hyperparameter question, and one your original interview's Chapter 4-style content didn't yet address for the fine-tuning-specific case.

---

## 1. The Core Fact, and the Reasoning Behind It

Fine-tuning learning rates are typically **10-100x smaller** than pretraining peak learning rates (Chapter 4, Lesson 2's ~1e-4 to 1e-3 pretraining range becomes roughly 1e-5 to 1e-4, or even lower, for fine-tuning). This isn't an arbitrary convention — it follows directly from what fine-tuning is actually doing.

**The reasoning, precisely:** pretraining starts from random initialization and needs to establish a good representation from scratch — large updates are appropriate and necessary early on. Fine-tuning starts from an **already well-trained** checkpoint (Chapter 7, Lesson 2's starting point) — the existing weights already encode substantial useful structure, and the goal is a comparatively small, targeted adjustment, not a rebuild. A learning rate sized for "rebuild from scratch" applied to "make a small adjustment" risks exactly the catastrophic forgetting mechanism Chapter 7, Lesson 2, Section 3 described — large parameter shifts overwriting distributed representations that encode capabilities the fine-tuning data never touches.

---

## 2. Connecting This to Warmup and Schedule — What Changes, What Doesn't

Chapter 3, Lesson 5's warmup mechanism (Adam's early variance-estimate unreliability) still applies in principle during fine-tuning, but the practical need is often smaller, precisely because fine-tuning runs are much shorter (fewer total steps, Chapter 7, Lesson 8's territory) — a long warmup phase (Chapter 3 Lesson 5's "few hundred to few thousand steps") could consume a large fraction of a short fine-tuning run's total budget. **A common practical adjustment:** shorter warmup (sometimes just tens of steps, or occasionally omitted for very short, low-LR fine-tuning runs where instability risk is already low given the small LR itself) — worth stating as a reasoned adjustment rather than either blindly reusing pretraining's warmup length or dropping it without justification.

**Decay schedule choice:** cosine decay (Chapter 3, Lesson 5) remains a reasonable default for fine-tuning too, but given the much shorter total step count, the practical difference between cosine and simpler linear decay is often smaller than it is at pretraining scale — worth knowing this is a lower-stakes choice in the fine-tuning context specifically.

---

## 3. Worked Example: Scaling From a Pretraining Reference

Say a 7B model's pretraining used a peak LR of `3e-4` (within Chapter 4, Lesson 2's mid-scale range). A reasonable fine-tuning LR starting point, applying the 10-100x reduction:

```
Conservative (safer against forgetting, slower adaptation): 3e-4 / 100 = 3e-6
Moderate (common practical starting point):                 3e-4 / 30  ≈ 1e-5
Aggressive (faster adaptation, higher forgetting risk):      3e-4 / 10  = 3e-5
```

**How to choose within this range, tied to Chapter 7's method-selection lesson:** a narrower-scope, lower-rank LoRA fine-tune (Lesson 1's rank guidance) can often tolerate the more aggressive end of this range with lower risk, since LoRA's fundamentally constrained parameter count already limits how much damage an aggressive LR can do to the frozen base model's capability (Chapter 7, Lesson 2, Section 4's point about PEFT reducing forgetting risk structurally). Full fine-tuning, given its full-parameter exposure to forgetting risk, more often warrants the conservative end.

---

## 4. Exceptions Worth Knowing — When the "10-100x Smaller" Rule Doesn't Hold

**DAPT is the clearest exception, and worth cross-referencing explicitly:** Chapter 7, Lesson 1 covered continued pretraining/DAPT as mechanically closer to original pretraining than to fine-tuning proper — DAPT's learning rate, while still typically somewhat lower than the original pretraining peak LR (to protect against forgetting general capability, per Lesson 1's own warning), doesn't follow the aggressive 10-100x fine-tuning reduction either, since DAPT is meant to genuinely add substantial new knowledge, not make a small targeted behavioral adjustment — it sits somewhere between the two regimes, worth explicitly distinguishing rather than applying either rule blindly.

**Alignment-stage training (Chapter 9) is another distinct regime:** RLHF/DPO-style training often uses even more conservative learning rates than typical SFT-style fine-tuning, partly because these methods are already operating on a model that's been through pretraining *and* instruction-tuning, and partly because alignment training's objective (preference-based, not next-token prediction) has different stability characteristics — Chapter 9 covers this specifically, but worth flagging now that "fine-tuning LR is always 10-100x lower than pretraining" isn't a universal rule across every training stage this curriculum covers.

---

## 5. Diagnosis & Mental Models: Reading LR-Related Symptoms Specific to Fine-Tuning

Directly extending Chapter 4, Lesson 6's LR symptom table to the fine-tuning-specific context:

| Symptom | Fine-tuning-specific interpretation |
|---|---|
| Training loss barely moves from the pretrained checkpoint's starting loss | LR likely far too conservative — check against Section 3's range, may need to move toward the more aggressive end |
| Training loss drops very fast, converges to a very low value within a handful of steps | Possible sign of LR too aggressive, especially concerning given fine-tuning's typically small dataset — this "too-good-too-fast" pattern is a specific fine-tuning red flag for memorization/overfitting risk (Chapter 7, Lesson 8) that looks superficially like a training-health win |
| Good fine-tuning task performance, but Chapter 6's regression check shows unrelated-capability degradation | Classic catastrophic forgetting signature (Chapter 7, Lesson 2) — first lever to try is reducing LR further, before assuming a method change (e.g., full fine-tuning to LoRA) is needed |

---

## Key Takeaways

- Fine-tuning LR is typically 10-100x smaller than pretraining peak LR, directly because fine-tuning starts from an already-good checkpoint and needs small, targeted adjustment rather than large-scale learning.
- Warmup and decay schedule choices carry over conceptually from pretraining but are lower-stakes and often shortened, given fine-tuning's much smaller total step count.
- The choice of where in the 10-100x range to land should be informed by method choice (LoRA vs. full fine-tuning) per Chapter 7's forgetting-risk reasoning.
- DAPT and alignment-stage training are genuine exceptions to the "10-100x smaller" fine-tuning rule, each with their own reasoning — worth distinguishing explicitly rather than over-generalizing one rule across every training stage.
- A "too-good-too-fast" loss drop is a fine-tuning-specific red flag for overfitting risk, not automatically a training-health success.

---

## Self-Check Before Moving to Lesson 3

1. Explain, from first principles (not just citing the rule), why fine-tuning needs a much lower learning rate than pretraining.
2. Why does DAPT not follow the same 10-100x reduction rule that general fine-tuning does?
3. A fine-tuning run's loss drops extremely fast and low within very few steps. Why is this a potential red flag rather than an unambiguous success, specific to the fine-tuning context?