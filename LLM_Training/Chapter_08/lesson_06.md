# Chapter 8 · Lesson 6 — Diagnosis & Mental Models: Fine-Tuning Loss Curves vs. Pretraining Loss Curves

> **Where this fits:** Chapter 3, Lessons 8-9 built pretraining loss-curve diagnosis; Chapter 7, Lesson 8 built a fine-tuning-specific triage flowchart. This lesson is the direct side-by-side comparison — making explicit exactly how and why the "expected shape" baseline differs between the two contexts, which is a natural, common follow-up question once someone has demonstrated pretraining diagnosis skill.

---

## 1. Why a Side-by-Side Comparison Earns Its Own Lesson

It's tempting to assume loss-curve diagnosis is one unified skill applied identically in both contexts. It isn't — the "expected baseline" (Chapter 3, Lesson 8, Section 1's starting point for any diagnosis) is fundamentally different between pretraining and fine-tuning, and misapplying one context's expectations to the other produces incorrect diagnoses.

---

## 2. The Core Differences, Direct Comparison

| Dimension | Pretraining (Chapter 3) | Fine-tuning (this chapter) |
|---|---|---|
| Starting point | Random initialization — initial loss ≈ `log(vocab_size)` (Chapter 2, Lesson 1) | Already-converged checkpoint — initial loss is whatever the pretrained model's loss already is on this new data distribution, often already fairly low |
| Typical total steps | Many thousands to millions | Often hundreds to low thousands (Lesson 3 of this chapter's small-dataset, few-epoch norm) |
| Expected shape early on | Steep drop during warmup, then gradual (Chapter 3, Lesson 8, Section 1) | Often a much smaller, quicker adjustment — the model isn't learning from scratch, just adapting |
| Warmup's role | Substantial — protects against early Adam-variance instability over a long run (Chapter 3, Lesson 5) | Often much shorter or minimal (Chapter 8, Lesson 2, Section 2) given the short total run length |
| Overfitting relevance | Rare concern — massive, diverse corpus, typically under one epoch (Chapter 3, Lesson 7) | Central concern — small dataset, multiple epochs (Lesson 3 of this chapter) |
| What a "good" final loss looks like | Compared against scaling-law expectations for the compute/data budget (Chapter 3, Lesson 7) | Compared against the pre-fine-tune baseline's loss on the same data, and against validation-set behavior (Lesson 5 of this chapter) |

---

## 3. Reinterpreting Chapter 3's Symptom Table for the Fine-Tuning Context

Directly revisiting Chapter 3, Lesson 8's symptom-to-cause table, with the fine-tuning-specific reinterpretation noted:

**"Loss decreases very slowly, smoothly, no instability"** — in pretraining, this suggested LR too low (Chapter 4, Lesson 6). In fine-tuning, this is a more ambiguous signal: it could still mean LR too conservative (Chapter 8, Lesson 2's range), but given fine-tuning's already-good starting point, it could also simply mean the model was already fairly close to the target behavior and only needed a small adjustment — worth checking against the pre-fine-tune baseline's behavior before assuming the LR is the problem.

**"Loss decreasing but very slowly, noisy trajectory"** — in pretraining, pointed toward effective batch size too small (Chapter 3, Lesson 6). In fine-tuning, this remains relevant but is compounded by Lesson 3 of this chapter's point that fine-tuning's smaller absolute batch sizes and shorter total runs make batch noise proportionally more disruptive — the same symptom, a somewhat amplified underlying cause.

**"Loss plateaus early, well before expected training length"** — in pretraining, pointed toward LR decay schedule mismatch (Chapter 3, Lesson 8). In fine-tuning, given the much shorter expected training length to begin with (Section 2's table), an early plateau is *more likely* to simply mean training has genuinely converged for this small dataset — worth checking Lesson 5's validation curve before assuming something is wrong, since "early" is a relative, context-dependent judgment here.

**A symptom with NO direct pretraining equivalent, specific to fine-tuning:** training loss continuing to improve smoothly while validation loss diverges upward (Chapter 7, Lesson 8's overfitting branch) — this pattern is much less commonly a central diagnostic concern in pretraining (Section 2's table, "rare concern" row) but is one of the most important patterns to watch for in fine-tuning specifically.

---

## 4. Worked Example: The Same Raw Symptom, Two Different Diagnoses

Symptom, stated identically in both contexts: *"loss plateaued after a relatively small number of steps and hasn't moved since."*

**In a pretraining context** (Chapter 3, Lesson 8/9): given the expectation of many thousands of steps, a plateau after a comparatively small number of steps is a genuine red flag — likely pointing to a decay schedule misconfiguration (Chapter 3, Lesson 9's playbook) or an LR that decayed too aggressively too early.

**In a fine-tuning context** (this chapter): the same raw description, applied to a run that was only ever expected to run for a few hundred steps total (Lesson 3's small-dataset norm), might simply mean the fine-tune has converged normally — the diagnostic first move isn't "assume something is broken," it's "check this against the expected total run length and the validation curve (Lesson 5) before concluding anything is actually wrong."

**The point of this worked example:** identical surface-level symptom descriptions can point to entirely different conclusions depending on which of the two "expected baseline" columns from Section 2's table applies — this is the single most important transferable insight of this lesson, more so than any individual row of the comparison table.

---

## Key Takeaways

- Pretraining and fine-tuning loss curves need different "expected baseline" assumptions before any symptom can be correctly interpreted — the same raw curve shape can mean different things in each context.
- Fine-tuning's much shorter total run length, smaller dataset, and already-good starting point change what counts as a red flag versus normal, expected behavior.
- Overfitting (train/validation divergence) is a central fine-tuning-specific diagnostic concern with no strong pretraining equivalent, given pretraining's typically sub-one-epoch corpus exposure.
- The single most important skill from this lesson is recognizing which "expected baseline" column applies before interpreting a symptom — not memorizing a merged, context-free symptom table.

---

## Self-Check Before Moving to Lesson 7

1. Reproduce Section 2's comparison table from memory for at least four of the six dimensions.
2. Explain why "loss plateaus early" means something different in a pretraining context versus a fine-tuning context.
3. Why is train/validation divergence a much more central diagnostic concern in fine-tuning than in pretraining? Connect this to a specific earlier-chapter fact about corpus size and epoch count.