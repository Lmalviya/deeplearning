# Chapter 10 · Lesson 4 — Stage-by-Stage Reality Check: Pretraining, SFT, Alignment

> **Where this fits:** Lessons 1-3 built the general strategy. This lesson grounds it with real, cited numbers from actual published training configurations at each stage — directly answering your concern that earlier chapters covered pretraining tuning but left fine-tuning and alignment's tuning practice unaddressed.

---

## 1. Pretraining — What Real Configs Actually Show

Extending Chapter 4's ranges with real reported search practice: published replications of the LLaMA training recipe (across several independent papers studying LLaMA-style models at 60M-1B scale) consistently show **a single hyperparameter searched** — learning rate — over a small discrete list (commonly 4-6 values, e.g. `{1e-4, 2e-4, ..., 1e-3}` or similar log-spaced sets), selected by validation perplexity. Every other hyperparameter — AdamW β1=0.9, β2=0.95, weight decay 0.1, gradient clip norm 1.0, cosine schedule with 10% warmup — appears **unchanged across papers and model scales**, treated as an inherited default (Lesson 3's pattern), not something re-searched per project.

**What this confirms about the actual scope of pretraining HPO in practice:** it is almost never a multi-dimensional search — it's a one-dimensional LR sweep on a cheap proxy scale, with everything else fixed by convention, exactly the "shrink the search space to nearly nothing" pattern from Lesson 2.

---

## 2. SFT / Instruction Tuning — What Real Configs Actually Show

Directly filling the gap you flagged — SFT tuning practice, grounded in real fine-tuning papers using Llama and Gemma models: the dominant pattern is again **a single-dimension LR sweep**, over a small discrete list scaled to model size — for example, one paper searching `{2e-6, 5e-6, 7e-6, 1e-5}` for a 1B-scale model and a correspondingly higher range (`{2e-5, 5e-5, 7e-5, 1e-4}`) for an 8B-scale model, selecting the winner by a task-specific validation metric (not just loss — often a downstream accuracy metric like GSM8K accuracy in the papers reviewed for this chapter).

**What's notably NOT searched, across essentially every SFT paper reviewed:** LoRA rank and alpha are typically fixed at conventional values (rank 8 or 16, alpha equal to or double the rank) rather than swept; batch size is fixed based on hardware constraints (Chapter 8, Lesson 3) rather than treated as a free hyperparameter; epochs are fixed at a small number (2-3) rather than searched, with early stopping (Chapter 8, Lesson 5) as the actual mechanism for finding the right stopping point within that budget rather than epoch count itself being a swept hyperparameter.

---

## 3. Alignment (DPO/RLHF) — What Real Configs Actually Show

This is where the research behind this chapter found the richest signal. Real published DPO configurations show **two dimensions typically searched**, not one: learning rate and β, each over a small discrete list.

```
A representative pattern found across multiple DPO papers:
  Learning rate: {5e-7, 1e-6, 5e-6, 1e-5}  — searched
  Beta (β):      {0.01, 0.05, 0.1, 1.0}    — searched, but often with far
                                              narrower effective range in
                                              practice (0.01–0.1 dominates
                                              published "best" configs)
```

**The consistent ratio confirmed across sources:** DPO's learning rate is consistently reported as roughly 10-100x lower than the SFT learning rate that preceded it in the same pipeline — directly the ratio Chapter 8, Lesson 2 and Chapter 9, Lesson 6 both anticipated, now confirmed as a real, widely-observed empirical pattern rather than just a theoretical prediction from this curriculum's reasoning.

**A specific, notable practice worth knowing:** at least one reviewed paper explicitly tuned hyperparameters on a **single random seed**, then carried the resulting best configuration over to the other seeds used for the paper's final reported results — a direct, concrete instance of Lesson 2's "tune cheap, transfer to expensive" principle, applied not to model scale but to the number of stochastic training repetitions.

---

## 4. Full Comparison Table Across Stages

| Stage | What's actually searched | Typical search size | What's inherited/fixed |
|---|---|---|---|
| Pretraining | LR only | ~4-6 discrete values | AdamW betas, weight decay, clip norm, schedule shape — from prior published recipes |
| SFT / instruction tuning | LR only (occasionally LoRA rank if genuinely novel task) | ~3-5 discrete values, scaled to model size | LoRA rank/alpha (convention), batch size (hardware-driven), epoch count (fixed small number, early-stopping-governed) |
| Alignment (DPO/RLHF) | LR AND β (two dimensions) | ~3-5 values each | The 10-100x LR ratio to SFT (a strong prior, so the searched LR range is already narrow), reward/preference data pipeline itself (Chapter 9, Lesson 4) |

**Why alignment gets a genuinely two-dimensional search while the other stages mostly don't, worth being able to explain rather than just observe:** β has no equivalent in SFT or pretraining at all — it's a fundamentally new hyperparameter controlling a tradeoff (Chapter 9, Lesson 6) that doesn't exist in those earlier stages, so there's no equivalent inherited prior to lean on the way LoRA rank/alpha conventions exist for SFT — this is a case where Lesson 2's "known ratios" technique doesn't (yet) fully apply, and a genuine small search is the honest answer.

---

## 5. Worked Example: Planning a Full Pipeline's Tuning Budget

Given a fixed total tuning budget across all three stages for a new model pipeline, applying this lesson's findings to allocate it sensibly:

```
Pretraining:  small proxy-scale LR sweep (Section 1) — cheap, budget: low
SFT:          LR sweep at the actual target scale, small discrete list
              (Section 2) — budget: low-moderate, since LR is the only
              real dimension and the range is narrow given known priors
Alignment:    LR AND beta sweep (Section 3) — budget: moderate, genuinely
              two dimensions, though each range is narrow given the
              10-100x-lower-than-SFT prior already constrains where to look
```

**The overall allocation principle this table demonstrates:** budget should scale with how many genuinely under-determined dimensions a stage has, not evenly across stages — alignment reasonably gets more search budget than pretraining or SFT specifically because it has a real second free dimension (β) that the other two stages don't.

---

## Key Takeaways

- Pretraining and SFT tuning practice, per real published configs, is almost entirely a single-dimension LR sweep with everything else inherited from convention.
- Alignment (DPO/RLHF) is the one stage with a genuine second free dimension (β), making it the stage most worth allocating extra search budget to.
- The 10-100x SFT-to-alignment LR ratio is confirmed as a real, consistently observed empirical pattern across published work, not just a theoretical prediction.
- Tuning-budget allocation across a pipeline should track how many genuinely under-determined dimensions each stage has, not be spread evenly.

---

## Self-Check Before Moving to Lesson 5

1. Reproduce Section 4's comparison table from memory, including the "what's inherited/fixed" column for each stage.
2. Explain why alignment tuning genuinely needs a two-dimensional search while pretraining and SFT mostly don't.
3. Given a new pipeline with a fixed total tuning budget, walk through how you'd allocate it across the three stages, using this lesson's reasoning.