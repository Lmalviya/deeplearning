# Chapter 8 · Lesson 7 — Interview Lab: Defending Specific Fine-Tuning Hyperparameter Choices

> **Where this fits:** This closes the loop with Chapter 4, Lesson 7's interview lab (which covered the general "how did you tune hyperparameters" question using pretraining-adjacent reasoning). This lesson is the fine-tuning-specific version, now with an entire chapter of fine-tuning-specific reasoning to draw on — a meaningfully more detailed answer should be possible now than at Chapter 4.

---

## 1. Why This Question Resurfaces, and What's Different Now

The same underlying question from Chapter 4, Lesson 7 — but a strong candidate's answer should sound different post-Chapter-7/8 than it would have right after Chapter 4 alone. Chapter 4's version could reasonably rely on general tuning-workflow reasoning; this version should reflect fine-tuning-specific knowledge — the 10-100x LR relationship (Lesson 2), epoch/overfitting tradeoffs (Lesson 3), why direct search is often feasible here (Lesson 4), and early-stopping mechanics (Lesson 5). An interviewer asking this question after establishing you're discussing a fine-tune specifically (not general pretraining) is testing for exactly this added specificity.

---

## 2. A Full Worked Answer, Reusing Chapter 7 Lesson 10's Scenario

**Prompt, continuing directly from Chapter 7, Lesson 10's scenario:** *"You chose LoRA rank 16 for the structured-output fine-tune — walk me through how you'd actually tune the full hyperparameter set, not just rank."*

> "I'd start from reasoned defaults rather than searching blind. For learning rate, given this is LoRA fine-tuning a 7B model, I'd anchor to roughly a 10-100x reduction from a typical pretraining peak LR for this model size — so somewhere in the 1e-5 to 3e-5 range as a starting point, leaning toward the higher end of that range since LoRA's constrained parameter count already limits catastrophic forgetting risk compared to full fine-tuning.
>
> For epochs, given this is a curated dataset in the low thousands of examples, I'd start with a moderate default — around 3 epochs — but I wouldn't commit to that number in advance. I'd monitor validation loss every epoch and use early stopping with a patience of 2, returning the best-validation checkpoint rather than whatever the final epoch produces, since fine-tuning on a small dataset makes overfitting a real risk if I ran the full planned epoch count regardless of what the validation curve showed.
>
> For the search itself, since this is a LoRA fine-tune, individual runs are cheap enough that I'd do a direct random search over LR and rank jointly, maybe 10-15 short runs, using an ASHA-style triage — killing clearly underperforming configurations after a small fraction of an epoch rather than running every candidate to completion, since fine-tuning's short total run length makes early trajectory a reasonably reliable predictor of final performance.
>
> For alpha, I'd keep it tied to rank at a fixed ratio, commonly 2x rank, rather than tuning it as a fully independent hyperparameter, since that ratio is what actually controls the update's effective magnitude — tuning alpha and rank as if they were unrelated would be redundant search over what's substantially the same underlying effect."

---

## 3. Why This Answer Is Stronger Than the Chapter 4 Version Would Have Been

Explicit comparison, since this is the actual point of the lesson: Chapter 4's version of this answer (Lesson 7 there) was necessarily more generic — "I'd fix low-sensitivity hyperparameters, derive some from constraints, sweep LR carefully." This version names **specific fine-tuning-specific numbers and reasoning** (the 10-100x relationship, why LoRA's forgetting-risk profile affects where in that range to land, why early stopping's patience mechanism matters given small-dataset overfitting risk, why alpha-as-ratio rather than independent search) — the kind of specificity that only comes from the fine-tuning-specific content this chapter added on top of Chapter 4's general framework.

---

## 4. Compressed Version for Time Pressure

> "I'd anchor learning rate to roughly 10-100x below typical pretraining LR for this model size, leaning higher given LoRA's lower forgetting risk. Epochs would start at a moderate default but be governed by early stopping with patience, not a fixed count, given small-dataset overfitting risk. Since LoRA runs are cheap, I'd do direct random search with ASHA-style early triage over LR and rank jointly, rather than needing pretraining-style proxy-model transfer. Alpha stays tied to rank at a fixed ratio rather than searched independently."

---

## 5. Follow-Up Questions to Have Pre-Loaded

**"Why lean toward the higher end of the LR range for LoRA specifically, rather than always playing it safe at the low end?"** → Direct callback to Lesson 2, Section 3: LoRA's constrained parameter count structurally limits how much damage an aggressive LR can do relative to full fine-tuning, so the "always be conservative" instinct is less necessary here — worth stating as a reasoned tradeoff, not just "higher LR is faster."

**"What would make you choose Bayesian optimization instead of random search here?"** → Honest, calibrated answer per Chapter 4, Lesson 3: if each run were meaningfully more expensive (e.g., this were full fine-tuning of a much larger model rather than cheap LoRA runs), the value of using every result to inform the next choice would outweigh Bayesian optimization's implementation overhead — for cheap LoRA runs specifically, random search's simplicity is usually good enough, per Lesson 4 of this chapter's reasoning about fine-tuning's generally favorable cost structure.

**"Your ASHA triage kills a configuration early that would have actually recovered and performed well with more steps — how do you guard against that risk?"** → A genuine, honest limitation to acknowledge: early triage trades some risk of prematurely discarding a late-blooming configuration for substantially reduced total search cost — a reasonable mitigation is not triaging too aggressively in the first round (keeping a slightly more generous survival fraction early, per Chapter 4 Lesson 3's Hyperband structure) rather than assuming the risk away entirely.

**"How does your approach change if this were DAPT instead of LoRA fine-tuning?"** → Direct callback to Lesson 2, Section 4 and Chapter 7, Lesson 1: DAPT sits closer to pretraining's cost and risk profile, so the LR wouldn't follow the same 10-100x fine-tuning reduction, and given DAPT's much higher per-run cost, Chapter 4's proxy-model transfer logic becomes relevant again rather than Lesson 4 of this chapter's direct-search default.

---

## Key Takeaways

- This question, asked after a fine-tuning-specific context has been established, expects fine-tuning-specific reasoning (the 10-100x LR relationship, epoch/overfitting tradeoffs, alpha-as-ratio) — not a repeat of Chapter 4's more general answer.
- A strong answer explicitly reasons through each hyperparameter's starting point and how it would actually be validated (search method, early stopping), not just naming final values.
- Follow-up questions test whether the reasoning generalizes — to different cost regimes (Bayesian vs. random search), different risk tolerances (ASHA's early-triage risk), and different training-stage contexts (DAPT vs. fine-tuning) — having each of these adaptations ready is what separates memorized numbers from portable understanding.

---

## Self-Check — Full Mock Rep

Say the full version (Section 2) out loud, targeting 90-120 seconds, then the compressed version (Section 4), targeting 30 seconds. Then have someone (or a future session with me) fire the four follow-ups from Section 5 in random order, and try adapting your answer to a hypothetical fifth scenario they invent on the spot.