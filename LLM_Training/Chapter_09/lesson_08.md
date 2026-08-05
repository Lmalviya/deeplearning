# Chapter 9 · Lesson 8 — Interview Lab: Choosing an Alignment Method Under Constraints

> **Where this fits:** This closes out the chapter with the live, comparative-reasoning question interviewers actually ask — not "explain RLHF" in isolation, but "given these constraints, which method would you pick, and why." This draws on every lesson in the chapter simultaneously, the same synthesis structure as Chapter 7, Lesson 10.

---

## 1. Why This Question Format Is Common and What It's Really Testing

An interviewer asking "which alignment method would you choose given X constraints" is testing whether the comparison table knowledge from Lessons 1-3 (RLHF/PPO vs. DPO vs. GRPO/RLOO/ReMax) can actually be applied under a specific, concrete constraint set — compute budget, team size/expertise, data availability, task type (verifiable-correctness vs. open-ended) — rather than recited as abstract facts about each method.

---

## 2. The Master Comparison, Organized for Live Retrieval

| Constraint | Favors |
|---|---|
| Limited engineering capacity for RL infrastructure (no separate critic/reward-model maintenance desired) | DPO (Lesson 2) — SFT-like training loop simplicity |
| Task has clear, verifiable correctness signal (math, code) | GRPO (Lesson 3) — group-sampling structure is a natural fit |
| Need online exploration — offline preference dataset may not anticipate policy drift | RLHF/PPO or GRPO/RLOO/ReMax (online methods) over DPO (offline) |
| Human preference data is expensive/slow to collect at needed scale | RLAIF/Constitutional AI (Lesson 5) to supplement or substitute |
| Safety/refusal calibration specifically is the target (Chapter 5, Lesson 10) | Favor methods with strong, deliberate control over preference-data balance (careful reward model training, Lesson 4) — human data likely still preferred here per Lesson 5, Section 4's circularity-risk concern |
| Per-step training cost is the binding constraint among online methods | ReMax (Lesson 3) — cheapest baseline computation among the critic-free alternatives |
| High confidence in reward model quality, want to squeeze maximum improvement | Lower β (Lesson 6) with any online method; higher reward-hacking vigilance required (Lesson 4/7) |

---

## 3. A Full Worked Response to a Realistic Prompt

**Prompt:** *"You're aligning a mid-size model for a customer-support use case, focused on safety calibration and helpfulness. You have a small ML team, limited RL infrastructure experience, and a moderate budget for human annotation. Which method would you choose?"*

> "Given limited RL infrastructure experience specifically, I'd lean away from full PPO — maintaining a separate reward model, critic model, and an actual RL training loop is a meaningfully heavier engineering lift than this team's stated experience suggests they're set up for, and getting that infrastructure subtly wrong is a real risk given how sensitive RL training is to implementation details.
>
> I'd choose DPO instead — it needs the same underlying preference data RLHF would use, but the training loop itself is much closer to standard supervised fine-tuning, which this team already has experience with from earlier stages. The tradeoff I'd flag explicitly is that DPO is offline — it won't adapt to policy drift during training the way an online method would — but for a safety-calibration-focused use case specifically, I'd actually consider that acceptable, since I'd rather have tighter, more predictable control grounded in carefully-curated preference data than open-ended online exploration for exactly the kind of over-refusal/under-refusal calibration Chapter 5 covered.
>
> Given the moderate annotation budget, I'd prioritize human-collected preference data specifically for the safety-calibration portion — both over-refusal and under-refusal examples deliberately, not just one direction — rather than leaning on RLAIF for that specific category, given the circularity risk of an AI judge sharing the same model family's blind spots on exactly the judgment that matters most here. I might use RLAIF to supplement volume for the broader helpfulness preferences, where the stakes of judge bias are lower.
>
> For the KL penalty, I'd start conservative — a higher β — given this is a first alignment run for this team, and validate against independent quality checks and the full Chapter 5 capability suite before considering loosening it, rather than tuning aggressively toward maximum reward-model score."

---

## 4. Why This Response Structure Works

It doesn't just name DPO — it walks through the *specific constraints* (limited RL experience, safety focus, moderate but not unlimited annotation budget) and connects each one to a specific piece of reasoning from Lessons 1-6, then closes with a concrete hyperparameter stance (conservative β) tied to Lesson 6/7's diagnostic discipline. This is the direct analogue of Chapter 7, Lesson 10's "narrate the flowchart, don't just state a conclusion" structure, applied to alignment method selection specifically.

---

## 5. Follow-Up Questions to Have Pre-Loaded

**"What would change your answer if this were a coding assistant instead of customer support?"** → Direct pivot to GRPO (Lesson 3, Section 2) — a coding assistant has genuinely verifiable correctness (tests passing, code executing correctly), which is exactly the structure GRPO's group-sampling approach is well-suited to, potentially outweighing the infrastructure-simplicity argument for DPO in this different context.

**"How would you detect if this DPO run had a reward-hacking-like problem, given DPO doesn't have an explicit reward model to monitor?"** → A genuinely good test of whether Lesson 2's derivation was actually understood: DPO's implicit reward is the policy/reference log-probability ratio (Lesson 2, Section 3) — the same detection principles from Lesson 4 still apply, just computed differently; monitoring this implicit reward's trend against independent quality checks, and watching for Lesson 7's specific patterns (length, sycophancy, refusal exploitation) directly in the policy's outputs, serves the same diagnostic role even without a separate reward model object to inspect.

**"Your chosen approach shows alignment tax on the model's reasoning capability — what's your next move?"** → Directly reuse Lesson 7, Section 4's framing: first, quantify the tax concretely against the full Chapter 5 reasoning-capability suite, then make an explicit judgment call about whether the safety-calibration gain justifies the reasoning-capability cost — not automatically assuming the alignment run needs to be redone, since some tax may be an acceptable, conscious tradeoff.

**"Why not just use RLAIF for everything, given the cost savings?"** → Direct callback to Lesson 5, Section 4's limits — circularity risk specifically matters most for exactly the categories (safety, nuanced value judgments) where getting it wrong is costliest, which is precisely why the worked response in Section 3 reserved human data for that category specifically rather than applying RLAIF uniformly.

---

## Key Takeaways

- This question format tests applied comparative reasoning under specific constraints, not recitation of each method's mechanics in isolation.
- A strong response explicitly connects each stated constraint (team experience, task type, budget, safety-criticality) to a specific piece of reasoning from earlier lessons, rather than naming a method and justifying it generically.
- Closing with a concrete hyperparameter stance (not just a method name) demonstrates the response is grounded in Lesson 6/7's tuning and diagnostic discipline, not just method selection in isolation.
- Follow-ups typically probe whether the reasoning transfers to a different task type, a different failure mode, or a challenge to the chosen approach's biggest weakness — all four in Section 5 are worth having ready before walking into this kind of question live.

---

## Self-Check — Full Mock Rep

Construct your own constraint scenario (different task, team, and budget profile) and produce a full worked response in the style of Section 3. Then have someone (or a future session with me) fire the four follow-ups from Section 5, adapted to your scenario, and try defending or revising your choice under that pressure.