# Chapter 9 · Lesson 3 — GRPO and Other PPO-Alternatives: RLOO, ReMax

> **Where this fits:** Lesson 2 covered DPO's offline simplification. This lesson covers a different family of simplifications — methods that stay in the online RL paradigm (unlike DPO) but simplify PPO's machinery specifically, most notably by removing the separate critic/value model Lesson 1 flagged as part of the infrastructure burden.

---

## 1. What These Methods Have in Common: Removing the Critic

Recall Lesson 1, Section 5: standard PPO requires a value/critic model, trained alongside the policy, to estimate expected future reward and reduce gradient variance in the policy update. This critic model is itself a substantial piece of infrastructure — typically another full copy of a large model, adding meaningfully to both memory footprint and training complexity. GRPO, RLOO, and ReMax are all, at their core, different answers to the same question: **can we get PPO's variance-reduction benefit without training a separate critic model?**

---

## 2. GRPO (Group Relative Policy Optimization)

**The core mechanism:** instead of a learned critic estimating expected reward, GRPO generates **multiple responses to the same prompt** (a "group"), and uses the **group's own mean reward as the baseline** for computing each response's relative advantage — directly replacing a learned value function with a simple, computed statistic from the group itself.

```
For a prompt, sample a group of G responses: r_1, r_2, ..., r_G
Compute reward for each: R_1, R_2, ..., R_G
Group baseline: R_mean = mean(R_1, ..., R_G)
Advantage for response i: A_i = (R_i - R_mean) / std(R_1, ..., R_G)
```

**Why this works as a variance-reduction technique, precisely:** the whole point of PPO's critic was providing a baseline to subtract from raw reward, so the policy update is driven by *how much better or worse this response was than expected*, not raw reward magnitude (which can vary a lot per-prompt for reasons unrelated to response quality, adding noise to the gradient signal). GRPO's group mean serves exactly this baseline role, computed directly from actual sampled outcomes rather than learned/estimated — cheaper, and grounded in real observed rewards for this specific prompt rather than a potentially-imperfect learned estimate.

```python
def grpo_advantages(rewards):
    """rewards: list of reward values for a group of responses to the same prompt"""
    mean_r = sum(rewards) / len(rewards)
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return [(r - mean_r) / (std_r + 1e-8) for r in rewards]
```

**Why GRPO has become particularly associated with reasoning-focused training (worth knowing as context, connecting to Chapter 5, Lesson 5):** GRPO's group-sampling structure is a natural fit for tasks with a clear, checkable correctness signal (e.g., math problems with verifiable answers) — sampling multiple attempts at the same problem and using relative performance within that group as the training signal maps cleanly onto "did this particular reasoning attempt do better or worse than other attempts at the same problem," which is part of why GRPO has seen substantial use in recent reasoning-model training approaches.

---

## 3. RLOO (REINFORCE Leave-One-Out)

**The core mechanism:** conceptually similar to GRPO's group-based baseline, but the baseline for each response is computed by **excluding that response itself** from the group average — a "leave-one-out" estimate, a well-established variance-reduction technique from classical statistics, applied here to policy gradient estimation.

```
For response i in a group of G: baseline_i = mean(R_1, ..., R_G excluding R_i)
Advantage_i = R_i - baseline_i
```

**Why leave-one-out rather than the full group mean (GRPO's approach) — the subtle but real distinction:** using the full group mean as every response's baseline means each response's own reward contributes to its own baseline, which introduces a small bias (a response with an unusually high reward pulls its own baseline up, slightly understating how good it actually was relative to a "fair" baseline). Leave-one-out removes this self-contribution, producing an unbiased baseline estimate at the cost of a marginally more involved computation — a real, if second-order, methodological refinement worth knowing by name if a follow-up asks "how is RLOO different from GRPO specifically."

---

## 4. ReMax

**The core mechanism:** uses a **single greedy (highest-probability) response** as the baseline, rather than a sampled group's statistics — compare each sampled response's reward against the reward of what the policy would have produced with greedy decoding for that same prompt.

```
For a prompt: sample response r (stochastically) AND compute r_greedy (greedy decoding)
Advantage = R(r) - R(r_greedy)
```

**The tradeoff versus GRPO/RLOO's group-sampling approach:** ReMax requires generating only one additional response (the greedy one) per prompt rather than a full group of G samples, making it cheaper per training step — at the cost of a noisier, less statistically robust baseline than an average over multiple samples, since a single greedy decode is one specific point estimate rather than a distribution-informed average.

---

## 5. Comparison Table — the Actual Decision-Relevant Differences

| Method | Baseline source | Relative cost per step | Best fit |
|---|---|---|---|
| Standard PPO (Lesson 1) | Learned critic model | Highest — separate model to train and maintain | When a well-trained critic genuinely improves variance reduction enough to justify its cost |
| GRPO | Group mean of sampled responses | Moderate — requires generating a group (G samples) per prompt | Tasks with clear, verifiable correctness (math, code) where group sampling is cheap relative to the value of the signal |
| RLOO | Leave-one-out group mean | Moderate, similar to GRPO | Same general fit as GRPO, with a statistically cleaner (unbiased) baseline at essentially the same cost |
| ReMax | Single greedy decode | Lowest among the online alternatives | When per-step cost matters more than baseline precision, or group sampling is impractically expensive |

**The unifying theme worth stating in an interview:** all three (GRPO, RLOO, ReMax) are answers to the same underlying question — how to get a usable advantage/baseline signal without a separately-trained critic model — differing only in exactly how that baseline is computed from sampled data. This framing (shared problem, different specific solutions) is a stronger answer than listing the three as unrelated named algorithms.

---

## Key Takeaways

- GRPO, RLOO, and ReMax all remove PPO's separate critic model, replacing a learned value estimate with a baseline computed directly from sampled responses.
- GRPO uses the full group mean as baseline; RLOO uses a leave-one-out mean for an unbiased estimate; ReMax uses a single greedy decode for the cheapest possible baseline.
- GRPO's structure is a particularly natural fit for verifiable-correctness tasks like math/code reasoning, connecting directly to why it's seen substantial recent use in reasoning-model training.
- The right choice among these three is a cost/statistical-robustness tradeoff, not a strict quality ranking — framing them as different answers to one shared problem is the stronger interview-level understanding.

---

## Self-Check Before Moving to Lesson 4

1. Explain, precisely, why removing a learned critic model and replacing it with a group-based baseline still provides useful variance reduction.
2. What specific bias does RLOO's leave-one-out approach correct for, relative to GRPO's full-group-mean approach?
3. Why is GRPO particularly well-suited to math/code reasoning tasks specifically, connecting to Chapter 5's reasoning-capability content?
4. If per-step training cost were the binding constraint, which of these three methods would you reach for first, and why?