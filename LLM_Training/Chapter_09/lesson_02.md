# Chapter 9 · Lesson 2 — DPO: Derivation From the RLHF Objective, Practical Implementation

> **Where this fits:** Directly motivated by Lesson 1, Section 5's complexity problem. DPO (Direct Preference Optimization) is the most consequential simplification of the RLHF pipeline — worth understanding its derivation precisely, since "DPO skips the reward model" is the shallow version of the answer, and the derivation is what makes the deeper version credible.

---

## 1. The Key Insight: The RLHF Objective Has a Closed-Form Solution

This is the mathematical heart of DPO, worth stating precisely rather than hand-waved: for a KL-constrained reward maximization objective of the exact form Lesson 1, Section 4 introduced (`maximize E[reward] - β·KL(policy || reference)`), there is a known **closed-form expression** for the optimal policy in terms of the reward function:

```
π*(response | prompt) = (1/Z(prompt)) * π_reference(response | prompt) * exp(reward(prompt, response) / β)
```

Where `Z(prompt)` is a normalization constant (a partition function, ensuring the distribution sums to 1). **This equation says: the optimal aligned policy is the reference policy's distribution, reweighted by how much reward each response gets, tempered by `β`.**

---

## 2. Rearranging to Express Reward in Terms of the Policy

The DPO derivation's key algebraic move: the closed-form solution above can be **rearranged to solve for the reward function** in terms of the optimal policy and the reference policy:

```
reward(prompt, response) = β * log(π*(response | prompt) / π_reference(response | prompt)) + log(Z(prompt))
```

**Why this rearrangement is the whole trick:** it expresses the reward implicitly as a function of the policy itself, rather than as a separately-trained, separate reward model. Substituting this expression for `reward(...)` back into Lesson 1, Section 3's Bradley-Terry pairwise preference loss — and noting that the `log(Z(prompt))` normalization term **cancels out** in the pairwise difference (since it depends only on the prompt, not which response, and appears identically for both chosen and rejected) — produces a loss function that can be optimized **directly on the policy model**, with no separate reward model and no RL loop at all.

---

## 3. The DPO Loss, Derived

```
DPO loss = -log(sigmoid(
    β * [log(π_policy(chosen|prompt)/π_ref(chosen|prompt)) - log(π_policy(rejected|prompt)/π_ref(rejected|prompt))]
))
```

**Reading this against Lesson 1's reward-model loss for direct comparison:** structurally nearly identical to the Bradley-Terry pairwise loss from Lesson 1, Section 3 — except where that loss compared a separate reward model's *scores*, DPO's loss compares **log-probability ratios between the policy and reference model**, directly. The policy model is simultaneously playing the role of "the thing being trained" and "the implicit reward model," which is precisely what eliminates the need for Lesson 1's separate reward-model training stage and its Stage-3 RL loop entirely.

```python
import torch.nn.functional as F

def dpo_loss(policy_model, reference_model, prompt, chosen, rejected, beta=0.1):
    # Log probabilities under the policy being trained
    policy_chosen_logp = compute_sequence_logprob(policy_model, prompt, chosen)
    policy_rejected_logp = compute_sequence_logprob(policy_model, prompt, rejected)

    # Log probabilities under the FROZEN reference model (typically the SFT checkpoint)
    with torch.no_grad():
        ref_chosen_logp = compute_sequence_logprob(reference_model, prompt, chosen)
        ref_rejected_logp = compute_sequence_logprob(reference_model, prompt, rejected)

    chosen_reward_implicit = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward_implicit = beta * (policy_rejected_logp - ref_rejected_logp)

    return -F.logsigmoid(chosen_reward_implicit - rejected_reward_implicit)
```

**Why the reference model is still needed, despite DPO "skipping the reward model" — a precision point worth stating explicitly:** DPO eliminates the separate *reward model*, not the reference model — the frozen reference policy (typically the SFT checkpoint, same role as in Lesson 1's PPO KL penalty) is still required, since the loss is defined in terms of the policy's behavior *relative to* that reference, not in absolute terms. This is a common imprecision in casual explanations of DPO worth correcting.

---

## 4. What DPO Actually Simplifies, and What It Doesn't

Directly connecting back to Lesson 1, Section 5's complexity list:

| | RLHF (Lesson 1) | DPO |
|---|---|---|
| Separate reward model needed? | Yes | No — implicit in the policy/reference log-ratio |
| Separate value/critic model needed? | Yes (standard PPO component) | No |
| RL training loop (sampling, PPO's clipped updates)? | Yes | No — a supervised-learning-style loss on fixed preference pairs, closer in engineering complexity to Chapter 7's SFT loop |
| Reference model still required? | Yes (for KL penalty) | Yes (built into the loss directly) |
| Preference data required? | Yes (to train the reward model) | Yes (used directly as DPO's training pairs) |

**The practical upshot:** DPO removes RLHF's most operationally burdensome pieces (separate reward model training, the RL sampling/optimization loop, the critic model) while requiring the *same underlying preference data* — this is why DPO is often described as making RLHF's benefits accessible with an SFT-like training loop's engineering simplicity.

---

## 5. Diagnosis & Mental Models: When DPO's Simplification Has a Real Cost

Worth naming honestly, not just presenting DPO as a strictly-better replacement: DPO's implicit reward is entirely defined by the fixed, offline preference dataset — it never generates and scores *new* responses during training the way PPO's RL loop does (Lesson 1, Section 4). This means **DPO cannot benefit from online exploration** — discovering that some response style not represented in the original preference data would score even better — the way an RL-based method in principle can. For a real production alignment problem where the fixed preference dataset may not fully anticipate every response pattern the policy might drift toward, this is a genuine, non-hypothetical tradeoff, not just a theoretical footnote — worth citing if asked whether DPO is a strictly superior choice.

---

## Key Takeaways

- DPO's derivation starts from the closed-form solution to the exact KL-constrained objective PPO is trying to solve numerically, then algebraically substitutes that solution back into the preference loss — this is why it's called "direct" preference optimization.
- The key algebraic move is expressing reward implicitly via the policy/reference log-probability ratio, which lets the partition function cancel out in the pairwise loss.
- DPO eliminates the separate reward model, critic model, and RL sampling loop, but still requires a frozen reference model and the same preference data RLHF would use.
- DPO's offline nature (no online exploration/generation during training) is a genuine tradeoff against PPO's online RL approach, not a strictly dominant simplification.

---

## Self-Check Before Moving to Lesson 3

1. Walk through the derivation from the closed-form optimal policy to the DPO loss, explaining why the partition function cancels out.
2. What specifically does DPO eliminate from Lesson 1's pipeline, and what does it still require?
3. Explain the online-vs-offline tradeoff between PPO and DPO, and why it's a genuine cost, not just a theoretical concern.