# Chapter 3 · Lesson 6 — Batch Size, Tokens-Per-Step, and the Linear Scaling Rule

> **Where this fits:** Lesson 4 showed *how* to reach a large effective batch size (accumulation) without running out of memory. This lesson is about *why* batch size matters, how it connects to the learning rate from Lesson 5, and how to reason about it when scaling up or down.

---

## 1. Batch Size Is a Statistical Estimate, Not Just a Memory Knob

Each training step computes a gradient that's an *estimate* of the true gradient over the entire data distribution, based on the batch sampled. A larger batch size gives a **less noisy** estimate of that true gradient — averaging over more samples reduces variance, by basic statistics (variance of a mean shrinks proportionally to `1/batch_size`).

This is the actual reason batch size and learning rate are linked, not just an empirical coincidence — it's worth being able to derive this connection rather than just quoting the rule in Section 2.

---

## 2. The Linear Scaling Rule

**The rule:** if you multiply the batch size by `k`, multiply the learning rate by `k` as well (within a reasonable range), to keep training behavior roughly consistent.

**Why, connected directly to Section 1:** a larger batch gives a less noisy (lower-variance) gradient estimate. A less noisy gradient can tolerate a *larger* step size without the noise causing instability — so scaling the learning rate up alongside the batch size roughly preserves the "signal-to-step-size" balance that made the original, smaller-batch learning rate work.

```
Original: batch_size = 256,  peak_lr = 3e-4
Scaled:   batch_size = 1024  (4x)  →  peak_lr ≈ 1.2e-3  (4x)
```

**The important caveat, worth stating unprompted:** this rule holds well within a moderate range, but breaks down at very large batch sizes — beyond some point (model- and data-dependent, but often in the range of hundreds of thousands to a few million tokens per step for large LLMs), further increasing batch size no longer meaningfully reduces gradient noise relative to the underlying signal, and naively continuing to scale the learning rate linearly causes instability rather than helping. This is sometimes discussed as approaching a "critical batch size" beyond which additional data-parallelism buys diminishing returns on convergence speed per token, even though it still buys wall-clock speed via parallelism.

---

## 3. Tokens-Per-Step — The Metric That Actually Matters at Scale

For LLM pretraining specifically, "batch size" in the traditional sense (number of sequences) is less useful than **tokens per step** — `batch_size × sequence_length` — because sequence length varies across training setups and datasets, but the actual quantity that determines gradient-estimate quality (Section 1) and compute cost is the total number of tokens contributing to the gradient.

```
batch_size = 512 sequences, sequence_length = 4096 tokens
tokens_per_step = 512 * 4096 = 2,097,152 tokens (~2M tokens/step)
```

Published training recipes (LLaMA, GPT-3, etc.) report figures in this form — "trained with ~4M tokens per batch" — specifically because it's the metric that's actually comparable across different sequence-length choices, and it's the number that connects most directly to the compute-optimal scaling laws covered in Lesson 7.

---

## 4. Worked Example: Reasoning About a Batch Size Change End to End

Say you're scaling from a 7B-parameter training run to a 70B-parameter run, and you also want to increase GPU count 8x for more parallelism (Lesson 3). Walking the full chain of reasoning an interviewer wants to see:

1. More GPUs (Lesson 3's DP) → can process a larger batch per step at the same wall-clock cost per step.
2. Larger batch (Section 1) → lower-variance gradient estimate.
3. Lower-variance gradient (Section 2) → can tolerate a proportionally larger learning rate via linear scaling.
4. But — check against Section 2's caveat: is the new tokens-per-step figure (Section 3) still comfortably below the critical-batch-size range for a model this size? If the scale-up is large enough to approach that range, linear scaling of learning rate will *not* hold cleanly, and empirical validation (small-scale LR sweep before committing the full run — this connects to Chapter 4's hyperparameter-transfer methods) becomes necessary rather than optional.

This four-step chain — not just "bigger batch, bigger learning rate" — is what a senior answer to "how would you adjust hyperparameters if we 8x'd our GPU count" actually sounds like.

---

## 5. Code: Deriving Effective Batch Size and Checking Against a Target

```python
def compute_training_config(gpus, per_gpu_batch_size, seq_len, accumulation_steps=1):
    effective_sequences = gpus * per_gpu_batch_size * accumulation_steps
    tokens_per_step = effective_sequences * seq_len
    return {
        "effective_sequences_per_step": effective_sequences,
        "tokens_per_step": tokens_per_step,
    }

# Example: 64 GPUs, 8 sequences per GPU fits in memory, 4096-token sequences,
# accumulate 2 micro-batches to hit a larger target
config = compute_training_config(gpus=64, per_gpu_batch_size=8, seq_len=4096, accumulation_steps=2)
print(config)
# {'effective_sequences_per_step': 1024, 'tokens_per_step': 4,194,304}
```

This is the kind of number worth being able to compute on the spot in an interview when asked "what's your effective batch size" — being able to derive it from GPU count × per-GPU batch × accumulation steps × sequence length, live, is a strong signal versus quoting a memorized number.

---

## 6. Diagnosis: Batch-Size-Related Symptoms

- **Increasing batch size (with linearly scaled LR) makes training less stable, not more** → likely approaching or past the critical batch size (Section 2's caveat) — a batch-size increase isn't universally safe to pair with proportional LR scaling.
- **Training is very slow to converge despite a reasonable learning rate** → check tokens-per-step (Section 3) is actually adequate for the model size; too small an effective batch, especially combined with a schedule (Lesson 5) tuned for a larger one, produces noisy, slow progress.
- **Same nominal "batch size" (sequence count) but very different training behavior across two runs** → check sequence length — Section 3's point that tokens-per-step, not sequence count alone, is the metric that actually matters.

---

## Key Takeaways

- Batch size controls gradient-estimate variance — this statistical fact is the actual justification for the linear scaling rule, not just an empirical convention.
- The linear scaling rule (LR scales with batch size) holds in a moderate range but breaks down near a model- and data-dependent critical batch size.
- Tokens-per-step (batch_size × sequence_length), not raw sequence count, is the metric that's actually comparable across setups and connects to scaling laws.
- Scaling GPU count → batch size → learning rate is a full chain of reasoning, not a single rule to apply blindly — always check against the critical-batch-size caveat.

---

## Self-Check Before Moving to Lesson 7

1. Derive, don't just state, why a larger batch size can tolerate a larger learning rate.
2. What's the actual failure mode of applying the linear scaling rule far beyond the critical batch size?
3. Two training runs both use "batch size 512" but behave very differently. What's the first question you'd ask to explain the discrepancy?
4. Compute tokens-per-step for: 32 GPUs, 4 sequences per GPU, sequence length 8192, accumulation steps 4.