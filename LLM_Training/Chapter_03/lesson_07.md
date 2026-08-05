# Chapter 3 · Lesson 7 — Scaling Laws: Chinchilla and Compute-Optimal Training

> **Where this fits:** Lesson 6 ended with tokens-per-step as the key metric. This lesson zooms out to the whole training run: given a fixed compute budget, how large should the model be, and how much data should it see? This is one of the most commonly asked "do you actually understand the field" questions in senior ML interviews.

---

## 1. The Question Scaling Laws Answer

Given a fixed compute budget `C` (measured in FLOPs), there's a tradeoff: spend it on a bigger model with less data, or a smaller model with more data. Scaling laws are empirical relationships, fit by training many models at different sizes and data amounts, that tell you how loss behaves as a function of model size `N`, dataset size `D`, and compute `C` — and from that, what the *optimal* split between `N` and `D` is for a given `C`.

---

## 2. The Kaplan et al. (2020) Finding, and Why It Was Later Revised

The original OpenAI scaling laws paper found that, for a fixed compute budget, loss improvements came predominantly from increasing model size, with data size playing a comparatively smaller role — this finding is part of why the field's early scaling trend (GPT-3 and contemporaries) favored very large models trained on comparatively modest amounts of data relative to their parameter count.

**What Chinchilla (Hoffmann et al., 2022) found differently:** re-running scaling experiments more carefully — notably, tuning the learning rate schedule properly for each individual training run rather than reusing one schedule across differently-sized experiments (a subtle but important methodological fix) — found that the earlier conclusion was skewed by that methodology issue, and that **model size and data size should actually scale roughly equally** for compute-optimal training, not model-size-dominant.

**The concrete, famous result:** Chinchilla (70B parameters, trained on 1.4 trillion tokens) outperformed Gopher (280B parameters, trained on only ~300 billion tokens) despite being 4x smaller — because Gopher was significantly **undertrained** relative to its parameter count for the compute that was spent on it. This is the single most citable, concrete fact in this lesson — know the actual numbers, not just "Chinchilla showed data matters too."

---

## 3. The Chinchilla Rule of Thumb

The commonly cited practical takeaway from the paper: compute-optimal training uses roughly **20 tokens per parameter**.

```
7B parameter model  → ~140B tokens for compute-optimal training
70B parameter model → ~1.4T tokens for compute-optimal training
```

**Important nuance worth stating unprompted — this is where a lot of surface-level answers stop short:** "compute-optimal" specifically means minimizing *training* loss for a given *training* compute budget. It does **not** account for inference cost. A smaller model trained on *more* than the Chinchilla-optimal token count (an "overtrained" model, in this specific technical sense) will have higher training compute cost than strictly optimal, but will be **cheaper to serve at inference** — and since most deployed models are queried far more times than they're trained once, many production models (LLaMA models are a well-known public example) are deliberately trained beyond the Chinchilla-optimal point, trading some training-compute efficiency for meaningfully lower inference cost over the model's deployed lifetime.

This distinction — training-compute-optimal vs. total-cost-optimal-including-inference — is exactly the kind of nuance that separates "I read the Chinchilla abstract" from "I understand why LLaMA didn't just follow Chinchilla's ratio literally."

---

## 4. The Actual Functional Form (Enough to Reason With, Not Full Derivation)

Loss as a function of model size and data size is commonly modeled as a sum of three terms:

```
L(N, D) = E + A/N^α + B/D^β
```

Where `E` is an irreducible loss floor (entropy of natural language itself — no model, however large, gets below this), and the other two terms are the loss contribution from finite model size and finite data size respectively, each shrinking as a power law as `N` or `D` grows.

**The intuition, not the calculus:** for a fixed compute budget `C ≈ 6ND` (a commonly used approximation for training FLOPs, since roughly 6 FLOPs are needed per parameter per token for the forward+backward pass), there's an optimal split of `C` between `N` and `D` that minimizes `L(N, D)` — and solving that optimization is what produces the "roughly equal scaling, ~20 tokens/parameter" result from Section 3.

---

## 5. Worked Example: Sizing a Model for a Given Compute Budget

Say you have a budget of `10^23` FLOPs available for a training run (a realistic-scale example). Using `C ≈ 6ND` and the Chinchilla-optimal ratio `D ≈ 20N`:

```
C = 6 * N * D = 6 * N * 20N = 120 * N²
N² = C / 120 = 10^23 / 120 ≈ 8.3 × 10^20
N ≈ sqrt(8.3 × 10^20) ≈ 2.9 × 10^10  ≈ 29 billion parameters
D ≈ 20 * N ≈ 580 billion tokens
```

**This is a genuinely useful skill to demonstrate live in an interview** — being asked "given X compute, how would you size the model" and being able to work through this arithmetic on the spot, even approximately, is a much stronger signal than citing "Chinchilla says 20 tokens per parameter" as an isolated fact.

---

## 6. Practical Limits and Caveats Worth Naming

- **Data availability is a real, separate constraint.** The Chinchilla-optimal token count for very large models can exceed the amount of high-quality training data actually available (particularly once deduplication and quality filtering from Chapter 1 are applied) — this is part of the motivation behind more recent interest in synthetic data generation and more aggressive multi-epoch training on curated data.
- **Downstream task performance doesn't always track pretraining loss perfectly.** Scaling laws are fit against pretraining loss; a compute-optimal *pretrained* model isn't automatically compute-optimal for whatever downstream fine-tuning/deployment scenario you actually care about — Chapter 5 and 6's fine-tuning content is a separate optimization layer on top of this one.
- **The `C ≈ 6ND` approximation is architecture-dependent** — MoE models (Chapter 2, Lesson 4) break this simple relationship, since active parameters and total parameters diverge; scaling laws for MoE architectures are an active, less settled area, worth flagging as such rather than presenting dense-model scaling laws as universally applicable.

---

## Key Takeaways

- Scaling laws answer: for fixed compute, what's the optimal split between model size and data size?
- Chinchilla's key correction to earlier work: proper per-run learning-rate-schedule tuning revealed model and data size should scale roughly equally (~20 tokens/parameter), not model-size-dominant as earlier findings suggested.
- Chinchilla-optimal minimizes *training* compute for a given loss — it does not account for inference cost, which is why many production models are deliberately trained beyond that ratio.
- `C ≈ 6ND` lets you work backward from a compute budget to a recommended model size and data size — a concrete, derivable skill, not just a fact to recite.
- MoE architectures and inference-cost considerations are real caveats to a naive "just follow the 20-tokens-per-parameter rule" answer.

---

## Self-Check Before Moving to Lesson 8

1. State the concrete Chinchilla-vs-Gopher comparison (sizes and token counts) from memory — this is a very commonly asked specific fact.
2. Why does "compute-optimal" not automatically mean "the model you should actually train and deploy"? Name the real-world factor that changes the answer.
3. Given a compute budget of `6 × 10^22` FLOPs, work through the arithmetic (as in Section 5) to estimate a Chinchilla-optimal model size and token count.
4. Why does the `C ≈ 6ND` relationship become less straightforward to apply for an MoE model?