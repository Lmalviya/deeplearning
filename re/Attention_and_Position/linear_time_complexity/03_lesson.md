# Lesson 3 — Linformer: Self-Attention with Linear Complexity

**Paper:** Wang, Li, Khabsa, Fang, Ma — *"Linformer: Self-Attention with Linear Complexity"* (2020)
**Source:** https://arxiv.org/abs/2006.04768
**Family:** Low-rank projection (different mechanism from Lessons 1–2's kernel-trick family)

---

## 1. Core idea (one line)

Standard attention's N×N matrix doesn't carry N×N worth of independent information — it's approximately **low-rank**. So instead of computing the full N×N matrix, compress K and V down to a small fixed size `k` *before* attention, making the computation N×k instead of N×N.

---

## 2. The mapping matrix P

```
P = softmax(QKᵀ/√d)
```

Just the standard attention matrix (N×N), given a name — row i is a probability distribution over how much query i attends to each of the N keys. Same object as in Lesson 1, just named `P` instead of computed inline.

**Paper's claim:** P is *approximately low-rank* — most of its information is concentrated in a small number of dimensions (largest singular values); the rest contributes little. Justified via the Johnson–Lindenstrauss lemma (high-dimensional data can be projected to much lower dimension while approximately preserving distances) plus empirical evidence.

**Why this matters:** if P is low-rank, you don't need the full N×N matrix to reconstruct a good approximation of `P·V` — you can route the computation through a much smaller intermediate size k.

---

## 3. The mechanism — projections E and F

Two learned linear projections, applied along the **sequence-length axis** (unusual — normal linear layers project the feature axis):

```
E ∈ R^{k×N}   — compresses K
F ∈ R^{k×N}   — compresses V

K' = E K     shape: (k×N)(N×d) = k×d      [was N×d]
V' = F V     shape: (k×N)(N×d) = k×d      [was N×d]
```

Attention becomes:

```
Attention = softmax( Q K'ᵀ / √d ) V'
```

**Shapes:** `Q` stays N×d, `K'ᵀ` is d×k → `QK'ᵀ` is **N×k** (not N×N). Multiply by `V'` (k×d) → output is N×d, same as standard attention.

**Cost:** O(N·k·d) instead of O(N²·d). k is a fixed constant chosen ahead of time (doesn't grow with N) → **O(N)**, linear.

**What E, F actually are:** ordinary learned weight matrices, trained end-to-end like any other layer — no exotic math required to define or use them. Conceptually similar to compressing N key/value vectors into k learned "summary" vectors, the way PCA compresses correlated features into a few components (except here it's learned, not eigendecomposition, and it compresses the *sequence* dimension, not the *feature* dimension).

---

## 4. Why Q is never projected — only K and V

This is a structural requirement, not a design choice:

- The attention output must have **one row per query position** — that's what "the output representation for token i" means. Output shape has to stay N×d, matching the N input tokens, so every token still gets an updated representation.
- If Q were compressed from N×d down to k×d, the output of `softmax(Q'K'ᵀ)V'` would only have **k rows** — meaning only k output vectors for what was originally N input tokens. You'd lose the ability to produce a representation for every token; the model would only output a fixed number of positions regardless of input length. That breaks the basic transformer contract of "one output vector per input token."
- K and V don't have this constraint. Their row count only determines **how many things there are to attend to** — not how many outputs get produced. Shrinking N keys/values down to k "summary" keys/values just means each query now attends over a smaller set of pooled representations; it doesn't change how many queries there are or how many outputs come out.

**One-line summary:** Q's row count determines the number of outputs (must stay N); K/V's row count only determines the size of what's being attended over (safe to shrink to k).

---

## 5. Important limitation — encoder-only, not decoder/causal

Linformer's projections `E, F` are built assuming access to the **entire** key/value sequence at once (they project all N positions down to k in one shot). This is incompatible with **causal (autoregressive) attention**, where token i may only attend to tokens `j ≤ i` — at each generation step, the "full sequence" E and F expect isn't actually available yet, and the set of visible tokens keeps growing one at a time.

This is why the original paper builds on **RoBERTa**, a bidirectional **encoder-only** model with a fixed, fully-visible input — not a decoder-style generation setup. Applying Linformer's projection scheme directly to causal/autoregressive decoding isn't straightforward and isn't what the method was designed for.

---

## 6. Trade-offs to hold onto

| | |
|---|---|
| **Benefit** | O(N) time/space instead of O(N²); simple mechanism, ordinary learned linear layers |
| **Cost** | Fixed k caps how much can be attended to densely; may discard long-tail attention patterns needed for precise long-range single-token retrieval |
| **Constraint** | Assumes full sequence visibility → naturally suited to encoder-style (bidirectional, fixed-input) settings, not causal decoding |
| **vs. Lesson 1/2 (kernel methods)** | Different lever entirely — compresses the *sequence dimension* of K/V rather than approximating the *similarity function* itself |

---

## 7. Interview calibration

The mechanism above (P, E/F projections, shapes, complexity, why Q is untouched) is whiteboard-level material — reasonable to be asked to derive or explain.

The paper's theoretical justification for *why* attention is low-rank (Johnson–Lindenstrauss-based error bounds, singular value analysis) is a much heavier detour into linear algebra and concentration inequalities — generally **not** expected at a standard Amazon/Google-style ML interview depth. Understanding the mechanism and being able to discuss trade-offs against alternatives (kernel methods, sparse attention) covers the expected bar; reproducing the proof does not.

---

---

## 8. Why low-rank truncation loses "long-tail" information that sparse methods preserve

### What the "long tail" actually is

Any matrix — including the N×N attention matrix `P = softmax(QKᵀ/√d)` — can be decomposed (via SVD) into a sum of simpler rank-1 pieces, each weighted by a singular value:

```
P ≈ σ_1·u_1v_1ᵀ + σ_2·u_2v_2ᵀ + ... + σ_N·u_Nv_Nᵀ        (sorted σ_1 ≥ σ_2 ≥ ... ≥ σ_N)
```

Empirically, most of P's "energy" concentrates in the first few terms — this is exactly the observation Linformer leans on. Low-rank projection to size k is mathematically equivalent to **truncating this sum**, keeping only the top-k terms and discarding the rest (`σ_{k+1}...σ_N` — the **long tail**).

### Why discarding the tail can hurt

"Small singular value on average" does not mean "unimportant for every input." The long-tail components often correspond to **rare, sharp, highly specific attention patterns** — e.g. a single query attending strongly to one particular distant token — rather than the broad, smooth patterns shared across many tokens that dominate the top-k modes.

**Concrete example:** a pronoun late in a long document ("it," "she," "the company") resolving to one specific noun phrase mentioned once, far earlier. This is a sparse, sharp, one-to-one attention spike — individually small in aggregate "energy," so it doesn't move the top singular values, but critical for that specific query. The top-k modes tend to capture generic, broadly-shared structure (local attention, sentence-boundary patterns, coarse syntax); precise long-range one-off dependencies scatter into the smaller, truncated tail. Cutting to top-k discards exactly this kind of rare-but-critical signal.

This mirrors a pattern seen elsewhere in this series: aggregate-average approximations look fine on average but silently drop rare, high-precision signals that specific tasks depend on (same theme as ReLU feature maps discarding negative-negative correlations that mattered for fine detail in vision tasks — Lesson 1 territory).

### Why sparse-connectivity methods (Longformer-style) don't share this failure mode

The two approximation families make fundamentally different kinds of errors:

- **Low-rank (Linformer):** approximates *every* entry of P using a compressed, shared, lower-dimensional basis. No query-key pair is ever fully zeroed out, but rare/small signals get smoothed away by construction. **Error type: precision loss on rare, sharp deviations from the dominant pattern.**
- **Sparse (Longformer/BigBird):** computes **exact, full-precision softmax attention**, just restricted to a chosen subset of (query, key) pairs — a local window plus a few global tokens. Any position actually attended to gets an exact score, not a reconstructed approximation. **Error type: coverage loss — a dependency is missed only if it falls entirely outside the window and isn't routed through a global token.**

**The trade-off shifts, rather than disappears:** low-rank methods risk losing precision on dependencies that are technically visible but get washed out in compression. Sparse methods risk losing dependencies that fall outside the fixed attention pattern entirely, but preserve full precision on whatever they do attend to.

**One-line summary:** Low-rank methods approximate every score using a shared, compressed basis — accurate on average, but can wash out rare, sharp, long-range dependencies. Sparse methods compute exact scores on a restricted set of positions — precise wherever they look, but can miss dependencies outside that fixed set.
