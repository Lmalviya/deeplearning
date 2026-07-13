# Lesson 2 — Performer (FAVOR+): Approximating Softmax Attention with Random Features

**Paper:** Choromanski et al. — *"Rethinking Attention with Performers"* (ICLR 2021)
**Source:** https://arxiv.org/abs/2009.14794
**Builds on:** Lesson 1 (Katharopoulos et al., "Transformers are RNNs") — the associativity/reordering trick (`φ(Q)(φ(K)ᵀV)` is O(N)) is reused unchanged here. What's different is *how φ is derived*.

---

## 0. The goal, and how it differs from Lesson 1

In Lesson 1, `φ(x) = elu(x)+1` was picked because it's non-negative and finite — it defines a **different, cheaper kernel** that isn't trying to reproduce true softmax similarity.

Performer sets a stricter goal: find a finite φ such that `φ(q)ᵀφ(k)` is a **provably unbiased estimator** of the *actual* softmax similarity `exp(qᵀk)`. In other words — don't substitute a different notion of similarity, actually approximate the real one, with a statistical guarantee that the approximation is correct on average. This is why the derivation needs more machinery.

**Name breakdown:** FAVOR+ = **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features (**+**).
- *Fast Attention* — reuses the Lesson-1 associativity trick.
- *Positive* — the random features are always non-negative (Section 3 below).
- *Orthogonal* — the random directions used are forced to be perpendicular to each other for lower variance (Section 4 below).
- *Random features* — the whole approach is built on randomly sampled projections, not a fixed deterministic formula like `elu(x)+1`.

---

## 1. Background: the Gaussian (RBF) kernel

### High-level idea

A kernel measures "how similar are x and y." You already know the dot-product-based kernel `k(x,y) = xᵀy`, which depends on the *angle and magnitude* of the two vectors. The **Gaussian kernel** (also called the **RBF kernel** — Radial Basis Function) encodes a completely different, more intuitive notion: similarity based purely on **distance**.

```
k_gauss(x, y) = exp( -‖x - y‖² / 2 )
```

`‖x-y‖²` is just the squared Euclidean distance between the two points.

### Building intuition with numbers (1D case, so ‖x-y‖ = |x-y|)

| Case | ‖x-y‖² | k_gauss(x,y) | Interpretation |
|---|---|---|---|
| x = y (identical) | 0 | exp(0) = 1 | Maximum similarity |
| x, y close | 0.1 | exp(-0.05) ≈ 0.95 | Still very similar |
| x, y moderately apart | 4 | exp(-2) ≈ 0.135 | Noticeably less similar |
| x, y far apart | 25 | exp(-12.5) ≈ 0.0000037 | Essentially unrelated |

**Shape:** a smooth bell curve. Equals 1 exactly when points coincide, decays rapidly and smoothly toward 0 as distance grows, never negative, never above 1. "Radial" because it depends only on distance (a radius), not on direction or absolute position.

### Contrast with the dot-product kernel

| | Dot product `xᵀy` | Gaussian kernel `exp(-‖x-y‖²/2)` |
|---|---|---|
| Depends on | angle + magnitude | pure distance |
| Range | unbounded, can be negative | always in (0, 1] |
| Two far-apart, same-direction vectors | can score *high* | scores *low* (they're far apart) |

This is a genuinely different similarity notion — it's worth not conflating the two.

---

## 2. Why Performer brings in the Gaussian kernel at all

This is the key algebraic bridge. Start from the identity for squared distance:

```
‖q - k‖² = ‖q‖² + ‖k‖² - 2qᵀk
```

Rearrange for `qᵀk`:

```
qᵀk = ( ‖q‖² + ‖k‖² - ‖q-k‖² ) / 2
```

Exponentiate both sides:

```
exp(qᵀk) = exp(‖q‖²/2) · exp(‖k‖²/2) · exp(-‖q-k‖²/2)
```

**Read the right-hand side term by term:**
- `exp(‖q‖²/2)` — depends only on q, cheap to compute per-token.
- `exp(‖k‖²/2)` — depends only on k, cheap to compute per-token.
- `exp(-‖q-k‖²/2)` — this is *exactly* the Gaussian kernel from Section 1.

**This is an exact identity, not an approximation (yet).** It says: the softmax similarity between q and k is the Gaussian (distance-based) similarity between them, corrected by two simple per-vector scaling factors.

**Why is this useful?** No one had a good finite random-feature approximation for `exp(qᵀk)` directly. But approximating the Gaussian kernel with random features was already a solved problem in classical ML (Random Fourier Features, Rahimi & Recht, 2007 — originally used to speed up kernel SVMs, unrelated to transformers). Performer's move: reuse that 13-year-old solved technique for the Gaussian piece, then patch in the two extra per-vector scalars to convert the answer back into an approximation of the exponential kernel actually needed.

---

## 3. Random Fourier Features — approximating the Gaussian kernel

### The classical result (Bochner's theorem, informally)

Any kernel that depends only on distance (like the Gaussian kernel) can be written as an **expectation over random frequencies**:

```
exp(-‖q-k‖²/2) = E_ω[ cos(ωᵀq - ωᵀk) ]      for ω ~ N(0, I)
```

**Intuition:** instead of computing the smooth bell-curve similarity directly, you get the *same value on average* by: (a) picking a random direction ω, (b) projecting q and k onto that direction, (c) checking how "in sync" the two projections are via a cosine, (d) averaging over many random draws of ω. Each single draw is a noisy, cheap estimate; enough draws average out to the true value. This is the same logic as Monte Carlo estimation generally — replace an expensive exact computation with cheap random samples that are correct in expectation.

### Turning this into a finite feature map

Sample `m` random vectors `ω_1,...,ω_m` from a Gaussian, and build:

```
φ_trig(x) = (1/√m) · [ sin(ω_1ᵀx), cos(ω_1ᵀx), ..., sin(ω_mᵀx), cos(ω_mᵀx) ]
```

Then `φ_trig(q)ᵀφ_trig(k) ≈ exp(-‖q-k‖²/2)`, unbiased, improving as `m` grows. Multiply in the two per-vector correction scalars from Section 2, and you have a finite, random, unbiased estimator of the *true* exponential/softmax kernel.

**This is Performer's first (baseline) version — trigonometric random features.** It already solves the "approximate real softmax with a finite φ" problem in principle. But it has a practical flaw — next section.

---

## 4. The "P" in FAVOR+ — Positive random features

### The problem with sin/cos features

`sin` and `cos` can be **negative**. Recall the non-negativity requirement (from Lesson 1, Section 2): attention weights must be non-negative for the weighted-average interpretation to hold, and to avoid the normalizing denominator landing near zero or flipping sign.

With trigonometric features, nothing stops `φ_trig(q)ᵀφ_trig(k)` from occasionally coming out near-zero or slightly negative due to random sampling noise. In practice this causes **high variance** in the estimate — especially over long sequences — which can destabilize training (this is noted in later comparisons, including Performer's own appendix flagging that the plain Linear Transformer with ELU+1 could hit exploding-gradient issues; trig features have an analogous instability risk).

### The fix: exponential-based positive features

Instead of sin/cos, use a feature map built from `exp(·)`, which is always strictly positive:

```
φ_pos(x) = (1/√m) · [ exp(ω_1ᵀx - ‖x‖²/2), ..., exp(ω_mᵀx - ‖x‖²/2) ]
```

This is derivable from the same underlying Gaussian-integral identity as the trig version (so it's still an unbiased estimator of the same target kernel), but **every coordinate of the output is strictly positive** by construction, since it's built entirely from exponentials. This directly satisfies the non-negativity requirement — while, unlike ELU+1, still being a genuine approximation of real softmax rather than a substitute kernel.

**This positivity is the "+"/"P" in FAVOR+.**

---

## 5. The "O" in FAVOR+ — Orthogonal random features

### The inefficiency in plain random sampling

When `ω_1,...,ω_m` are sampled independently from a Gaussian, some will, by chance, point in nearly the same direction. Two nearly-parallel ω's carry almost redundant information — you're "spending" two samples to learn approximately one thing.

### The fix: force exact orthogonality

Performer instead constrains `ω_1,...,ω_m` to be **exactly orthogonal** (perpendicular to each other), via Gram-Schmidt orthogonalization, while preserving each individual ω's original marginal distribution (so unbiasedness is untouched). This spreads the random directions out to cover the space more evenly.

**Effect:** provably **lower variance** in the kernel estimate for the same number of samples `m`. Practically, this means you need fewer random features to hit a given accuracy target, which directly reduces compute — a smaller `m` is cheaper per token.

**Name now makes full sense:** Fast Attention (Lesson-1 associativity trick) + Via **P**ositive (Section 4) **O**rthogonal (Section 5) **R**andom features = FAVOR+.

---

## 6. Putting it together — the full pipeline

1. Rewrite `exp(qᵀk)` exactly as `exp(‖q‖²/2) · exp(‖k‖²/2) · exp(-‖q-k‖²/2)` (Section 2).
2. Approximate the Gaussian-kernel piece `exp(-‖q-k‖²/2)` with random features `φ(x)` — positive (Section 4) and orthogonal (Section 5) for a stable, low-variance, unbiased estimate.
3. Fold the two per-vector scalar corrections into φ, giving a finite φ such that `φ(q)ᵀφ(k) ≈ exp(qᵀk)`, unbiased.
4. Plug this φ into the exact same reordering trick from Lesson 1:
   ```
   (φ(Q) φ(K)ᵀ) V   =   φ(Q) (φ(K)ᵀ V)
   ```
   Same associativity argument, same O(N²) → O(N) benefit — nothing new here, this step is 100% reused from Lesson 1.

**What's actually new in Performer, relative to Lesson 1:** not the complexity trick itself, but *how φ is derived* — a principled, statistically-grounded approximation of true softmax, instead of an arbitrary non-negative substitute.

---

## 7. Bidirectional vs. unidirectional attention — why the paper treats them separately

This split is **not** a fork in the underlying theory — the kernel/associativity trick above applies identically either way. It's about what extra engineering is needed to realize O(N) in practice for each case.

- **Bidirectional (BERT-style):** every token attends to every other token, no ordering constraint. Compute `φ(K)ᵀV` once over the whole sequence (a fixed-size matrix), then `φ(Q) · (that matrix)`. Clean, one-shot, O(N). No special handling needed.

- **Unidirectional (causal/GPT-style):** token i can only attend to tokens `j ≤ i` — same causal-masking setup as Lesson 1, Section 6. This means you need the same running-sum recurrence idea (`S_i = S_{i-1} + φ(K_i)V_iᵀ`) that you already worked through by hand in Lesson 1's worked example. The extra piece in Performer specifically: because φ now involves random projections (not a simple fixed function like `elu(x)+1`), naively computing all N partial sums `S_1,...,S_N` on GPU/TPU hardware efficiently requires a specific **prefix-sum algorithm**. This is an implementation/hardware-efficiency detail, not a new mathematical idea — the underlying recurrence is conceptually identical to Lesson 1's causal case.

**One-line answer to hold onto:** the associativity trick (and therefore linear complexity) works for both bidirectional and unidirectional attention. The causal case additionally reveals an RNN-like recurrence and needs extra care to compute that recurrence efficiently — that's true for both Lesson 1's ELU+1 and Performer's random features; Performer's paper is just more explicit about the implementation detail because the φ is more complex.

---

## Summary table

| Concept | What it is | Why it matters |
|---|---|---|
| Gaussian / RBF kernel | `exp(-‖x-y‖²/2)` — distance-based similarity, bounded in (0,1] | Different notion than dot-product similarity; the piece Performer reuses classical theory for |
| `exp(qᵀk)` decomposition | `= exp(‖q‖²/2)·exp(‖k‖²/2)·exp(-‖q-k‖²/2)` | Exact identity that rewrites softmax similarity in terms of the Gaussian kernel + two per-vector scalars |
| Random Fourier Features | `E_ω[cos(ωᵀq - ωᵀk)] = Gaussian kernel` | Classical (2007) solved technique for approximating distance-based kernels with random projections |
| Positive random features ("P") | Swap sin/cos for `exp(·)`-based features | Guarantees non-negativity, avoiding the high-variance/instability of signed trig features |
| Orthogonal random features ("O") | Force sampled ω's to be exactly perpendicular via Gram-Schmidt | Lower variance for the same number of samples → fewer features needed for the same accuracy |
| FAVOR+ | Fast Attention (Lesson-1 trick) + Positive + Orthogonal Random features | Finite φ that's a genuine, low-variance, unbiased approximation of true softmax — not a substitute kernel |
| Bidirectional vs. unidirectional | Same theory either way; causal case needs a recurrence + prefix-sum algorithm | Mirrors Lesson 1's Section 6 exactly; the split is about implementation, not math |
