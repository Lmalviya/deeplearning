# Linear Attention — Detailed Study Notes

**Paper:** Katharopoulos, Vyas, Pappas, Fleuret — *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"* (ICML 2020)
**Source:** https://arxiv.org/abs/2006.16236

---

## 1. Setting up the transformer layer

A transformer is a stack of layers `T_1, ..., T_L`. Each layer:

```
T_l(x) = f_l( A_l(x) + x )
```

Two pieces, doing two very different jobs:

- **`f_l(·)`** — a small two-layer feedforward network. Applied **row-wise**: it processes each token's feature vector on its own, with zero knowledge of any other token. If you fed it token 5's vector alone, it would compute the exact same output as if the whole sequence were present.
- **`A_l(·)`** — the self-attention function. This is the *only* place in the whole layer where token 5 gets to look at token 2, token 2 gets to look at token 5, etc. Every cross-token interaction in the model funnels through this one function.

### Why does this decomposition matter?

Because it tells you exactly where to look for the O(N²) cost.

- `f_l` touches N rows, each independently → cost scales as O(N). No matter how big or fancy the feedforward network is internally, it never compares one token to another, so it can never become quadratic in sequence length.
- `A_l` computes a similarity between *every* query and *every* key — N queries × N keys = N² comparisons. This is the only quadratic piece in the entire architecture.

**Consequence:** if you can make `A_l` linear in N, the *entire transformer layer* becomes linear in N, because `f_l` was never the problem. This is why the whole paper is laser-focused on rewriting attention specifically — it's not "one possible place to optimize," it's the *only* place that matters for this complexity class.

---

## 2. Softmax attention, and the generalized version

Standard scaled dot-product attention:

```
A_l(x) = V' = softmax(QKᵀ/√D) V
```

Here `Q, K, V` are all derived from x via learned linear projections. `Q` and `K` have dimension D per row, `V` has dimension M per row (often D = M, but not necessarily).

### Rewriting it row by row

Instead of thinking about the whole matrix operation, look at what happens to a single output row `V'_i` (the output for query position i). Equation 3 in the paper generalizes this to *any* similarity function `sim`, not just the softmax's exponential:

```
V'_i = [ Σ_{j=1}^N sim(Q_i, K_j) · V_j ] / [ Σ_{j=1}^N sim(Q_i, K_j) ]
```

**Read this as:** the output for token i is a *weighted average* of all value vectors `V_j`, where the weight given to `V_j` is `sim(Q_i,K_j)`, normalized by the sum of all such weights so they add up to 1.

Softmax attention is the special case where:

```
sim(q, k) = exp(qᵀk / √D)
```

Plugging this into the general formula and simplifying reproduces exactly `softmax(QKᵀ/√D)V`. So softmax attention isn't a fundamentally different mechanism — it's one specific choice of `sim` inside a much more general "weighted average of values" template. This reframing is what opens the door to swapping in other similarity functions.

### Why must `sim(·,·)` be non-negative?

This trips people up because softmax's *output* happens to live in [0,1], and it's tempting to think that's the reason. It's not — the real reason is more basic.

Look again at the structure:

```
weight_j = sim(Q_i, K_j) / Σ_j sim(Q_i, K_j)
```

For `V'_i` to be a genuine **weighted average** of the values — the thing that makes attention interpretable and numerically sane — the weights need to behave like a probability distribution: each `weight_j ≥ 0`, and they sum to 1. The division by the sum automatically handles the "sums to 1" part *regardless* of what `sim` is. But it does **not** automatically guarantee non-negativity.

If `sim` were allowed to output negative numbers:
- Individual weights could go negative, meaning `V'_i` would no longer be a genuine average — it could extrapolate *outside* the range spanned by the `V_j` vectors, which breaks the "attention pools information from other tokens" interpretation.
- The denominator `Σ_j sim(Q_i,K_j)` could become very small, exactly zero, or even negative if positive and negative terms cancel — causing division by zero or a sign flip, both numerically catastrophic.

So the constraint isn't "must match softmax's range," it's "must be non-negative for the weighted-average interpretation and the normalization to make sense at all." And you're right that `sim` is *not* required to be cosine similarity — cosine similarity is just one example of a similarity notion; `sim` here is a free design choice, softmax's exponential dot-product being one instance, cosine another, and (as we'll see) polynomial kernels yet another.

---

## 3. Kernels and feature maps — the concept that makes linearization possible

### What is a kernel?

A **kernel function** `k(x, y)` takes two vectors and returns a single non-negative scalar, interpreted as "how similar are x and y":

```
k : R^F × R^F → R⁺
```

(The paper's notation `k(x,y): R^{2×F} → R⁺` is just saying the same thing — the pair of F-dimensional inputs together lives in a 2×F space, and the output is a non-negative real number.)

You already know one kernel-like object: the plain dot product `x·y`. But it's *not* a valid kernel for our purposes because it can be negative. So we need something related, but constrained to be non-negative.

### The feature-map trick (Mercer's theorem, informally)

Here's the deep and slightly surprising fact that the whole paper hinges on: for many useful kernel functions `k(x,y)`, there exists some transformation `φ(·)` — called a **feature map** — such that:

```
k(x, y) = φ(x)ᵀ φ(y)
```

In words: instead of evaluating some possibly complicated nonlinear similarity function `k` directly on x and y, you can first transform x and y *separately* through `φ`, into a (usually higher-dimensional) space, and then just take an ordinary, boring dot product of the transformed vectors — and you get the exact same number back.

This sounds abstract, so here's a concrete worked example.

**Worked example:** Let `k(x,y) = (x·y)²` for 2D vectors `x = (x1,x2)`, `y = (y1,y2)`.

Expand it:
```
(x·y)² = (x1y1 + x2y2)²
       = x1²y1² + 2x1y1x2y2 + x2²y2²
```

Now define:
```
φ(x) = (x1², √2·x1x2, x2²)
```

Then:
```
φ(x)ᵀφ(y) = x1²y1² + 2x1x2y1y2 + x2²y2²
```

Same expression. So `k(x,y) = (x·y)²` — despite looking like a nonlinear function of x and y — is secretly nothing more than a plain dot product, once you've moved into the 3-dimensional feature space defined by φ. This generalizes: many nonlinear kernels are "linear in disguise," if you're willing to transform the inputs first.

### Why do we care about this specific property?

Because plain dot products can be **regrouped algebraically** — this is what makes the O(N²)→O(N) trick possible in the next section. A generic `sim(q,k)` function has no guarantee of this regrouping property. But if you deliberately *choose* `sim(q,k) = φ(q)ᵀφ(k)` for some feature map φ, you inherit two things simultaneously:
1. Non-negativity is easy to guarantee by picking φ so that `φ(q)ᵀφ(k) ≥ 0` always (e.g., because φ's outputs are always non-negative — this is exactly why the paper later chooses `φ(x) = elu(x)+1`, which is always positive).
2. The associativity of ordinary matrix multiplication, which is the mathematical fact that unlocks the linear-time computation.

---

## 4. The linearization trick itself (equations 4–6)

### Substituting the kernel form into generalized attention

Take equation 3 and replace `sim(Q_i,K_j)` with `φ(Q_i)ᵀφ(K_j)`:

```
V'_i = [ Σ_{j=1}^N φ(Q_i)ᵀ φ(K_j) V_j ] / [ Σ_{j=1}^N φ(Q_i)ᵀ φ(K_j) ]         (eq. 4)
```

Nothing has changed conceptually yet — this is just eq. 3 with a specific *kind* of `sim` plugged in.

### The reordering step — this is the actual trick

Look closely at the numerator's sum:

```
Σ_{j=1}^N φ(Q_i)ᵀ φ(K_j) V_j
```

Each term in this sum is: **(a scalar)** `φ(Q_i)ᵀφ(K_j)`, **times (a vector)** `V_j`. Critically, `φ(Q_i)` is the *same* for every term in the sum — it doesn't depend on the summation index `j` at all, only on the fixed query `i`. That means it can be factored out of the sum, exactly the way you'd factor `a` out of `a·b1 + a·b2 + a·b3 = a·(b1+b2+b3)`:

```
φ(Q_i)ᵀ [ Σ_{j=1}^N φ(K_j) V_jᵀ ]                                              (eq. 5, numerator)
```

The denominator factors the same way:

```
φ(Q_i)ᵀ [ Σ_{j=1}^N φ(K_j) ]
```

**The crucial observation:** the bracketed term `Σ_j φ(K_j)V_jᵀ` involves *only* K and V — it has completely lost any dependence on `i`. This means it is the **same for every query row**. You do not need to recompute it once per query; you compute it **once, total**, and then every query just does one cheap dot product against it.

This is the entire mechanism behind the speedup. Before: for every one of N queries, sum over N keys → N² work. After: sum over N keys **once** (build one shared matrix), then N queries each do one lookup against that shared matrix → N work.

### Matrix form (equation 6)

Written for the whole matrices at once (Q, K ∈ R^{N×D}, V ∈ R^{N×M}), φ applied row-wise to Q and K:

```
(φ(Q) φ(K)ᵀ) V   =   φ(Q) (φ(K)ᵀ V)
```

Both sides compute the *exact same* N×M output matrix. The only difference is which multiplication you do first — and that's legal purely because ordinary matrix multiplication is **associative**: `(AB)C = A(BC)` for any conformable matrices A, B, C. This is why the kernel/feature-map setup mattered: it turned an opaque similarity function into an actual matrix product that associativity applies to.

### Counting the cost explicitly

Let Q, K ∈ R^{N×D} (D = feature-map output dimension), V ∈ R^{N×M}.

**Left side — naive order:** `(φ(Q) φ(K)ᵀ) V`
- Step 1: `φ(Q) φ(K)ᵀ` → (N×D)·(D×N) = an **N×N matrix**. Cost: O(N²D).
- Step 2: multiply that N×N matrix by V (N×M). Cost: O(N²M).
- **Total: O(N²)** — the N×N intermediate matrix is unavoidable in this order, and it's the source of the quadratic cost.

**Right side — reordered:** `φ(Q) (φ(K)ᵀ V)`
- Step 1: `φ(K)ᵀ V` → (D×N)·(N×M) = a **D×M matrix**. Crucially, this shape does *not* grow with N — its size is fixed once D and M are fixed. Cost: O(NDM), linear in N.
- Step 2: `φ(Q)` (N×D) times that fixed D×M matrix → an N×M output. Cost: O(NDM), linear in N.
- **Total: O(N)** — no N×N object is ever formed.

Same numerical answer, dramatically different cost, purely because of computation order. This is the whole "aha" of the paper in one picture.

### Why can't you do this to softmax directly?

`softmax(QKᵀ)` is not a plain matrix product you're free to regroup. Softmax first computes `QKᵀ` (forcing that N×N matrix into existence), then applies an elementwise nonlinearity, then normalizes each row by that row's sum. That normalization step couples every entry in a row together *before* you ever multiply by V — there's no way to "distribute" the softmax operation the way we distributed a plain scalar-times-vector term above. Associativity requires an actual product structure, and softmax breaks that structure by design (that's what makes it softmax rather than just "dot product"). Swapping in `φ(q)ᵀφ(k)` removes the coupling and restores the product structure, at the cost of no longer being *exactly* softmax.

---

## 5. Why not linearize softmax exactly? (Exponential vs. polynomial kernels)

### The exponential kernel needs infinite dimensions

Softmax's similarity is `sim(q,k) = exp(qᵀk/√D)`. Can we find a *finite* feature map φ such that `exp(qᵀk) = φ(q)ᵀφ(k)` exactly? Look at the Taylor series of the exponential function:

```
e^(qᵀk) = Σ_{n=0}^∞ (qᵀk)ⁿ / n!
        = 1 + qᵀk + (qᵀk)²/2! + (qᵀk)³/3! + ...
```

Each individual term `(qᵀk)ⁿ` is a degree-n polynomial in q and k, and (exactly like the worked example in Section 3) each such term *can* be written as a dot product of a finite-dimensional degree-n polynomial feature map. But the exponential kernel is the sum of **infinitely many** such terms, one for every degree n = 0, 1, 2, 3, .... To represent the full exponential kernel exactly as a single dot product `φ(q)ᵀφ(k)`, φ would need to output a vector with one coordinate slot for every polynomial degree — infinitely many coordinates.

**Why this kills exact linearization:** the entire speedup depended on building `φ(K)ᵀV`, a finite D×M matrix, once. If φ(k) is infinite-dimensional, that matrix is infinite-dimensional too — you can't store it, let alone compute it in finite time. So exact linearization of softmax attention isn't just hard, it's **mathematically infeasible** with this approach. Every practical linear-attention method (this paper's `elu(x)+1`, Performer's random Fourier features, etc.) is therefore necessarily an *approximation* to true softmax — none of them reproduce it exactly, because reproducing it exactly is provably impossible under a finite feature map.

### The polynomial kernel — a finite alternative

Instead of trying to approximate the infinite exponential kernel, you can just pick a genuinely finite kernel outright. A degree-2 polynomial kernel:

```
sim(q,k) = (qᵀk)²          (or a variant like (qᵀk + 1)²)
```

This is a **single, finite term** — not an infinite sum. As shown in the Section 3 worked example, it has an *exact*, finite-dimensional feature map (no approximation needed for this kernel itself — the approximation only enters relative to softmax, which this kernel is *not* trying to be).

- **Trade-off:** it's not softmax, so the attention pattern it induces is different in character. But Tsai et al. (2019) found empirically that polynomial kernels perform comparably to the exponential/RBF kernel on real tasks — the exact exponential form of softmax isn't uniquely necessary for good performance.
- **Benefit:** because it's finite by construction, you get exact linear-attention computation with no infinite-dimension issue to work around.

### Feature-map dimensionality and the O(ND²M) cost

For a degree-2 polynomial feature map over F-dimensional inputs, the feature map's output needs one coordinate for every pairwise product `x_i · x_j` (as in the worked example, `x1², √2x1x2, x2²` — three coordinates for a 2-dimensional input). In general this scales roughly as **F²** — quadratic in the *original* input dimension, not the sequence length.

Recall the general linear-attention cost from Section 4: O(N·D·M), where D is the feature map's output dimension. For the degree-2 polynomial map, D itself is roughly (original dim)², so substituting gives:

```
O(N · D² · M)
```

**Important distinction:** this D² is a cost in the *feature/model* dimension, not the *sequence length* N. The scaling with respect to N is still perfectly linear — O(N) — regardless of kernel choice. What changes between kernel choices is only the constant/dimensional overhead per token (via D and M), which is a genuine engineering trade-off: richer kernels (higher polynomial degree, or better approximations to the exponential kernel) buy closer fidelity to true softmax attention, at the price of larger effective D and thus more compute per token — but sequence-length scaling never leaves O(N).

---

## 6. Causal masking — turning linear attention into a recurrence

### Why masking is needed at all

For autoregressive generation (predicting token i+1 from tokens 1..i), token i must **not** be allowed to see tokens after it — otherwise the model could "cheat" by looking at the future during training. This is enforced by restricting the sum in the attention formula to only past and current positions:

```
V'_i = [ Σ_{j=1}^{i} sim(Q_i,K_j) V_j ] / [ Σ_{j=1}^{i} sim(Q_i,K_j) ]        (eq. 8)
```

The only change from eq. 3 is the upper limit of the sum: `N` becomes `i`. Everything else about the weighted-average interpretation from Section 2 still applies — this is a weighted average of *only the values seen so far*.

### Linearizing the masked version

Apply the exact same kernel substitution and factoring from Section 4, just with the sum capped at `i` instead of `N` (eq. 9):

```
V'_i = φ(Q_i)ᵀ [ Σ_{j=1}^{i} φ(K_j) V_jᵀ ] / φ(Q_i)ᵀ [ Σ_{j=1}^{i} φ(K_j) ]
```

### Naming the running sums — S_i and Z_i

Define:

```
S_i = Σ_{j=1}^{i} φ(K_j) V_jᵀ          (eq. 10)   — a running (D×M) matrix
Z_i = Σ_{j=1}^{i} φ(K_j)               (eq. 11)   — a running (D-dim) vector
```

So the output simplifies to:

```
V'_i = φ(Q_i)ᵀ S_i / φ(Q_i)ᵀ Z_i        (eq. 12)
```

### Why this is a recurrence (the "Transformers are RNNs" idea)

Notice `S_i` differs from `S_{i-1}` by exactly one new term:

```
S_i = S_{i-1} + φ(K_i) V_iᵀ
Z_i = Z_{i-1} + φ(K_i)
```

This is precisely the structure of an RNN's hidden-state update: a fixed-size "state" (`S_i`, `Z_i`) that gets updated by a constant amount of work at every timestep, carrying forward a compressed summary of everything seen so far. `S_i` in particular plays the role of a **memory/state matrix** — it never grows in size as i increases, it just gets updated in place.

**Cost accounting:**
- Updating `S_i → S_{i+1}` and `Z_i → Z_{i+1}`: fixed-size matrix/vector addition → **O(1)** per step, independent of how large i is.
- Doing this for all N positions: N steps × O(1) = **O(N)** total.
- Contrast with standard softmax attention at generation time: at step i, you'd need to recompute attention over all i previous tokens from scratch (no reusable running state), so generating N tokens costs `1+2+3+...+N = O(N²)` in total.

**Practical implication:** during autoregressive decoding (generating tokens one at a time), linear attention lets you carry forward a small fixed-size state and do O(1) work per new token, instead of an ever-growing recomputation. This is the direct payoff of viewing linear attention as recurrent — it's not just a theoretical curiosity, it's what gives the reported "up to 4000x faster" autoregressive inference in the paper.

---

## 6a. Worked numeric example — the causal recurrence by hand

Toy setup: N=3, 1-dimensional K/V for simplicity, and φ = identity just to keep the arithmetic light (the real paper uses `elu(x)+1`, but the recurrence *structure* below is identical regardless of φ):

```
K_1=2, V_1=3
K_2=1, V_2=4
K_3=3, V_3=2
```

**i=1:**
```
S_1 = φ(K_1)·V_1 = 2·3 = 6
Z_1 = φ(K_1) = 2
V'_1 = S_1/Z_1 = 6/2 = 3
```
(With only one token seen, output = that token's own value — makes sense.)

**i=2** (update from previous state, don't recompute from scratch):
```
S_2 = S_1 + φ(K_2)·V_2 = 6 + (1·4) = 10
Z_2 = Z_1 + φ(K_2) = 2 + 1 = 3
V'_2 = S_2/Z_2 = 10/3 ≈ 3.33
```

**i=3:**
```
S_3 = S_2 + φ(K_3)·V_3 = 10 + (3·2) = 16
Z_3 = Z_2 + φ(K_3) = 3 + 3 = 6
V'_3 = S_3/Z_3 = 16/6 ≈ 2.67
```

Each step only touches the *previous* S, Z and the *current* K, V — token 1's raw values are never revisited at step 3, they're already folded into the running state. That's the O(1)-per-step property in action. Standard softmax attention, by contrast, would recompute a fresh sum over all previous keys at every single step.

---

## 7. Why the paper chooses φ(x) = elu(x) + 1

Recall two requirements established earlier:
- **Section 2:** `sim(·)` (and hence φ) must be non-negative, or the weighted-average interpretation and normalization break down.
- **Section 5:** the exponential kernel needs an *infinite*-dimensional feature map, which is infeasible — any usable φ must be finite.

`elu(x)+1` is a specific, deliberately simple function satisfying both:

```
elu(x) = x            if x > 0
elu(x) = exp(x) − 1    if x ≤ 0

φ(x) = elu(x) + 1:
φ(x) = x + 1          if x > 0
φ(x) = exp(x)          if x ≤ 0
```

**Why this particular form:**

- **Non-negativity by construction.** For `x ≤ 0`, `φ(x) = exp(x) > 0` always (exponentials are never zero or negative). For `x > 0`, `φ(x) = x+1 > 1`. So `φ(x) > 0` for every real input — the Section 2 constraint is satisfied automatically, not enforced by clipping or an extra constraint during training.
- **Finite by construction.** Unlike the true exponential kernel (Section 5), this φ is just an ordinary elementwise function producing an ordinary finite-dimensional output vector — no Taylor expansion, no infinite sum. It sidesteps the infinite-dimensionality problem rather than trying to approximate the infinite kernel.
- **Why not the simpler ReLU(x)?** ReLU also guarantees non-negativity (`max(0,x) ≥ 0`) and is cheaper to compute, but its gradient is *exactly zero* for all negative inputs. During training, any query/key component that lands in negative territory gets no gradient signal there — a "dead" direction that can stall learning. `elu(x)+1` stays smooth and keeps a small but non-zero gradient for negative inputs (decaying gradually via the exponential rather than hitting a hard wall at zero). This smoothness was the paper's stated reason for preferring ELU over ReLU.
- **An honest caveat:** the paper itself treats this as a pragmatic choice, not a uniquely "correct" or derived-from-first-principles one. It satisfies the required properties (non-negative, finite, smooth gradient) and was found empirically to perform close to full softmax attention — but it is not claimed to be optimal. This is exactly why the space stayed open afterward: later papers propose different φ entirely —
  - **Performer (FAVOR+)** — uses random projections to build a finite-dimensional φ that is a statistically *unbiased approximation* of the true exponential kernel, rather than a different kernel altogether. Roughly: `φ(x) = [h(x)/√m] · [exp(Rx); exp(-Rx)]`, where R is a matrix of m random projection directions. As m grows, this gets closer to true softmax attention (in expectation) — a direct trade-off between fidelity and compute cost, echoing the D-dimension trade-off from Section 5.
  - **cosFormer, DPFP, learned ReLU variants (Kasai et al.), and others** — each explore different φ, treating "what's the best finite non-negative feature map" as an open design question rather than something this paper definitively solved.

**Key philosophical distinction to hold onto:** ELU+1 is a *different, cheaper kernel* that happens to satisfy the required properties — it does not claim to approximate softmax specifically. Performer's random features, by contrast, are explicitly constructed to approximate the *true* exponential/softmax kernel. Both land in the same complexity class (O(N)), but they represent two different philosophies for getting there.

---

## 8. What the experiments actually showed

The accuracy story is task-dependent, not uniformly "linear attention is basically free":

- **Autoregressive image generation:** the linear transformer performed **on par with** the standard softmax transformer — a genuine free lunch on this task.
- **Automatic speech recognition (ASR):** performance was **noticeably worse** than standard softmax attention — the ELU approximation does not transfer equally well to every domain.
- **Speed:** the headline result — **up to 4000x faster** than a standard transformer for autoregressive prediction on very long sequences. This comes directly from the O(1)-per-step recurrence (Section 6) replacing the O(N) recomputation-from-scratch that ordinary softmax attention requires at every generation step.
- **Training stability (noted in later work, e.g. Performer's own appendix):** the ELU-based linear transformer can suffer from training instability — exploding gradients in some configurations — something not heavily emphasized in the original paper's own experiments but flagged by subsequent comparisons.

**Takeaway:** linear attention's cost savings are real and large, but the accuracy trade-off is genuinely task-dependent — good on some tasks (image generation), worse on others (ASR), with stability caveats surfacing in later scrutiny. This nuance (rather than "linear attention is a strictly better free upgrade") is exactly why the field kept iterating on alternative feature maps (Performer, cosFormer, DPFP, etc.) after this paper — a good point to raise if an interviewer asks "are there any downsides?"

---

## Summary table — the whole argument in one place

| Concept | What it is | Why it matters |
|---|---|---|
| `f_l` vs `A_l` split | f_l is row-wise (O(N)), A_l mixes all tokens (O(N²)) | Tells you attention is the *only* bottleneck worth fixing |
| Generalized attention (eq. 3) | Softmax attention is one instance of a weighted-average template with a free `sim` function | Opens the door to swapping in other similarity functions |
| Non-negativity constraint on `sim` | Needed for the weighted-average / normalization to make sense | Not about matching softmax's [0,1] range specifically |
| Kernel + feature map | `k(x,y) = φ(x)ᵀφ(y)` — nonlinear similarity rewritten as a plain dot product in transformed space | Restores associativity, which is the actual source of the speedup |
| Reordering `(φ(Q)φ(K)ᵀ)V → φ(Q)(φ(K)ᵀV)` | Same result, different computation order | O(N²) → O(N); the N×N matrix is never formed |
| Exponential kernel | Needs an infinite-dimensional feature map (Taylor series) | Exact softmax linearization is mathematically infeasible |
| Polynomial kernel | Finite-dimensional by construction | A viable, exact alternative; empirically competitive with softmax |
| O(ND²M) | Cost of degree-2 polynomial linear attention | Still linear in N; D² is a per-token cost from the feature map's dimensionality |
| Causal masking + S_i, Z_i | Running sums updated in O(1) per step | Makes linear attention literally recurrent → O(N) generation, not O(N²) |
| φ(x) = elu(x)+1 | Non-negative + finite + smooth-gradient feature map choice | Satisfies both the non-negativity and finite-dimensionality requirements; not claimed optimal |
| Performer / FAVOR+ | Random-feature approximation of the true exponential kernel | Different philosophy from ELU+1 — approximates real softmax rather than substituting a new kernel |
| Experimental results | On par on image generation, worse on ASR, up to 4000x faster autoregressive inference | Accuracy trade-off is task-dependent, not a free upgrade across the board |