# Lesson 4 — Longformer: The Long-Document Transformer

**Paper:** Beltagy, Peters, Cohan — *"Longformer: The Long-Document Transformer"* (2020)
**Source:** https://arxiv.org/abs/2004.05150
**Family:** Sparse connectivity (different mechanism from Lessons 1–2's kernel-trick family and Lesson 3's low-rank family)

---

## 1. Core idea (one line)

Instead of approximating the full N×N attention matrix (Linformer) or the similarity function itself (Katharopoulos/Performer), Longformer just **doesn't compute most of the matrix at all**. It restricts, upfront, which (query, key) pairs are considered — and for the pairs it does consider, the score is computed exactly, full precision, no approximation.

**Contrast with Lesson 3, Section 8:** low-rank methods approximate every score but can wash out rare sharp signals (precision loss). Sparse methods compute exact scores but only on a chosen subset of positions (coverage risk instead).

---

## 2. Building block 1 — Sliding window attention

Each token attends only to a fixed-size neighborhood: window size `w`, so `w/2` tokens on each side.

```
Complexity: O(n·w)   — linear in n, since w is fixed
```

**Intuition:** most useful attention is local anyway — nearby tokens modify each other's meaning most directly. Directly analogous to a CNN's convolutional kernel — a fixed-size receptive field sliding across the sequence.

**How far-apart tokens still get seen — stacking layers:** a single layer's window is narrow, but through `l` stacked layers the receptive field grows to `l × w`. A token's representation at a later layer has indirectly absorbed information from a much wider span than its immediate neighbors, because each layer's local mixing compounds — same idea as deep CNNs building up a large receptive field from small kernels.

---

## 3. Building block 2 — Dilated sliding window

Refinement borrowed from dilated CNNs: instead of `w/2` *contiguous* tokens on each side, leave **gaps of size d** between attended positions.

```
Receptive field at top layer: l × d × w     (instead of l × w)
```

**Why it helps:** same computational budget (still O(n) — same *number* of tokens attended, just spread further apart), but the effective reach grows much faster. Same intuition as dilated convolutions in vision — cover a larger area without adding compute, at the cost of not seeing every position densely.

**Practical detail:** dilation isn't applied uniformly. Lower layers use small/no dilation (fine-grained local patterns); dilation increases in some heads at higher layers (broader, more global structure) — mirrors the general CNN pattern of early layers = local detail, late layers = large-scale structure.

---

## 4. Building block 3 — Global attention

Sliding windows alone can't handle tokens that genuinely need to see the *entire* sequence — e.g. `[CLS]` for classification needs to summarize the whole document; in QA, question tokens need to attend across the entire passage.

**Mechanism:** a small, task-chosen set of tokens get **full, unrestricted, bidirectional attention** — they attend to every token, and every token attends back to them. Symmetric by design.

**Why it stays O(n):** the number of globally-attending tokens is small and fixed (doesn't grow with sequence length) → extra cost is O(n) × (small constant) = still O(n) overall.

**Detail worth remembering:** separate learned projection matrices (Q/K/V) are used for global attention vs. local (sliding-window) attention. Reasoning: attending locally and attending globally are different jobs — separate parameters let the model specialize instead of forcing one set of weights to do both.

---

## 5. Total complexity

```
O(n) overall — sliding window O(n·w) + dilation (same complexity class, larger reach)
+ global attention O(n) (small constant number of global tokens)
```

All three pieces are linear → the sum is linear. This is the headline result.

---

## 6. Trade-offs to hold onto

| | |
|---|---|
| **Benefit** | Exact attention scores wherever computed — no approximation error on visible pairs |
| **Cost** | Coverage risk — a dependency outside the window and not routed through a global token is missed entirely |
| **vs. Linformer (Lesson 3)** | Opposite failure mode: Linformer risks precision loss on in-view dependencies; Longformer risks coverage loss on out-of-view dependencies |
| **vs. Katharopoulos/Performer (Lessons 1–2)** | Different lever again — restricts *which* pairs are compared, rather than approximating *how* similarity is computed |
| **Causal compatibility** | Sliding window naturally supports causal masking (window looks backward only) — unlike Linformer, no structural incompatibility with autoregressive decoding |

---

## 7. Interview questions — tricky ones and how to answer them

**Q1: "How is this different from just using a smaller fixed context window, like early GPT models did?"**
*Trap:* sounds like Longformer just truncates context. A plain fixed window has no mechanism at all for long-range dependencies — information never crosses the boundary within a layer. Longformer's receptive field *grows with depth* (stacked layers) and global tokens provide an explicit unrestricted bridge across the whole sequence — so it can still capture long-range dependencies, via depth + global tokens instead of every layer seeing everything.

**Q2: "Does dilation lose information, like max-pooling does in CNNs?"**
Yes — worth saying directly. Dilation trades density for reach: fewer positions attended per window, so a token falling in a "gap" at every relevant layer/head could be missed entirely. Genuine coverage gap, mitigated (not eliminated) by making dilation head/layer-specific rather than uniform.

**Q3: "Compare Longformer's complexity guarantee to Linformer's — are they really the same O(n)?"**
Same big-O class, different constants and different failure modes. Longformer: O(n·w), constant = window size w. Linformer: O(n·k), constant = projection size k. Big-O hides the constants, and more importantly they fail differently: Longformer risks coverage gaps outside the window; Linformer risks precision loss on in-view rare dependencies. Naming both the shared complexity class *and* the different error character is the strong answer here.

**Q4: "Why not just make every token a global token — wouldn't that be more accurate?"**
Tests whether you understand *why* global attention is kept small. If every token were global, it's back to full O(n²) attention. Global attention stays cheap only because it's a small, fixed-size subset — a deliberate trade-off: spend the "expensive" full-attention budget only on the few tokens that most need whole-sequence visibility, let everything else rely on local + dilated windows.

**Q5: "Can Longformer handle causal/autoregressive decoding?"**
Good one to flag proactively (ties back to Lesson 3's encoder-only limitation for Linformer). Sliding-window attention is naturally compatible with causal masking — a window can simply look backward only (`w` tokens before position i, none after). No structural incompatibility like Linformer has, since Linformer's projections assume the full sequence is visible at once.

**Q6: "If sliding window attention is basically a CNN, why not just use a CNN instead of a Transformer?"**
Acknowledge the parallel honestly, then explain the real difference: the local mixing step is CNN-like, but it's still attention — weights (how much to attend to each neighbor) are content-dependent and dynamically computed per input via Q/K dot products, not fixed learned filters applied identically everywhere. Global attention also has no clean CNN analog — an explicit long-range bridge a CNN doesn't have without much deeper stacking or a different design. Borrows CNN intuition for receptive-field growth, but the weight-generating mechanism is still genuinely attention.

---

## Open questions / things to revisit
- [ ] Read BigBird (Zaheer et al., 2020) — adds a random-attention component on top of window + global attention; compare its guarantees to Longformer's.
- [ ] Look at how Longformer's staged training (increasing window size across training phases) affects the final receptive field vs. training with the full window size from the start.
- [ ] Understand why the paper uses separate Q/K/V projections for global vs. local attention in more depth — what breaks if they're shared?