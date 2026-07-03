# 9.3 Open-Set / Embedding Architecture at 50-Class Scale

## Problem

Chapter 4.1 deliberately deferred a question: at what point does comparing an embedding against
the reference set (Chapter 2.3's classification mechanism) stop being viable as a brute-force
operation and require a dedicated vector index (ANN — Approximate Nearest Neighbor search)?
This deserves an honest, numbers-based answer rather than an assumption that "50 classes" alone
demands specialized infrastructure — echoing the same discipline applied to sharding in Chapter
4.4.

## Solution / Concept: An Honest Capacity Check

**Reference-set size at 50 classes**, using a reasonable curated-prototype approach (not "every
processed document becomes a reference example," which is a different, much larger-scale
design considered separately below): assume roughly 10–50 curated reference examples per class.

```
50 classes × (10 to 50) examples/class = 500 to 2,500 reference vectors total
```

**Brute-force comparison cost**: each reference vector is a 768-dimensional embedding (Chapter
4.1's assumption). A cosine similarity computation against one reference vector is a dot
product plus two norms — on the order of a few thousand floating-point operations. Comparing
against the full reference set:

```
2,500 reference vectors × ~1,500 FLOPs/comparison ≈ 3.75 million FLOPs per inference
```

This is trivial for modern hardware — sub-millisecond even computed naively on a CPU, and
effectively free on the GPU already running the embedding backbone itself. At the target
page-inference rate from Chapter 1.2 (≈58/sec average, higher at spike), brute-force comparison
against a reference set of this size is **not** a meaningful bottleneck, whether performed via
`pgvector`'s exact search in Postgres or even a simple in-memory comparison loaded into each
Classification Service instance.

**Conclusion: at 50 classes with curated reference sets in the hundreds-to-low-thousands of
vectors, a dedicated ANN vector database is not justified by throughput or latency concerns.**
Exact brute-force search, cached per Chapter 6.1's caching strategy, is simpler to operate and
gives exact (not approximate) results — a real accuracy advantage over ANN with no offsetting
cost at this scale.

## When Vector DB / ANN Search Actually Becomes Necessary

The real trigger is **reference-set size**, not class count. Two scenarios would push reference
sets from thousands into the hundreds-of-thousands-to-millions range, where brute-force
comparison genuinely does become a bottleneck:

1. **Growing the number of reference examples per class far beyond a curated prototype set** —
   e.g., deciding that better coverage of within-class visual/textual variation (Lesson 9.1's
   confusability concern) requires hundreds of examples per class rather than dozens.
2. **Moving toward a fully non-parametric design** — comparing against a large, continuously
   growing pool of past labeled/reviewed documents (effectively, every reviewed document ever
   becomes a reference point) rather than a small curated set — a design with real accuracy
   upside (more coverage) but a fundamentally different reference-set scale.

If either of these is adopted, reference-set size can cross into the range where brute-force
comparison cost (which scales linearly with reference-set size) starts measurably affecting
per-inference latency, and an ANN index (e.g., HNSW or IVF, available via `pgvector`'s own ANN
index support, or a dedicated vector database such as Pinecone, Weaviate, or Milvus) becomes
worth its cost.

## Trade-offs

| Approach | Gain | Cost | Justified at this system's 50-class, curated-prototype scale? |
|---|---|---|---|
| Brute-force exact search (pgvector exact / in-memory) | Simple to operate, exact (not approximate) results, no additional infrastructure | Comparison cost scales linearly with reference-set size — eventually a real bottleneck | **Yes** — the honest math above shows this is not currently a bottleneck |
| ANN index (pgvector HNSW/IVF, or a dedicated vector DB) | Sub-linear (or near-constant) search cost even at very large reference-set sizes | Approximate results (a small, tunable recall/accuracy trade-off), additional operational complexity (index building/maintenance, or an entirely separate system to operate) | **Not yet** — only once reference-set size genuinely grows into the hundreds-of-thousands-plus range |

## When to Use Which

- **Stay with brute-force exact search** at the taxonomy's current 50-class, curated-prototype
  scale — adopting ANN now would trade away exact results and add real operational complexity
  for a performance problem that doesn't currently exist.
- **Move to ANN search** only if reference-set size is deliberately grown (more examples per
  class, or a non-parametric design) past the point where the brute-force math above stops
  holding — and even then, prefer `pgvector`'s built-in ANN index support first, since it
  avoids introducing an entirely separate vector database system, reserving a dedicated vector
  DB for the point where `pgvector`'s own scaling limits are actually reached.

## Summary

At 50 classes with curated reference sets in the hundreds-to-low-thousands of vectors, brute-force
exact similarity search is not a bottleneck — the honest FLOP-level math shows this clearly,
and adopting a dedicated vector database or ANN index at this scale would trade away exact
results and add operational complexity for no real performance benefit. The genuine trigger for
ANN search is reference-set *size* growing into the hundreds-of-thousands-plus range (via more
examples per class or a non-parametric design), not class count alone — consistent with the
same "don't over-engineer ahead of an observed bottleneck" discipline applied to sharding in
Chapter 4.4 and multi-region in Chapter 7.4.