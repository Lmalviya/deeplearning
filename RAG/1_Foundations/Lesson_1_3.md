# Lesson 1.3 — Anatomy of a Retrieval Pipeline

---

## Overview

The retrieval pipeline is everything that happens at query time — from the moment a user submits a question to the moment the LLM receives its prompt. This is the pipeline that runs live, for every single query, under latency pressure.

Your current pipeline looks like this:

```
User Query
    ↓
Query Rewrite / Expansion
    ↓
Hybrid Retrieval (Dense + BM25)
    ↓
RRF (Reciprocal Rank Fusion)
    ↓
Re-ranking (Cross-Encoder)
    ↓
Generation (LLM)
```

We will go through each stage in depth — what it does, why it exists, how it works internally, what happens when it fails, and what the design decisions are at each step.

---

## Stage 1 — Query Understanding

The raw user query is almost never ideal for retrieval. Users write queries the way they talk, not the way documents are written. They use shorthand, make typos, ask ambiguous questions, or phrase things in a way that a keyword or embedding search handles poorly.

Query understanding is the set of techniques that transform the raw query into a better retrieval signal before touching the index.

### Query Rewriting

A rewriter takes the raw query and produces a cleaner, more precise version using an LLM.

**Example:**
- Raw: "how do i get out of my contract early"
- Rewritten: "early termination clause, contract exit conditions, penalty for breaking contract"

The rewriter is typically a small, fast LLM call with a prompt like: "Rewrite the following user question to be more suitable for document retrieval. Make it specific and use formal terminology."

**Design decisions:**
- Do you rewrite in place (one rewritten query replaces the original), or do you keep both?
- Keeping both is safer — the original query may contain nuance the rewriter drops.
- Rewriting adds latency (one LLM call). Use a small, fast model (not your most capable one).

**When it fails:**
- The rewriter changes the meaning of the query, not just the phrasing.
- The rewriter is too conservative and barely changes anything.
- The rewriter is too aggressive and loses specific terms the user cared about (product names, codes, identifiers).

### Query Expansion

Instead of one rewritten query, expansion generates multiple related queries. You retrieve for each of them separately, then merge the results.

**Example:**
- Original: "machine learning model deployment"
- Expanded: ["ML model serving in production", "deploying neural networks at scale", "model inference infrastructure", "MLOps deployment pipeline"]

This increases **recall** — you catch relevant chunks that would have been missed by a single query. The cost is higher retrieval latency (multiple searches) and potentially more noise.

**Techniques for expansion:**
- LLM-generated synonyms and related phrasings.
- Step-back prompting — generate a more general version of the query ("what broad topic does this fall under?") and retrieve for both specific and general.
- HyDE — generate a hypothetical answer and embed that. Covered in Lesson 3.5.

### Sub-question Decomposition

For complex queries, decompose into atomic sub-questions. Each sub-question targets a specific retrievable fact.

**Example:**
- Original: "Compare the pricing and refund policies of our Enterprise and Pro plans"
- Decomposed: ["What is the pricing for the Enterprise plan?", "What is the pricing for the Pro plan?", "What is the refund policy for the Enterprise plan?", "What is the refund policy for the Pro plan?"]

Each sub-question gets its own retrieval pass. The results are merged and passed together to the LLM for synthesis.

**When to use decomposition:** When the query contains multiple distinct information needs that would require different chunks to answer. A single retrieval pass would either miss some of them or return an incoherent mix.

---

## Stage 2 — Hybrid Retrieval

After query understanding, you search the index. Hybrid retrieval means running two fundamentally different search mechanisms in parallel and combining their results.

### Dense Retrieval (Semantic Search)

Dense retrieval works by comparing vector representations.

At index time, each chunk is passed through an **embedding model** — a neural network that converts text into a high-dimensional vector (typically 768 to 3072 dimensions). This vector captures the *semantic meaning* of the chunk. Semantically similar text produces vectors that are close together in this high-dimensional space.

At query time, the query is embedded using the same model. You then find the chunks whose vectors are most similar to the query vector. The similarity metric is almost always **cosine similarity** (or equivalently, dot product for normalized vectors).

The search itself is done using **Approximate Nearest Neighbor (ANN)** algorithms — you are not doing an exact exhaustive comparison of the query vector against every chunk (too slow at scale). Instead, the vector database uses index structures like HNSW (Hierarchical Navigable Small World graphs) to find approximate nearest neighbors in milliseconds. We cover this in depth in Lesson 3.1.

**What dense retrieval is good at:**
- Semantic matching: "heart attack" ↔ "myocardial infarction"
- Paraphrase matching: same concept, different words
- Cross-lingual retrieval (with multilingual embedding models)
- Finding conceptually related content even with no keyword overlap

**What dense retrieval is bad at:**
- Exact keyword matching: rare terms, codes, product names, identifiers
- Retrieval for very short queries (little signal for the embedding model)
- Out-of-distribution queries — if the query style is very different from training data, embedding quality drops

### Sparse Retrieval (BM25)

BM25 is a classical information retrieval algorithm. It does not use neural networks or embeddings. It works entirely on token frequency statistics.

BM25 scores a document against a query based on:
- How often query terms appear in the document (**term frequency**).
- How rare those terms are across the entire corpus (**inverse document frequency** — rare terms are more informative).
- Document length normalization (so longer documents do not get unfair advantage just from having more tokens).

The formula in simplified terms: a chunk scores high if the query's words appear frequently in it and those words are rare across the whole corpus.

**What BM25 is good at:**
- Exact keyword matching: product codes, person names, legal clause numbers, medical codes
- Rare or specific terms that the embedding model may not encode well
- Short, specific queries
- Cases where the user knows the exact terminology in the document

**What BM25 is bad at:**
- Semantic similarity with no keyword overlap
- Synonyms and paraphrases
- Conceptual or thematic queries

### Why You Need Both

Dense and sparse retrieval have almost **complementary failure modes**. When one fails, the other often succeeds. This is the core justification for hybrid retrieval — not that each method is mediocre, but that they fail on different types of queries.

A healthcare system searching through clinical notes: "myocardial infarction treatment protocol" — dense retrieval finds documents about heart attacks even if they never use the exact phrase. But when a clinician searches for a specific drug code like "ICD-10 I21.9" — BM25 finds it exactly, while the embedding model may not have encoded that specific code well.

Running both and merging the results covers far more of the query space than either alone.

---

## Stage 3 — Reciprocal Rank Fusion (RRF)

You now have two ranked lists of retrieved chunks — one from dense retrieval, one from BM25. Possibly more if you did query expansion and retrieved for multiple queries. You need to merge these into a single ranked list.

The naive approach — average the scores — does not work well because the score scales are completely different. A dense retrieval cosine similarity of 0.85 and a BM25 score of 23.4 are not comparable numbers. Normalizing scores is tricky and brittle.

**RRF solves this elegantly.** It ignores the actual scores entirely and works only on ranks.

The RRF formula for a chunk `d` given a set of ranked lists `R`:

```
RRF_score(d) = Σ  1 / (k + rank_r(d))
              r∈R
```

Where `k` is a constant (typically 60) and `rank_r(d)` is the rank of chunk `d` in list `r`.

**Intuition:** A chunk that appears at rank 1 in one list and rank 3 in another gets a much higher RRF score than a chunk that appears at rank 1 in one list and nowhere in another. Consistent high ranking across multiple lists is rewarded.

**Why k=60?** This constant smooths out the difference between rank 1 and rank 2 (which would otherwise be huge: 1/1 vs 1/2). With k=60, rank 1 gives 1/61 ≈ 0.016 and rank 2 gives 1/62 ≈ 0.016 — very close. This prevents the top-ranked item from dominating and gives more weight to consistently good performers across lists.

**Properties of RRF:**
- Score-agnostic: works regardless of how different the score scales are.
- Robust: a chunk that ranks highly in multiple lists consistently wins.
- Simple: no normalization, no hyperparameter tuning of score weights.
- Well-studied: has been shown to outperform linear score combination in many IR benchmarks.

**Alternative to RRF:** Weighted linear combination of normalized scores. This gives you explicit control over how much to weight dense vs. sparse retrieval, but requires careful tuning and is sensitive to score distribution changes.

---

## Stage 4 — Re-ranking

After RRF, you have a merged ranked list of, say, top-50 chunks. Re-ranking is the step that takes this coarse list and produces a much more accurate final ranking.

### Why Retrieval Ranking Is Coarse

Both BM25 and embedding similarity are designed to be fast — they need to search millions of chunks in milliseconds. This speed comes at a cost: they use relatively simple comparison mechanisms (keyword frequency statistics, single-vector dot product). They are good at getting the right chunks into the top-50 or top-100. They are not good at precisely ordering those 50 chunks.

### Cross-Encoder Re-ranking

A cross-encoder is a neural model that takes a (query, chunk) pair as input and outputs a single relevance score. Unlike embedding models (which encode query and chunk independently), the cross-encoder sees both together — it can model fine-grained interactions between query and chunk text.

This joint encoding makes cross-encoders much more accurate at relevance scoring. They can understand nuances like: the chunk uses the query's term but in a different context (low relevance), or the chunk answers the query's intent even without exact keyword overlap (high relevance).

**The trade-off:** Cross-encoders are slow. You cannot use them to search through millions of chunks — they require one full forward pass per (query, chunk) pair. This is why you use them only on the small shortlist (top-20 to top-50) that fast retrieval already found. The pipeline structure is:

```
Fast retrieval (ANN + BM25): search millions → return top-50
Cross-encoder: score top-50 precisely → return top-5 to top-10
```

This two-stage design gives you the speed of approximate retrieval with the accuracy of cross-encoder scoring.

**Popular cross-encoder models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, good general performance
- `cross-encoder/ms-marco-electra-base` — more accurate, slower
- Cohere Rerank API — managed service, strong performance, no infrastructure to run

### LLM as Re-ranker

An emerging approach is using the LLM itself as a re-ranker. You pass the query and all candidate chunks to the LLM and ask it to rank them by relevance. This can be more accurate than cross-encoders for nuanced relevance judgments, but is significantly more expensive and slower.

Practical technique: **listwise re-ranking** — ask the LLM to output a ranked list of chunk IDs given the query. Or **pointwise** — score each chunk independently with a relevance prompt.

### What Re-ranking Catches

Re-ranking most commonly corrects these errors from initial retrieval:
- A chunk that contains the query's keywords prominently but answers a different question.
- A chunk that is semantically close to the query topic but not the specific answer the query seeks.
- A chunk that is genuinely the best answer but was ranked 15th by embedding similarity due to phrasing differences.

In practice, re-ranking consistently improves NDCG (normalized discounted cumulative gain) by 10-30% over just taking the top-K from retrieval. For high-stakes applications, this improvement is not optional.

---

## Stage 5 — Context Assembly

After re-ranking, you have your final set of chunks — typically 3 to 10. Before passing them to the LLM, you need to assemble them into a coherent context block.

This step is often ignored in tutorials but matters in production.

### Ordering

How should you order the chunks in the prompt? Research (Liu et al., "Lost in the Middle") shows that LLMs pay more attention to content at the beginning and end of the context, and less to the middle. 

Two strategies:
- **Relevance order** — most relevant chunk first. Simple, intuitive.
- **Sandwich order** — most relevant first, second most relevant last, others in the middle. Exploits the primacy and recency effect.

### Deduplication

If you ran query expansion with multiple sub-queries, you may have retrieved the same chunk multiple times. Deduplicate before assembly — sending the same chunk twice wastes context tokens and can confuse the model.

### Source Tagging

Tag each chunk with its source metadata (document title, section, page number, date). This allows the LLM to cite sources in its answer, which is critical for trust and auditability.

```
[Source: HR Policy Manual, Section 4.2, Updated Jan 2024]
Employees are entitled to 16 weeks of parental leave...

[Source: Benefits Guide 2024, Page 12]
Parental leave runs concurrently with any applicable state leave...
```

### Handling Context Length

If your re-ranked chunks together exceed the LLM's context window (or your cost budget), you need to truncate. Strategies:
- Drop lowest-relevance chunks first.
- Truncate individual chunks from their ends (documents often front-load key information).
- Use a summarization step to compress each chunk before assembly.

---

## Stage 6 — Generation

The assembled context and the user query go into the LLM's prompt. The LLM generates the final answer.

At this stage, the quality of the answer depends on:

**Prompt design** — Does the prompt instruct the model to stay grounded in the provided context? Does it tell the model what to do when the context does not contain the answer ("say I don't know" vs. "use your general knowledge as a fallback")? Does it specify output format (bullet list, prose, JSON)?

**Faithfulness** — The model should answer from the retrieved context, not from its parametric memory. A well-designed prompt reinforces this: "Answer the question using only the provided context. If the answer is not in the context, say so."

**Citation** — For high-stakes domains, ask the model to cite the specific source for each claim. This makes answers auditable and helps users verify.

**Post-processing** — The raw LLM output may need cleaning: stripping markdown if the output is going to a UI that does not render it, extracting structured fields if the answer should be JSON, validating citations reference real sources.

---

## The Full Pipeline with Failure Modes

Here is the complete pipeline with the most common failure mode at each stage:

| Stage | Purpose | Most Common Failure |
|---|---|---|
| Query Rewrite | Improve retrieval signal | Rewrites lose specific terms or change meaning |
| Query Expansion | Increase recall | Too many expansions add noise and slow retrieval |
| Dense Retrieval | Semantic matching | Fails on rare terms, codes, identifiers |
| BM25 Retrieval | Keyword matching | Fails on paraphrases and synonyms |
| RRF Fusion | Merge ranked lists | Poor results when one retrieval method dominates |
| Cross-Encoder Re-ranking | Accurate final ranking | Slow; can be bottleneck at high QPS |
| Context Assembly | Build LLM prompt | Noisy or unordered context confuses the LLM |
| Generation | Produce answer | Model ignores context and hallucinates from memory |

---

## Latency Budget

At production scale, every stage adds latency. A realistic latency budget for your pipeline:

| Stage | Typical Latency |
|---|---|
| Query rewrite (small LLM) | 100–300ms |
| Dense embedding of query | 10–50ms |
| ANN vector search | 5–30ms |
| BM25 search | 5–20ms |
| RRF merge | < 1ms |
| Cross-encoder re-ranking (top-50) | 50–200ms |
| LLM generation | 500ms–3s |
| **Total** | **~1–4 seconds** |

The LLM generation step dominates. Re-ranking is the second biggest cost. Query rewriting adds a meaningful overhead. This is why production systems often parallelize where possible — run dense and sparse retrieval simultaneously, not sequentially.

> **Interview note:** When asked about optimizing RAG latency, the answer is: (1) parallelize retrieval stages, (2) use a faster/smaller re-ranker or reduce the re-ranking candidate set, (3) cache query embeddings and results for repeated queries, (4) use streaming generation so the user sees tokens as they arrive rather than waiting for the full response.

---

## Summary

- Query understanding (rewrite, expand, decompose) transforms raw queries into better retrieval signals before touching the index.
- Dense retrieval handles semantic similarity; BM25 handles keyword matching. They fail on different queries — run both.
- RRF merges ranked lists from multiple retrieval sources without needing to normalize incompatible scores.
- Cross-encoder re-ranking runs on a small shortlist to produce an accurate final ranking — much more precise than embedding similarity but too slow for full-corpus search.
- Context assembly — ordering, deduplication, source tagging, length management — affects generation quality in ways that are easy to overlook.
- Generation quality depends on prompt design, faithfulness enforcement, and post-processing.
- The entire pipeline has a realistic latency of 1–4 seconds. Parallelism and caching are the primary levers for reducing it.

---