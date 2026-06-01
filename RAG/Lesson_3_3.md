# Lesson 3.3 — Hybrid Search Design and RRF vs. Score-Based Fusion

---

## Why Hybrid Search Is More Than "Run Both and Merge"

The naive mental model of hybrid search: run dense retrieval, run BM25, combine the results somehow. Done.

The reality is more nuanced. Hybrid search is a system design problem with several non-obvious decisions:

- How do you merge two ranked lists with incompatible score scales?
- When should dense results dominate? When should sparse results dominate?
- How do you tune this balance without overfitting to a test set?
- What happens when one retriever returns a result the other does not find at all?
- How does query type affect the optimal fusion strategy?

Getting these decisions right is the difference between hybrid search that consistently outperforms either retriever alone and hybrid search that sometimes helps and sometimes hurts.

---

## Understanding Why Each Retriever Fails

Before designing fusion, understand precisely what each retriever gets wrong. The complementarity of dense and sparse retrieval is not vague — it maps to specific, predictable failure patterns.

### Dense Retrieval Failure Patterns

**Vocabulary mismatch with rare or domain-specific terms:**
A user queries "HIPAA Section 164.312(a)(2)(iv)" — a specific regulatory citation. The dense embedding of this query produces a vector that may loosely point toward "healthcare compliance" content, but the specific regulation number may not be encoded meaningfully. Any chunk containing that exact citation will likely rank higher in BM25 than dense search.

**Named entity confusion:**
Product names, person names, and company names that are rare or proprietary often have poor embedding representations. "Does our SLA cover Prometheus monitoring?" — the dense model may conflate "Prometheus" (the monitoring tool) with other uses of the name. BM25 treats it as an exact token.

**Short queries with little semantic signal:**
A query of two or three words gives the embedding model very little to work with. "API rate limits" produces a query vector that is broadly "about API stuff" rather than precisely pointing at rate limiting content. BM25 will directly find chunks containing both "API" and "rate limits."

**When dense retrieval wins:**
Long, natural language questions. Queries using synonyms or paraphrases of document vocabulary. Cross-lingual queries. Conceptual questions ("explain the mechanism behind X").

### Sparse Retrieval (BM25) Failure Patterns

**Semantic mismatch:**
"What are the consequences of not meeting deadlines?" — BM25 looks for chunks containing "consequences", "meeting", "deadlines". A policy document section titled "Late Delivery Penalties" that perfectly answers the question contains none of those terms. BM25 score is zero; dense retrieval finds it easily.

**Multi-concept queries:**
"How do I set up two-factor authentication for the admin portal?" — BM25 gives high scores to chunks containing "two-factor" OR "authentication" OR "admin" OR "portal" in any context. A chunk about admin portal UI design scores high even though it is irrelevant. Dense retrieval evaluates the full query intent.

**Paraphrase blindness:**
Any time the user's vocabulary does not match the document's vocabulary, BM25 fails. This is the most common failure mode. Dense retrieval handles this by design.

**When BM25 wins:**
Exact code lookups, regulatory citations, product serial numbers, precise technical terms, error messages, specific named concepts, queries where exact term matching is the intent.

---

## The Score Incompatibility Problem

Dense retrieval produces cosine similarity scores in [-1, 1], typically concentrated in [0.6, 1.0] for reasonable matches.

BM25 produces scores in [0, ∞), typically in [0, 20] for a corpus of thousands of chunks but unbounded for larger corpora.

These scores are not comparable. You cannot simply add them: a dense score of 0.85 and a BM25 score of 12.3 cannot be meaningfully combined without normalization.

Two approaches to this problem:

**1. Normalize scores first, then combine** (score-based fusion).
**2. Ignore scores entirely and work only on ranks** (rank-based fusion, specifically RRF).

---

## Score-Based Fusion

Normalize each retriever's scores to [0, 1] using min-max normalization within the result set, then take a weighted combination.

```python
def normalize_scores(results: list[dict], score_key: str = "score") -> list[dict]:
    """Min-max normalize scores to [0, 1]."""
    scores = [r[score_key] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range == 0:
        # All scores equal — assign uniform normalized score
        for r in results:
            r[f"normalized_{score_key}"] = 1.0
    else:
        for r in results:
            r[f"normalized_{score_key}"] = (r[score_key] - min_score) / score_range
    
    return results

def score_based_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3
) -> list[dict]:
    """
    Weighted combination of normalized scores.
    dense_weight + sparse_weight should sum to 1.0.
    """
    assert abs(dense_weight + sparse_weight - 1.0) < 1e-6
    
    # Normalize each list independently
    dense_results = normalize_scores(dense_results, "dense_score")
    sparse_results = normalize_scores(sparse_results, "sparse_score")
    
    # Build lookup by chunk_id
    dense_lookup = {r["chunk_id"]: r["normalized_dense_score"] 
                    for r in dense_results}
    sparse_lookup = {r["chunk_id"]: r["normalized_sparse_score"] 
                     for r in sparse_results}
    
    # Collect all unique chunk_ids
    all_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
    
    fused = []
    for chunk_id in all_ids:
        dense_score = dense_lookup.get(chunk_id, 0.0)
        sparse_score = sparse_lookup.get(chunk_id, 0.0)
        combined = dense_weight * dense_score + sparse_weight * sparse_score
        
        fused.append({
            "chunk_id": chunk_id,
            "combined_score": combined,
            "dense_score": dense_score,
            "sparse_score": sparse_score
        })
    
    return sorted(fused, key=lambda x: x["combined_score"], reverse=True)
```

### The Problem with Score-Based Fusion

Min-max normalization within the result set creates a subtle but important instability.

Consider: dense retrieval returns 50 results. The minimum score is 0.62, maximum is 0.94. After normalization, the worst result gets score 0.0 and the best gets 1.0.

Now imagine a different query where all dense results cluster tightly between 0.88 and 0.91 (the query was easy, many highly relevant chunks exist). After normalization, the worst gets 0.0 and the best gets 1.0 — same normalized scale as before, even though the actual quality distribution was completely different.

This means the normalized score carries no information about the overall quality of the dense retrieval for this query. A dense search that found five excellent results looks the same as a dense search that found five mediocre ones after normalization.

Additionally, a chunk that appears in dense results but not sparse results has sparse score 0.0 — as if BM25 actively scored it zero. But BM25 did not score it zero; BM25 simply did not rank it in the top-K. There is a difference. Setting unseen results to 0.0 penalizes chunks that one retriever found but the other did not.

Score-based fusion with fixed weights can work well when:
- You have tuned the weights on a representative evaluation set.
- Both retrievers are reliably returning results for most queries (few misses).
- Score distributions are relatively stable across query types.

It becomes unreliable when query difficulty varies widely or when one retriever frequently misses.

---

## Reciprocal Rank Fusion (RRF)

RRF sidesteps the score incompatibility problem entirely by ignoring scores and working only on ranks.

### The Formula (Deep Dive)

For a set of ranked lists R and a constant k, the RRF score for document d is:

```
RRF(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))
```

Where `rank_r(d)` is the 1-based rank of document d in list r (1 = top result). If document d does not appear in list r, it contributes 0 to the sum (the term is simply omitted).

**Worked example:**

Three ranked lists: dense retrieval (D), sparse retrieval (S), and a third retrieval from a different query expansion (E). k = 60.

| chunk_id | rank in D | rank in S | rank in E | RRF score |
|----------|-----------|-----------|-----------|-----------|
| chunk_42 | 1 | 3 | 2 | 1/61 + 1/63 + 1/62 = 0.0489 |
| chunk_17 | 2 | 1 | — | 1/62 + 1/61 = 0.0325 |
| chunk_91 | — | 2 | 1 | 1/62 + 1/61 = 0.0325 |
| chunk_7  | 3 | — | 5 | 1/63 + 1/65 = 0.0312 |

chunk_42 wins because it appears highly in all three lists. chunk_17 and chunk_91 are tied despite one appearing in only two lists — consistent high ranking across two lists equals inconsistent ranking across three.

### Why k = 60

The constant k controls how much the ranking position matters relative to the smoothing floor.

Without k (or k=0):
- Rank 1 contributes 1/1 = 1.0
- Rank 2 contributes 1/2 = 0.5
- Rank 3 contributes 1/3 = 0.33

The gap between rank 1 and rank 2 is enormous (0.5 difference). A single first-place ranking dominates everything else.

With k = 60:
- Rank 1 contributes 1/61 ≈ 0.0164
- Rank 2 contributes 1/62 ≈ 0.0161
- Rank 3 contributes 1/63 ≈ 0.0159

Ranks 1, 2, and 3 are nearly indistinguishable. A document ranked 1st in one list and 5th in another beats a document ranked 1st in one list and absent in the other — by a meaningful but not overwhelming margin.

k = 60 is not magic — it was empirically found to work well across many retrieval benchmarks. You can tune it, but in practice k = 60 performs well across most scenarios. Smaller k emphasizes top-rank positions more; larger k flattens the distribution.

### RRF With More Than Two Lists

RRF extends naturally to any number of ranked lists — this is a significant practical advantage for multi-query retrieval (query expansion, sub-question decomposition):

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "chunk_id"
) -> list[dict]:
    """
    Merge any number of ranked lists using RRF.
    
    ranked_lists: list of ranked result lists, each item is a dict with id_key
    k: RRF constant (default 60)
    id_key: the key to use as chunk identifier
    """
    scores = {}      # chunk_id -> cumulative RRF score
    payloads = {}    # chunk_id -> metadata from first appearance
    
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            chunk_id = result[id_key]
            rrf_contribution = 1.0 / (k + rank + 1)  # rank is 0-based here
            
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                payloads[chunk_id] = result
            
            scores[chunk_id] += rrf_contribution
    
    # Build merged result list
    merged = [
        {
            id_key: chunk_id,
            "rrf_score": score,
            **payloads[chunk_id]
        }
        for chunk_id, score in scores.items()
    ]
    
    return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)


# Usage: merge dense, sparse, and two query expansion results
merged_results = reciprocal_rank_fusion([
    dense_results,
    sparse_results,
    expanded_query_1_results,
    expanded_query_2_results
], k=60)
```

### RRF Weaknesses

**Cannot express confidence.** A dense retrieval that found its top result with similarity 0.98 (extremely confident) looks the same to RRF as a retrieval where the top result had similarity 0.61 (weak). If the scores themselves carry meaningful signal (not just the ranks), RRF throws that signal away.

**Equal weight to all lists.** RRF gives equal weight to every ranked list in the fusion. If you know that dense retrieval is significantly more reliable than BM25 for your domain, RRF cannot express this preference without modification.

**Unseen documents.** A chunk that dense retrieval found at rank 48 but BM25 never found gets only 1/(60+48) contribution. A chunk BM25 found at rank 1 gets 1/61. The dense-only chunk might actually be more relevant, but RRF has no way to know — it just sees "found in one list at rank 48."

### Weighted RRF

An extension that addresses the equal-weight limitation:

```python
def weighted_rrf(
    ranked_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
    id_key: str = "chunk_id"
) -> list[dict]:
    """
    RRF with per-list weights. Useful when you know one retriever
    is more reliable than another.
    """
    assert len(ranked_lists) == len(weights)
    
    scores = {}
    payloads = {}
    
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(ranked_list):
            chunk_id = result[id_key]
            rrf_contribution = weight / (k + rank + 1)
            
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                payloads[chunk_id] = result
            scores[chunk_id] += rrf_contribution
    
    merged = [
        {id_key: cid, "rrf_score": score, **payloads[cid]}
        for cid, score in scores.items()
    ]
    
    return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)

# Dense retrieval weighted 2× more than sparse
merged = weighted_rrf(
    [dense_results, sparse_results],
    weights=[2.0, 1.0],
    k=60
)
```

---

## RRF vs. Score-Based Fusion: When to Use Which

| Scenario | Recommendation |
|---|---|
| Default / no tuning data | RRF — works well out of the box |
| You have a labeled evaluation set | Tune score weights on eval set |
| Highly variable query difficulty | RRF — score normalization is unstable across difficulty levels |
| Combining 3+ ranked lists | RRF — extends cleanly, score fusion gets messy |
| Query expansion (multiple query variants) | RRF — natural fit |
| Need to weight one retriever more | Weighted RRF or score fusion with tuned weights |
| Dense scores carry confidence signal | Score fusion preserves this; RRF discards it |

In practice, RRF is the safe default. It requires no tuning, handles any number of lists, is robust to score scale differences, and empirically performs well across diverse retrieval tasks. Use score-based fusion when you have sufficient labeled data to tune the weights and your score distributions are stable.

---

## Query-Adaptive Fusion

A more sophisticated approach: dynamically adjust the fusion strategy based on query characteristics. Different queries have different optimal dense/sparse balance.

```python
def classify_query_type(query: str) -> str:
    """
    Classify query to determine optimal retrieval strategy.
    Returns: 'keyword', 'semantic', or 'hybrid'
    """
    import re
    
    # Signals that BM25/sparse should dominate
    keyword_signals = [
        r'\b[A-Z]{2,}\d+\b',        # codes like ICD-10, HIPAA, ISO27001
        r'\b\d+\.\d+\.\d+\b',       # version numbers like 3.14.2
        r'"[^"]+"',                  # quoted exact phrases
        r'\b[A-Z][a-z]+[A-Z]\w*\b', # CamelCase (product names, APIs)
        r'\berror\s+\d+\b',         # error codes
        r'\b[A-Z_]{3,}\b'           # ALL_CAPS identifiers
    ]
    
    # Signals that dense should dominate
    semantic_signals = [
        r'\b(explain|describe|why|how does|what is the difference|compare|summarize)\b',
        r'\b(concept|principle|mechanism|process|approach)\b',
    ]
    
    keyword_score = sum(1 for pattern in keyword_signals 
                        if re.search(pattern, query))
    semantic_score = sum(1 for pattern in semantic_signals 
                         if re.search(pattern, query, re.IGNORECASE))
    
    if keyword_score >= 2:
        return 'keyword'
    elif semantic_score >= 1:
        return 'semantic'
    else:
        return 'hybrid'


async def adaptive_retrieve(query: str, ...) -> list[dict]:
    query_type = classify_query_type(query)
    
    if query_type == 'keyword':
        # BM25 dominant: high sparse weight, lower dense weight
        dense_results = await dense_retrieve(query, k=30)
        sparse_results = await sparse_retrieve(query, k=50)
        return weighted_rrf([dense_results, sparse_results], 
                            weights=[1.0, 3.0])
    
    elif query_type == 'semantic':
        # Dense dominant: skip or de-weight sparse
        dense_results = await dense_retrieve(query, k=50)
        sparse_results = await sparse_retrieve(query, k=20)
        return weighted_rrf([dense_results, sparse_results], 
                            weights=[3.0, 1.0])
    
    else:
        # Balanced hybrid
        dense_results = await dense_retrieve(query, k=50)
        sparse_results = await sparse_retrieve(query, k=50)
        return reciprocal_rank_fusion([dense_results, sparse_results])
```

The keyword classifier can be improved by using a small LLM or fine-tuned classifier instead of regex patterns. But regex-based classification adds zero latency and captures the most obvious cases reliably.

---

## Tuning Hybrid Retrieval

If you have a labeled evaluation set (query → relevant chunk IDs), you can tune your fusion parameters systematically.

### Building a Tuning Set

Collect 100–500 query-relevant_chunk pairs. Sources:
- User queries + clicks/thumbs-up ratings from your deployed system.
- Synthetic queries generated by LLM from known-answer chunks.
- Manual annotation by domain experts.

### Grid Search Over Fusion Parameters

```python
import numpy as np
from itertools import product

def evaluate_fusion(
    dense_results_by_query: dict,
    sparse_results_by_query: dict,
    ground_truth: dict,  # query_id -> list of relevant chunk_ids
    dense_weight: float,
    sparse_weight: float,
    k_eval: int = 10
) -> float:
    """Returns mean Recall@k across all test queries."""
    
    recalls = []
    for query_id, relevant_ids in ground_truth.items():
        dense = dense_results_by_query[query_id]
        sparse = sparse_results_by_query[query_id]
        
        fused = score_based_fusion(dense, sparse, dense_weight, sparse_weight)
        retrieved_ids = [r["chunk_id"] for r in fused[:k_eval]]
        
        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
        recalls.append(recall)
    
    return np.mean(recalls)

# Grid search
best_recall = 0
best_params = {}

for dense_w in np.arange(0.3, 0.9, 0.1):
    sparse_w = round(1.0 - dense_w, 1)
    
    recall = evaluate_fusion(
        dense_results_by_query,
        sparse_results_by_query,
        ground_truth,
        dense_weight=dense_w,
        sparse_weight=sparse_w
    )
    
    if recall > best_recall:
        best_recall = recall
        best_params = {"dense_weight": dense_w, "sparse_weight": sparse_w}

print(f"Best params: {best_params}, Recall@10: {best_recall:.3f}")
```

> **Interview note:** When asked "how did you tune your hybrid retrieval weights?", the answer they want: (1) build a labeled evaluation set of query-chunk pairs, (2) measure recall@K for different weight combinations, (3) pick the combination with highest recall on your eval set, (4) validate on a held-out test set to confirm you did not overfit. Never say "I tried a few values and picked what felt right."

---

## Practical Hybrid Search Architecture

A production hybrid search system needs more than just the fusion logic. Here is the full picture:

```
Query arrives
    │
    ├─── Dense retrieval ─────────────────────────────────────────────┐
    │    │                                                             │
    │    ├── Embed query (embedding model)                             │
    │    └── ANN search (vector DB, with metadata filter)             │
    │                                                                  │
    ├─── Sparse retrieval ─────────────────────────────────────────── Merge (RRF)
    │    │                                                             │
    │    ├── Tokenize query                                            │
    │    └── BM25/SPLADE search (inverted index or Qdrant sparse)     │
    │                                                                  │
    └─── [Optional] Query expansion results ─────────────────────────┘
                                                    │
                                              Top-N candidates
                                                    │
                                            Cross-encoder re-ranking
                                                    │
                                              Final top-K results
                                                    │
                                            Context assembly + LLM
```

Key implementation decisions:
- Dense and sparse retrieval run **in parallel**, not sequentially. This halves the retrieval latency.
- Run RRF before re-ranking. Re-ranking is expensive — run it on the merged top-N, not separately on each retriever's output.
- Metadata filters apply to dense retrieval (pre-filtered ANN search). For BM25, apply filters as post-retrieval filtering or include them in the BM25 query as required terms.

---

## Summary

- Dense retrieval fails on exact terms, codes, identifiers, and rare vocabulary. Sparse retrieval fails on semantic paraphrases and synonym-rich queries. These failure modes are predictable and complementary.
- Score-based fusion: normalize scores to [0,1], combine with weights. Intuitive but unstable across variable query difficulty. Requires tuning on labeled data.
- RRF: ignore scores, combine only ranks using 1/(k + rank). Robust, requires no tuning, extends to any number of ranked lists. The safe default for most systems.
- RRF constant k=60 smooths rank differences. Smaller k emphasizes top positions more; larger k flattens everything.
- Weighted RRF extends standard RRF to express known differences in retriever reliability.
- Query-adaptive fusion dynamically adjusts dense/sparse balance based on query characteristics — regex-based classification adds zero latency.
- Tune fusion weights using a labeled evaluation set and grid search. Measure recall@K and NDCG@K, validate on a held-out test set.
- Always parallelize dense and sparse retrieval. Apply RRF before cross-encoder re-ranking, not after.

---

## What's Next

Lesson 3.4 covers query understanding in depth — query rewriting, expansion, decomposition, and the design decisions that determine whether your query understanding pipeline helps or hurts retrieval quality.