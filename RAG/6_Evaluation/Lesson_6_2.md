# Lesson 6.2 — Retrieval Metrics in Depth: Precision@K, Recall@K, MRR, MAP, NDCG, Hit Rate, Coverage

---

## Why Retrieval Metrics Matter Independently

Retrieval and generation are separate failure points. A RAG system can fail because retrieval is poor (wrong chunks retrieved) or because generation is poor (right chunks retrieved, wrong answer generated). Conflating these makes debugging impossible — if you only measure final answer quality, you cannot tell which stage to fix.

Retrieval metrics measure only the retrieval stage: given a query, did the system find the right chunks? They require knowing the "correct" chunks for each query — the ground truth. Building this ground truth is work, but it is essential for systematic retrieval improvement.

All retrieval metrics assume you have, for each query:
- A set of **relevant chunks** (ground truth): the chunks that actually contain the answer.
- A ranked list of **retrieved chunks**: what your system returned.

---

## Setting Up the Evaluation Framework

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RetrievalResult:
    query_id: str
    query: str
    retrieved_chunk_ids: list[str]     # Ordered by rank (index 0 = rank 1)
    retrieved_chunk_scores: list[float]  # Corresponding relevance scores
    
@dataclass  
class GroundTruth:
    query_id: str
    relevant_chunk_ids: list[str]          # Binary: these chunks are relevant
    graded_relevance: Optional[dict] = None  # chunk_id -> relevance score (0-3 scale)

def evaluate_retrieval(
    results: list[RetrievalResult],
    ground_truth: list[GroundTruth],
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """Compute all retrieval metrics across a set of queries."""
    
    gt_map = {gt.query_id: gt for gt in ground_truth}
    
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"hit_rate@{k}": [] for k in k_values})
    metrics["mrr"] = []
    metrics["map"] = []
    metrics["ndcg@10"] = []
    
    for result in results:
        gt = gt_map.get(result.query_id)
        if not gt:
            continue
        
        relevant_set = set(gt.relevant_chunk_ids)
        
        for k in k_values:
            top_k = result.retrieved_chunk_ids[:k]
            
            # Precision@K
            hits_at_k = sum(1 for cid in top_k if cid in relevant_set)
            metrics[f"precision@{k}"].append(hits_at_k / k)
            
            # Recall@K
            metrics[f"recall@{k}"].append(
                hits_at_k / len(relevant_set) if relevant_set else 0
            )
            
            # Hit Rate@K
            metrics[f"hit_rate@{k}"].append(
                1.0 if any(cid in relevant_set for cid in top_k) else 0.0
            )
        
        # MRR
        metrics["mrr"].append(compute_mrr(result.retrieved_chunk_ids, relevant_set))
        
        # MAP
        metrics["map"].append(compute_ap(result.retrieved_chunk_ids, relevant_set))
        
        # NDCG@10
        if gt.graded_relevance:
            metrics["ndcg@10"].append(
                compute_ndcg(result.retrieved_chunk_ids, gt.graded_relevance, k=10)
            )
    
    # Aggregate
    return {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
```

---

## Metric 1 — Precision@K

**Definition:** Of the K chunks retrieved, what fraction are relevant?

```
Precision@K = |relevant ∩ retrieved_top_K| / K
```

**Example:**
- Retrieved top-5: [chunk_A, chunk_B, chunk_C, chunk_D, chunk_E]
- Relevant: {chunk_A, chunk_C, chunk_F}
- Relevant in top-5: chunk_A, chunk_C → 2 hits
- Precision@5 = 2/5 = 0.40

**Interpretation:** Precision measures how much of what you retrieved was actually useful. Low precision means many retrieved chunks are irrelevant, wasting the LLM's context window.

**When it is high:** Your retrieval is precise — most retrieved chunks are on-topic.
**When it is low:** Retrieval is noisy — lots of irrelevant chunks in the top-K.

**Limitation:** Precision penalizes you for missing relevant documents. If there are 10 relevant chunks and you retrieve 5 of them correctly, Precision@5 = 1.0 but you missed 5 relevant chunks. Precision does not care about coverage.

```python
def compute_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k if k > 0 else 0.0
```

---

## Metric 2 — Recall@K

**Definition:** Of all relevant chunks, what fraction are in the top-K retrieved?

```
Recall@K = |relevant ∩ retrieved_top_K| / |relevant|
```

**Example (same as above):**
- Relevant: {chunk_A, chunk_C, chunk_F} → 3 total relevant
- Relevant in top-5: chunk_A, chunk_C → 2 hits
- Recall@5 = 2/3 = 0.67

**Interpretation:** Recall measures coverage — did you find most of the relevant chunks? Low recall means relevant chunks are being missed, even if those retrieved are good.

**When it is high:** Your index contains the relevant chunks and retrieval finds them.
**When it is low:** Either relevant chunks are missing from the index (coverage gap) or retrieval fails to find them (retrieval failure).

**The precision-recall trade-off:** Increasing K increases recall but decreases precision. More retrieved chunks means more chances to find relevant ones, but also more irrelevant ones. The optimal K balances coverage with noise.

**For RAG specifically:** Recall@K matters more than Precision@K for most applications. It is worse to miss a relevant chunk (and produce an incomplete answer) than to retrieve an irrelevant chunk (which a good LLM can largely ignore). Optimize for recall, then control noise through re-ranking.

```python
def compute_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0  # If no relevant chunks, recall is trivially 1
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)
```

---

## Metric 3 — Hit Rate@K (also called Recall@K binary or Success@K)

**Definition:** For a query, did the retrieval return at least one relevant chunk in the top-K?

```
Hit Rate@K = 1 if |relevant ∩ retrieved_top_K| > 0 else 0
```

Averaged across all queries: the fraction of queries for which at least one relevant chunk was retrieved.

**Interpretation:** Hit rate is the most forgiving metric — it only asks whether the system found any relevant chunk, not how many or where they are ranked. It answers: "What fraction of queries does retrieval help at all?"

**When it matters:** For RAG, if hit rate is below 0.90 at K=10, your retrieval is systematically failing on many queries. Something fundamental is wrong — coverage gaps, embedding mismatch, or query understanding failures.

**Target:** Hit Rate@5 above 0.90, Hit Rate@10 above 0.95 for a production system on in-scope queries.

```python
def compute_hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0
```

---

## Metric 4 — MRR (Mean Reciprocal Rank)

**Definition:** For each query, find the rank of the first relevant chunk. Take its reciprocal. Average across queries.

```
MRR = (1/|Q|) × Σ 1/rank_of_first_relevant(q)
```

**Example:**
- Query 1: first relevant chunk is at rank 1 → 1/1 = 1.0
- Query 2: first relevant chunk is at rank 3 → 1/3 = 0.33
- Query 3: no relevant chunk retrieved → 1/∞ = 0.0
- MRR = (1.0 + 0.33 + 0.0) / 3 = 0.44

**Interpretation:** MRR rewards finding a relevant chunk quickly. Rank 1 is perfect; rank 3 gives you a third of the credit; rank 10 gives you 1/10. If no relevant chunk is found, the query contributes zero.

**When it matters:** MRR is the right metric when you care about finding at least one good answer fast — like when the LLM uses only the top-1 or top-3 chunks for generation.

**When it is misleading:** MRR ignores everything after the first relevant chunk. A system that always puts one relevant chunk at rank 1 and misses all others gets the same MRR as a system that puts all relevant chunks in the top 5.

```python
def compute_mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0
```

---

## Metric 5 — MAP (Mean Average Precision)

**Definition:** For each query, compute Average Precision (AP). Average AP across queries.

Average Precision is the area under the precision-recall curve for a single query:

```
AP = (1/|relevant|) × Σ Precision@k × Relevance(k)

Where Relevance(k) = 1 if chunk at rank k is relevant, else 0
```

**Example:**
- Relevant chunks: {A, C, E}
- Retrieved: [A, B, C, D, E, F]
- Rank 1 (A): relevant → Precision@1 = 1/1 = 1.0
- Rank 2 (B): not relevant → skip
- Rank 3 (C): relevant → Precision@3 = 2/3 = 0.67
- Rank 4 (D): not relevant → skip
- Rank 5 (E): relevant → Precision@5 = 3/5 = 0.60
- AP = (1.0 + 0.67 + 0.60) / 3 = 0.76

**Interpretation:** MAP rewards:
- Finding relevant chunks early (high rank → high precision contribution)
- Finding all relevant chunks (missing relevant chunks hurts recall, which hurts AP)

MAP is sensitive to both ranking order and coverage. It is the most comprehensive binary relevance metric.

```python
def compute_ap(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for rank, chunk_id in enumerate(retrieved, 1):
        if chunk_id in relevant:
            hits += 1
            sum_precisions += hits / rank
    
    return sum_precisions / len(relevant)
```

---

## Metric 6 — NDCG (Normalized Discounted Cumulative Gain)

**Definition:** A graded relevance metric that rewards finding highly relevant chunks at high ranks.

Unlike the binary metrics above (relevant/not relevant), NDCG supports graded relevance: some chunks are more relevant than others. A chunk that directly answers the query is more valuable than a chunk that only tangentially relates to it.

```
DCG@K = Σ (2^relevance_score - 1) / log2(rank + 1)

IDCG@K = DCG@K of the ideal ranking (perfect order)

NDCG@K = DCG@K / IDCG@K
```

**Graded relevance scale for RAG:**
- 3 = Directly answers the question completely
- 2 = Contains the answer but requires inference
- 1 = Partially relevant, provides some context
- 0 = Not relevant

**Example:**
- Retrieved with scores: [(chunk_A, rel=3), (chunk_B, rel=0), (chunk_C, rel=2)]
- DCG@3 = (2³-1)/log₂(2) + (2⁰-1)/log₂(3) + (2²-1)/log₂(4)
         = 7/1 + 0/1.58 + 3/2
         = 7 + 0 + 1.5 = 8.5
- Ideal ranking (best chunks first): [(rel=3), (rel=2)]
- IDCG@3 = 7/1 + 3/1.58 = 7 + 1.9 = 8.9
- NDCG@3 = 8.5 / 8.9 = 0.955

**Interpretation:** NDCG@K = 1.0 means perfect ranking — the most relevant chunks are at the top. NDCG measures both whether you found relevant chunks and whether you ranked them well.

**When NDCG is preferred over MAP:**
- When relevance has degrees (not binary)
- When ranking order matters (e.g., the LLM pays more attention to the first chunk)
- When evaluating re-ranker quality specifically

```python
def compute_ndcg(
    retrieved: list[str],
    graded_relevance: dict[str, int],  # chunk_id -> relevance score
    k: int = 10
) -> float:
    """Compute NDCG@K given graded relevance scores."""
    
    def dcg(ranking: list[str], k: int) -> float:
        gain = 0.0
        for rank, chunk_id in enumerate(ranking[:k], 1):
            rel = graded_relevance.get(chunk_id, 0)
            gain += (2**rel - 1) / np.log2(rank + 1)
        return gain
    
    # Actual DCG
    actual_dcg = dcg(retrieved, k)
    
    # Ideal DCG: sort by relevance descending
    ideal_order = sorted(
        graded_relevance.keys(),
        key=lambda cid: graded_relevance[cid],
        reverse=True
    )
    ideal_dcg = dcg(ideal_order, k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg
```

---

## Metric 7 — Coverage

**Definition:** The fraction of queries for which the relevant chunk exists in the index at all, regardless of whether retrieval finds it.

```
Coverage = queries where relevant chunk is indexed / total queries
```

Coverage is not a retrieval quality metric — it is an indexing quality metric. If coverage is below 1.0, some queries are unanswerable by design because the relevant content was never indexed.

```python
async def compute_coverage(
    ground_truth_queries: list[GroundTruth],
    vector_db,
    embedding_model
) -> dict:
    """
    Check what fraction of relevant chunks are actually in the index.
    """
    
    total_relevant_chunks = 0
    indexed_count = 0
    missing_by_query = []
    
    for gt in ground_truth_queries:
        query_missing = []
        
        for chunk_id in gt.relevant_chunk_ids:
            total_relevant_chunks += 1
            
            # Check if this chunk exists in the index
            exists = await vector_db.chunk_exists(chunk_id)
            
            if exists:
                indexed_count += 1
            else:
                query_missing.append(chunk_id)
        
        if query_missing:
            missing_by_query.append({
                "query_id": gt.query_id,
                "missing_chunks": query_missing
            })
    
    coverage = indexed_count / total_relevant_chunks if total_relevant_chunks > 0 else 0
    
    return {
        "coverage": coverage,
        "indexed_chunk_count": indexed_count,
        "total_relevant_chunks": total_relevant_chunks,
        "missing_count": total_relevant_chunks - indexed_count,
        "affected_queries": len(missing_by_query),
        "missing_by_query": missing_by_query
    }
```

**Target:** Coverage should be 1.0. If it is below 0.95, you have systematic indexing gaps that are causing retrieval failures that no retrieval improvement can fix.

---

## Choosing Which Metrics to Report

For a RAG system, I recommend this standard reporting set:

**Primary retrieval metrics (always report):**
- Hit Rate@5 and Hit Rate@10 — answers "do users get any relevant result?"
- Recall@5 and Recall@10 — answers "do users get all the relevant results they need?"
- MRR — answers "is the most relevant chunk near the top?"

**Secondary metrics (report for specific evaluations):**
- Precision@5 — when context noise is a concern
- MAP — when you want a single comprehensive binary metric
- NDCG@10 — when evaluating re-ranker specifically (requires graded relevance)
- Coverage — when investigating indexing quality

**Interpretation guide:**

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| Hit Rate@5 | > 0.90 | 0.80–0.90 | < 0.80 |
| Recall@5 | > 0.70 | 0.55–0.70 | < 0.55 |
| Recall@10 | > 0.85 | 0.70–0.85 | < 0.70 |
| MRR | > 0.75 | 0.60–0.75 | < 0.60 |
| NDCG@10 | > 0.80 | 0.65–0.80 | < 0.65 |

These thresholds are general — calibrate to your domain. Medical or legal systems may need higher bars.

---

## Computing the Full Metrics Report

```python
async def full_retrieval_evaluation(
    eval_dataset: list[dict],  # [{query, relevant_chunk_ids, graded_relevance?}]
    retriever,
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """
    Run the full retrieval evaluation pipeline.
    """
    
    results = []
    ground_truths = []
    
    for item in eval_dataset:
        # Retrieve
        retrieved = await retriever.retrieve(item["query"], k=max(k_values))
        
        results.append(RetrievalResult(
            query_id=item["query_id"],
            query=item["query"],
            retrieved_chunk_ids=[r["chunk_id"] for r in retrieved],
            retrieved_chunk_scores=[r.get("rerank_score", 0) for r in retrieved]
        ))
        
        ground_truths.append(GroundTruth(
            query_id=item["query_id"],
            relevant_chunk_ids=item["relevant_chunk_ids"],
            graded_relevance=item.get("graded_relevance")
        ))
    
    # Compute all metrics
    metrics = evaluate_retrieval(results, ground_truths, k_values)
    
    # Find worst-performing queries for debugging
    query_metrics = []
    gt_map = {gt.query_id: gt for gt in ground_truths}
    
    for result in results:
        gt = gt_map[result.query_id]
        relevant = set(gt.relevant_chunk_ids)
        
        query_hit_rate = compute_hit_rate_at_k(result.retrieved_chunk_ids, relevant, k=10)
        query_recall = compute_recall_at_k(result.retrieved_chunk_ids, relevant, k=10)
        
        query_metrics.append({
            "query_id": result.query_id,
            "query": result.query,
            "hit_rate@10": query_hit_rate,
            "recall@10": query_recall,
            "mrr": compute_mrr(result.retrieved_chunk_ids, relevant)
        })
    
    # Sort by performance to find worst cases
    worst_queries = sorted(query_metrics, key=lambda x: x["recall@10"])[:10]
    
    return {
        "aggregate_metrics": metrics,
        "worst_performing_queries": worst_queries,
        "query_count": len(eval_dataset)
    }
```

---

## Interpreting Metric Combinations

Individual metrics tell you something. Combinations tell you more:

**High Recall@10, Low Precision@10:**
Your retrieval finds the relevant chunks but also retrieves many irrelevant ones. Fix: improve re-ranking to push irrelevant chunks lower, or reduce K if noise is hurting generation.

**Low Recall@10, High Precision@10:**
Retrieved chunks are very precise but you are missing many relevant ones. Fix: increase K, improve query expansion to cast a wider net, check if some relevant chunks are missing from the index.

**High Hit Rate@5, Low MRR:**
The system finds a relevant chunk most of the time, but it is usually not at the top. Fix: improve ranking — better embedding model, better re-ranker, or better query-chunk alignment.

**Low Hit Rate@5 for specific query types:**
Systematic failure on certain query patterns. Fix: investigate the failing queries — are they out-of-scope? Is relevant content missing from the index? Is the embedding model failing on this vocabulary?

---

## Summary

- Retrieval metrics require ground truth: for each query, which chunks are actually relevant?
- Precision@K: what fraction of retrieved chunks are relevant? Measures noise.
- Recall@K: what fraction of relevant chunks are retrieved? Measures coverage.
- Hit Rate@K: does retrieval find at least one relevant chunk? The minimum bar.
- MRR: where is the first relevant chunk ranked? Rewards fast finding.
- MAP: comprehensive metric combining ranking order and coverage for binary relevance.
- NDCG@K: like MAP but supports graded relevance. Best for re-ranker evaluation.
- Coverage: fraction of relevant chunks actually in the index. An indexing quality metric, not retrieval quality.
- Standard reporting: Hit Rate@5/10, Recall@5/10, MRR as primary metrics. Add NDCG when graded relevance data is available.
- Interpret metric combinations to diagnose whether problems are in ranking (MRR/NDCG), coverage (Recall), noise (Precision), or indexing (Coverage/Hit Rate).

---

## What's Next

Lesson 6.3 covers generation metrics in depth — Exact Match, F1 Score, BLEU, ROUGE, METEOR, BERTScore, and Semantic Similarity — with the specific strengths and failure modes of each for RAG evaluation.