# Lesson 7.2 — The "Accuracy Drops at 5K Documents" Problem: Root Cause Analysis and Fixes

---

## The Problem Statement

This is one of the most common and most misdiagnosed problems in production RAG systems. You build a system, test it on a few hundred documents, and get excellent retrieval accuracy. You index 5,000 documents. Accuracy drops to 70%. You index 50,000 documents. Accuracy drops to 55%.

The common misdiagnosis is to treat this as a single problem with a single fix. It is not. It is a class of several distinct problems that all manifest the same way — lower retrieval accuracy at larger scale — but have entirely different root causes and different fixes.

This lesson goes through each root cause systematically, how to diagnose which one is affecting your system, and how to fix it.

---

## Root Cause 1 — HNSW Graph Degradation

### What Happens

HNSW builds a navigable graph by connecting each vector to its approximate nearest neighbors at index time. When you start with 100 vectors, the graph is built over a small, coherent set. Every node is well-connected to its true neighbors.

As you add vectors incrementally, each new vector is inserted into the existing graph with connections to its nearest neighbors in the current graph state. But the graph was not designed for the new vectors — the upper layers (which enable long-range jumps) have connections that reflect the old distribution.

After 50,000 incremental insertions into a graph built for 5,000 vectors, the graph structure is suboptimal for the current distribution. Long-range connections that should exist do not. Short-range connections that should be rewired have not been. The greedy traversal gets stuck in local minima more often.

### Evidence

```python
async def measure_hnsw_degradation(
    vector_db,
    embedding_model,
    test_query_embeddings: list,
    ground_truth_neighbors: list[list[str]],  # True top-10 for each test query
    k: int = 10
) -> dict:
    """
    Measure current ANN recall vs. expected recall.
    If ANN recall has dropped since the last measurement, HNSW has degraded.
    """
    
    import numpy as np
    
    recalls = []
    
    for query_emb, true_neighbors in zip(test_query_embeddings, ground_truth_neighbors):
        ann_results = await vector_db.search(
            query_vector=query_emb,
            limit=k
        )
        ann_ids = set(r.id for r in ann_results)
        true_ids = set(true_neighbors[:k])
        
        recall = len(ann_ids & true_ids) / len(true_ids)
        recalls.append(recall)
    
    current_recall = float(np.mean(recalls))
    
    # Compare to baseline (stored from when index was last rebuilt)
    baseline_recall = await get_stored_baseline_recall()
    
    degradation = baseline_recall - current_recall if baseline_recall else None
    
    return {
        "current_recall_at_k": current_recall,
        "baseline_recall_at_k": baseline_recall,
        "degradation": degradation,
        "is_degraded": degradation is not None and degradation > 0.05,
        "severity": (
            "high" if degradation and degradation > 0.15
            else "medium" if degradation and degradation > 0.05
            else "low"
        )
    }
```

To establish the ground truth neighbors, compute exact nearest neighbors (brute force) on a sample of test queries when the index is small and well-performing. Store these as your baseline. Compare ANN results against them as the index grows.

### Fix

**Short-term (no downtime):** Increase the `ef` parameter at query time. This makes HNSW explore more of the graph per query, compensating for degraded structure at the cost of higher latency.

```python
# Increase ef to compensate for graph degradation
# Normal: ef=128
# After degradation: ef=256 or ef=512

results = await vector_db.search(
    query_vector=query_embedding,
    limit=k,
    search_params=SearchParams(hnsw_ef=256)  # Qdrant syntax
)
```

**Medium-term (planned downtime or blue-green):** Rebuild the HNSW index from scratch. A freshly built index over all current vectors will have far better graph quality than one built incrementally.

```python
# Qdrant: trigger index rebuild
# This can be done on a blue collection without taking down the live one

# 1. Create new collection
await client.create_collection(
    collection_name="documents_v2",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=32, ef_construct=400)  # Higher quality rebuild
)

# 2. Re-upload all vectors to new collection
await bulk_upsert_all_vectors(
    source_collection="documents",
    target_collection="documents_v2"
)

# 3. Verify recall on documents_v2
# 4. Switch traffic to documents_v2
# 5. Delete documents (old collection)
```

**Prevention:** Rebuild the HNSW index on a schedule proportional to your insertion rate. For a corpus growing by 10,000 vectors/week, rebuild monthly. For rapid growth (100,000 vectors/week), rebuild weekly.

---

## Root Cause 2 — BM25 IDF Score Drift

### What Happens

BM25 weights terms by their Inverse Document Frequency (IDF). A term that appears in 10% of documents gets a lower IDF weight than a term appearing in 0.1% of documents. Rare terms are more informative.

When you build a BM25 index with 500 documents, certain domain-specific terms are rare and get high IDF weights — correctly so. As you grow to 5,000 documents covering the same domain, those same terms may appear in 30% of documents. Their IDF weight drops significantly.

The net effect: terms that were highly discriminative at small scale become less discriminative at large scale because they are no longer rare. BM25 scores for queries using these terms become less useful for ranking.

Additionally, document length statistics (used for length normalization) shift as the corpus grows. The average document length parameter `avgdl` in BM25 changes as new documents are added, affecting scores for all queries.

### Evidence

```python
def detect_idf_drift(
    current_vocab_stats: dict,  # {term: document_frequency} current
    baseline_vocab_stats: dict  # {term: document_frequency} at baseline
) -> dict:
    """
    Identify terms whose IDF has drifted significantly between baseline and current.
    """
    import math
    
    n_current = sum(current_vocab_stats.values())
    n_baseline = sum(baseline_vocab_stats.values())
    
    drifted_terms = []
    
    for term, current_df in current_vocab_stats.items():
        if term not in baseline_vocab_stats:
            continue  # New term, not drift
        
        baseline_df = baseline_vocab_stats[term]
        
        # IDF at baseline and current
        idf_baseline = math.log((n_baseline - baseline_df + 0.5) / (baseline_df + 0.5) + 1)
        idf_current = math.log((n_current - current_df + 0.5) / (current_df + 0.5) + 1)
        
        relative_change = abs(idf_current - idf_baseline) / max(idf_baseline, 0.01)
        
        if relative_change > 0.30:  # 30% change in IDF
            drifted_terms.append({
                "term": term,
                "baseline_df": baseline_df,
                "current_df": current_df,
                "idf_baseline": round(idf_baseline, 3),
                "idf_current": round(idf_current, 3),
                "idf_change_pct": round(relative_change * 100, 1)
            })
    
    # Sort by magnitude of change
    drifted_terms.sort(key=lambda x: x["idf_change_pct"], reverse=True)
    
    return {
        "n_drifted_terms": len(drifted_terms),
        "top_drifted_terms": drifted_terms[:20],
        "significant_drift": len(drifted_terms) > 50
    }
```

### Fix

**Rebuild BM25 index periodically.** Unlike HNSW, BM25 index rebuilds are fast (no GPU, no embedding computation required — just tokenization and term counting). Rebuild weekly or monthly, or whenever corpus size doubles.

**For production systems using Elasticsearch:** Elasticsearch recomputes IDF dynamically across shards using a per-shard calculation. With large enough shards, IDF is stable. With many small shards, IDF can be inconsistent across the corpus. Use `search_type=dfs_query_then_fetch` to compute global IDF across all shards for more consistent scoring at the cost of extra network roundtrips.

---

## Root Cause 3 — Embedding Space Crowding

### What Happens

An embedding model maps all text to a fixed-dimensional space. With 100 documents, the populated region of that space is sparse. Each document occupies a relatively unique location. Similarity scores are well-distributed.

With 50,000 documents from the same domain, the embedding space becomes crowded. Many documents about similar topics cluster in the same region. The cosine similarity between a query and the 100th most relevant document may be barely lower than the similarity to the 5th most relevant document.

The practical effect: the signal-to-noise ratio of similarity scores decreases. Documents that are somewhat relevant get scores very close to highly relevant documents. The ranking becomes noisier.

This is worsened when documents are repetitive. If 1,000 of your 5,000 documents are policy documents that use similar formal language, they will cluster tightly in embedding space. Any policy query retrieves a mix of relevant and irrelevant policy documents with barely distinguishable scores.

### Evidence

```python
def measure_embedding_space_crowding(
    embeddings: np.ndarray,  # Sample of embeddings from the index
    k: int = 10
) -> dict:
    """
    Measure the density and distinguishability of the embedding space.
    High crowding = worse retrieval accuracy at scale.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    knn.fit(embeddings)
    
    distances, _ = knn.kneighbors(embeddings)
    
    # Exclude self (distance 0)
    nn_distances = distances[:, 1:]  # k nearest neighbor distances
    
    # Average distance to nearest neighbor
    avg_nn_distance = float(nn_distances[:, 0].mean())
    
    # Score spread: difference between closest and k-th closest neighbor
    score_spread = float((nn_distances[:, -1] - nn_distances[:, 0]).mean())
    
    # Crowding ratio: if score spread is small, space is crowded
    crowding_ratio = score_spread / avg_nn_distance if avg_nn_distance > 0 else 0
    
    return {
        "avg_nearest_neighbor_distance": avg_nn_distance,
        "avg_score_spread_top_k": score_spread,
        "crowding_ratio": crowding_ratio,
        "is_crowded": crowding_ratio < 0.3,  # Low spread = high crowding
        "interpretation": (
            "Dense, hard to distinguish relevant from irrelevant" if crowding_ratio < 0.2
            else "Moderate crowding — re-ranking becomes more important"
            if crowding_ratio < 0.4
            else "Good separation in embedding space"
        )
    }
```

### Fixes

**More aggressive re-ranking.** When embedding space is crowded, the initial retrieval ranking is noisy. Re-ranking with a cross-encoder (which can distinguish subtle differences that embedding similarity cannot) compensates for crowding. Increase the number of candidates passed to the re-ranker (K=100 instead of K=50).

**Metadata filtering to narrow the search space.** If you know that a query is about a specific document type, department, or time period, filtering before vector search reduces the effective search space. With a smaller, more targeted search space, crowding effects are reduced.

**Retrieval partitioning.** Instead of one large index, maintain multiple smaller indices partitioned by topic, department, or document type. Queries are routed to the appropriate partition first. Each partition has less crowding within its domain.

```python
async def partitioned_retrieval(
    query: str,
    query_metadata: dict,  # What we know about the query context
    partition_router,       # Routes queries to appropriate partitions
    partition_retrievers: dict  # {partition_id: retriever}
) -> list[dict]:
    """
    Route query to the most relevant partition(s) before retrieval.
    Reduces crowding by narrowing the search space.
    """
    
    # Determine which partitions to search
    relevant_partitions = await partition_router.route(query, query_metadata)
    
    # Retrieve from relevant partitions in parallel
    partition_tasks = [
        partition_retrievers[p_id].retrieve(query, k=20)
        for p_id in relevant_partitions
    ]
    
    partition_results = await asyncio.gather(*partition_tasks)
    
    # Merge results from all partitions
    all_results = [
        result
        for partition_result in partition_results
        for result in partition_result
    ]
    
    # Re-rank the merged results
    return rerank(query, all_results)[:10]
```

**Fine-tune the embedding model on your domain.** A domain-fine-tuned model learns to push similar-but-different documents further apart in the embedding space. This directly reduces crowding for your specific domain vocabulary.

---

## Root Cause 4 — Metadata Filter Selectivity Problem

### What Happens

At small scale, metadata filters narrow the search to a small, well-defined subset. The HNSW graph is large relative to the filtered subset, but the result count is manageable.

At large scale, aggressive metadata filters may narrow the search to an extremely small subset (0.1% of 500K vectors = 500 vectors). The HNSW graph — built for 500K vectors — is poorly suited for searching a 500-vector subset. The graph traversal visits many nodes that fail the filter, finding only a few that pass.

Some databases handle this by falling back to brute-force search when the filtered subset is small. This is slower but accurate. Others try to use the graph anyway, producing degraded recall with no warning.

### Evidence

```python
async def audit_filter_impact_on_recall(
    vector_db,
    test_queries_with_filters: list[dict],
    ground_truth: list[dict]
) -> dict:
    """
    Compare retrieval quality with and without metadata filters.
    Large gap indicates filter selectivity is hurting ANN quality.
    """
    
    import numpy as np
    
    filtered_recalls = []
    unfiltered_recalls = []
    
    for query_item, gt in zip(test_queries_with_filters, ground_truth):
        query_emb = query_item["embedding"]
        metadata_filter = query_item.get("metadata_filter")
        true_ids = set(gt["relevant_chunk_ids"])
        
        # Search without filter
        unfiltered = await vector_db.search(
            query_vector=query_emb, limit=10
        )
        unfiltered_recall = len(set(r.id for r in unfiltered) & true_ids) / len(true_ids)
        unfiltered_recalls.append(unfiltered_recall)
        
        # Search with filter
        if metadata_filter:
            filtered = await vector_db.search(
                query_vector=query_emb,
                filter=metadata_filter,
                limit=10
            )
            filtered_recall = len(set(r.id for r in filtered) & true_ids) / len(true_ids)
        else:
            filtered_recall = unfiltered_recall
        
        filtered_recalls.append(filtered_recall)
    
    avg_filter_impact = float(np.mean(unfiltered_recalls)) - float(np.mean(filtered_recalls))
    
    return {
        "avg_recall_without_filter": float(np.mean(unfiltered_recalls)),
        "avg_recall_with_filter": float(np.mean(filtered_recalls)),
        "filter_impact": avg_filter_impact,
        "filter_hurts_recall": avg_filter_impact > 0.05,
        "recommendation": (
            "Filters significantly degrade recall — consider payload indexes or partition strategy"
            if avg_filter_impact > 0.10
            else "Moderate filter impact — ensure payload indexes are created"
            if avg_filter_impact > 0.05
            else "Filter impact acceptable"
        )
    }
```

### Fixes

**Create payload indexes for all filter fields.** Without explicit payload indexes, filters require scanning all vectors. With indexes, the database can efficiently identify qualifying vectors before ANN search.

```python
# Create indexes for all fields you filter on
for field_name, field_type in [
    ("document_status", "keyword"),
    ("department", "keyword"),
    ("effective_date", "datetime"),
    ("document_type", "keyword")
]:
    await client.create_payload_index(
        collection_name="documents",
        field_name=field_name,
        field_schema=field_type
    )
```

**Switch to pre-filtering with a fallback.** Configure the database to use pre-filtering (only search qualifying vectors) with automatic fallback to brute force when the filtered subset is too small for good ANN results.

**Increase ef proportionally for filtered queries.** When a query has a restrictive filter, increase ef to compensate for the degraded graph performance on the small subset.

```python
async def adaptive_filtered_search(
    vector_db,
    query_embedding: list[float],
    metadata_filter: dict,
    k: int = 10,
    base_ef: int = 128
) -> list:
    """
    Adjust ef based on expected filter selectivity.
    """
    
    # Estimate the selectivity of this filter
    estimated_match_count = await vector_db.count(
        collection="documents",
        count_filter=metadata_filter
    )
    
    total_vectors = await vector_db.count(collection="documents")
    selectivity = estimated_match_count / total_vectors if total_vectors > 0 else 1.0
    
    # Increase ef for highly selective filters
    if selectivity < 0.01:      # Less than 1% matches
        ef = base_ef * 4
    elif selectivity < 0.05:    # Less than 5% matches
        ef = base_ef * 2
    else:
        ef = base_ef
    
    return await vector_db.search(
        collection="documents",
        query_vector=query_embedding,
        filter=metadata_filter,
        limit=k,
        search_params={"hnsw_ef": ef}
    )
```

---

## Root Cause 5 — Retrieval Precision Dilution

### What Happens

At small scale, the top-K retrieved chunks are nearly all relevant — there are not many chunks to compete with. At large scale, with 50,000 chunks in your index, the top-K must be drawn from a vastly larger pool of potential matches. Many chunks that are marginally related will have higher similarity scores than the genuinely relevant chunks from less-common topics.

This is not the same as embedding space crowding (which is a structural issue with the embedding space). Precision dilution is a simple statistical effect: with more chunks in the index, there are more chances for false positives to appear in the top-K.

### Evidence

Track Precision@K (not just Recall@K) as corpus size grows. Precision should remain stable if the retrieval system is scaling well. A declining Precision@K with growing corpus size indicates precision dilution.

```python
def track_precision_by_corpus_size(
    recall_history: list[dict]  # [{corpus_size, recall_at_k, precision_at_k, date}]
) -> dict:
    """
    Identify whether precision is declining as corpus grows.
    """
    import numpy as np
    
    if len(recall_history) < 3:
        return {"status": "insufficient_data"}
    
    sorted_by_size = sorted(recall_history, key=lambda x: x["corpus_size"])
    
    corpus_sizes = [r["corpus_size"] for r in sorted_by_size]
    precisions = [r["precision_at_k"] for r in sorted_by_size]
    
    # Check if precision is decreasing as corpus grows
    # Simple linear regression
    x = np.array(corpus_sizes)
    y = np.array(precisions)
    
    correlation = float(np.corrcoef(x, y)[0, 1])
    
    return {
        "corpus_sizes": corpus_sizes,
        "precisions": precisions,
        "trend_correlation": correlation,
        "precision_declining": correlation < -0.5,
        "interpretation": (
            "Precision declining with scale — add more aggressive re-ranking"
            if correlation < -0.5
            else "Precision stable with scale — retrieval scaling well"
        )
    }
```

### Fixes

**Increase re-ranker K (candidates to re-rank).** With more documents in the index, the initial retrieval needs to cast a wider net to find the truly relevant chunks. Increase K from 50 to 100 or 200 as corpus grows, then let re-ranking bring the quality K to the top.

**Two-stage retrieval with category pre-filtering.** Route queries to a topic category first (using a lightweight classifier or metadata), then search within that category. This reduces the effective corpus size for any given query.

**Adaptive K based on corpus size.** Do not use a fixed K. Scale K with corpus size.

```python
def compute_adaptive_k(
    corpus_size: int,
    base_k: int = 20,
    base_corpus_size: int = 1000
) -> int:
    """
    Scale the number of retrieved candidates with corpus size.
    At 1K documents: K=20
    At 10K documents: K=45 (sqrt scaling)
    At 100K documents: K=100
    """
    import math
    
    scale_factor = math.sqrt(corpus_size / base_corpus_size)
    adaptive_k = int(base_k * scale_factor)
    
    # Cap at a reasonable maximum (latency constraint)
    max_k = 200
    return min(adaptive_k, max_k)
```

---

## Putting It Together: The Scale Degradation Diagnostic

When you observe that accuracy has dropped as corpus size grew, run this diagnostic to identify which root cause(s) are responsible:

```python
async def diagnose_scale_degradation(
    vector_db,
    embedding_model,
    test_data: dict,
    baseline_metrics: dict  # Metrics when system was performing well
) -> dict:
    """
    Comprehensive scale degradation diagnostic.
    """
    
    findings = {}
    
    # 1. Check HNSW degradation
    hnsw_result = await measure_hnsw_degradation(
        vector_db=vector_db,
        embedding_model=embedding_model,
        test_query_embeddings=test_data["query_embeddings"],
        ground_truth_neighbors=test_data["ground_truth"]
    )
    findings["hnsw_degradation"] = hnsw_result
    
    # 2. Check BM25 IDF drift (if accessible)
    if test_data.get("vocab_stats"):
        findings["idf_drift"] = detect_idf_drift(
            current_vocab_stats=test_data["vocab_stats"]["current"],
            baseline_vocab_stats=test_data["vocab_stats"]["baseline"]
        )
    
    # 3. Check embedding space crowding
    sample_embeddings = test_data.get("sample_embeddings")
    if sample_embeddings is not None:
        findings["space_crowding"] = measure_embedding_space_crowding(sample_embeddings)
    
    # 4. Check filter selectivity impact
    if test_data.get("filtered_queries"):
        findings["filter_impact"] = await audit_filter_impact_on_recall(
            vector_db=vector_db,
            test_queries_with_filters=test_data["filtered_queries"],
            ground_truth=test_data["filtered_ground_truth"]
        )
    
    # 5. Check precision trend
    if test_data.get("precision_history"):
        findings["precision_trend"] = track_precision_by_corpus_size(
            test_data["precision_history"]
        )
    
    # Synthesize recommendations
    recommendations = []
    
    if findings.get("hnsw_degradation", {}).get("is_degraded"):
        severity = findings["hnsw_degradation"]["severity"]
        recommendations.append({
            "priority": "P0" if severity == "high" else "P1",
            "action": "Rebuild HNSW index",
            "detail": f"ANN recall has degraded by {findings['hnsw_degradation'].get('degradation', 0):.1%}"
        })
    
    if findings.get("idf_drift", {}).get("significant_drift"):
        recommendations.append({
            "priority": "P1",
            "action": "Rebuild BM25 index with current IDF statistics",
            "detail": f"{findings['idf_drift']['n_drifted_terms']} terms have drifted significantly"
        })
    
    if findings.get("space_crowding", {}).get("is_crowded"):
        recommendations.append({
            "priority": "P1",
            "action": "Increase re-ranker K and/or add partitioning",
            "detail": "Embedding space is crowded — retrieval ranking is noisy"
        })
    
    if findings.get("filter_impact", {}).get("filter_hurts_recall"):
        recommendations.append({
            "priority": "P1",
            "action": "Create payload indexes for all filter fields",
            "detail": f"Filters are degrading recall by {findings['filter_impact']['filter_impact']:.1%}"
        })
    
    if findings.get("precision_trend", {}).get("precision_declining"):
        recommendations.append({
            "priority": "P2",
            "action": "Implement adaptive K scaling with corpus size",
            "detail": "Precision declining as corpus grows — more candidates needed pre-re-ranking"
        })
    
    return {
        "findings": findings,
        "recommendations": sorted(recommendations, key=lambda x: x["priority"]),
        "primary_cause": recommendations[0]["action"] if recommendations else "No clear degradation detected"
    }
```

---

## Prevention: Proactive Scaling Checklist

Before your corpus reaches a new order of magnitude, run this checklist:

**At 10K vectors:**
- [ ] Create payload indexes for all metadata filter fields.
- [ ] Verify HNSW parameters (M=32, ef_construct=200 minimum for quality).
- [ ] Establish baseline recall@K measurement for future comparison.

**At 100K vectors:**
- [ ] Schedule monthly HNSW rebuilds.
- [ ] Rebuild BM25 index with current statistics.
- [ ] Increase re-ranker K from 50 to 100.
- [ ] Run crowding audit on sample embeddings.

**At 1M vectors:**
- [ ] Evaluate index sharding by topic or document type.
- [ ] Consider product quantization (IVFPQ) for memory management.
- [ ] Switch from RAM-based HNSW to disk-backed quantized index.
- [ ] Measure ef and adaptive-K requirements at new scale.

---

## Summary

- The "accuracy drops at scale" problem has five distinct root causes, each requiring a different fix.
- HNSW graph degradation: incremental insertions degrade the graph structure. Fix: increase ef short-term, rebuild index long-term.
- BM25 IDF drift: term frequencies shift as corpus grows, changing relative term weights. Fix: rebuild BM25 index periodically.
- Embedding space crowding: more documents in the same domain cluster tightly, reducing retrieval precision. Fix: more aggressive re-ranking, partitioning, embedding model fine-tuning.
- Metadata filter selectivity: highly selective filters degrade ANN performance. Fix: payload indexes, adaptive ef, pre-filtering strategies.
- Retrieval precision dilution: more documents means more false positive competitors in top-K. Fix: adaptive K scaling, two-stage retrieval.
- Run the full diagnostic before choosing a fix. Multiple causes often co-exist, but address them in priority order.
- Establish baseline recall@K measurements when the system is performing well. These are the reference for detecting future degradation.

---

## What's Next

Lesson 7.3 covers data conflicts and knowledge inconsistency resolution — how to handle cases where your corpus contains contradicting information, and what the system should do when retrieved chunks disagree with each other.