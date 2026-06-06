# Lesson 6.7 — Data Drift and Distribution Shift: Detection and Response

---

## What Drift Means in RAG Systems

A RAG system is not static. Three things change over time, each of which can degrade quality without any change to the code:

**1. Query distribution shift:** The kinds of questions users ask change. New topics emerge, new user segments join, existing users develop new use cases. A system optimized for Q1 queries may perform poorly on Q4 queries if the distribution has shifted.

**2. Corpus drift:** The documents in your knowledge base change. Policies are updated, products are discontinued, regulations change. The index that was accurate in January may have stale, contradicted, or missing content by June.

**3. Embedding drift:** If you update your embedding model, existing vectors become incompatible with new query embeddings. Even if you do not update the model, the embedding space can effectively "drift" as new documents with different vocabulary and style are added to the index.

None of these produce errors. The system continues running, continues returning answers, and continues looking healthy on infrastructure dashboards — while quality silently degrades.

This lesson covers how to detect each type of drift and what to do about it.

---

## Type 1 — Query Distribution Shift

### What It Is

The statistical distribution of incoming queries changes over time. Concretely: topics that were rare become common, topics that were common become rare, entirely new topics appear.

**Causes:**
- New user segments (a different department starts using the system).
- Seasonal patterns (questions about benefits spike in open enrollment, tax questions spike in April).
- External events (a new regulation passes, queries about it spike overnight).
- Product changes (a new feature ships, support queries about it begin).
- System discovery (users find a capability they did not know existed).

### Detection

**Embedding-based distribution comparison:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

class QueryDriftDetector:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(embedding_model_name)
    
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, normalize_embeddings=True)
    
    def detect_distribution_shift(
        self,
        reference_queries: list[str],   # Historical baseline
        current_queries: list[str],     # Recent window
        n_components: int = 50          # PCA dimensions for comparison
    ) -> dict:
        """
        Detect whether the current query distribution has shifted
        relative to the reference distribution.
        """
        from sklearn.decomposition import PCA
        
        if len(reference_queries) < 50 or len(current_queries) < 50:
            return {"status": "insufficient_data"}
        
        # Embed both sets
        ref_embeddings = self.embed_queries(reference_queries)
        cur_embeddings = self.embed_queries(current_queries)
        
        # Reduce to comparable lower-dimensional space
        all_embeddings = np.vstack([ref_embeddings, cur_embeddings])
        pca = PCA(n_components=n_components)
        all_projected = pca.fit_transform(all_embeddings)
        
        ref_projected = all_projected[:len(reference_queries)]
        cur_projected = all_projected[len(reference_queries):]
        
        # Compare distributions along each principal component
        ks_stats = []
        p_values = []
        
        for dim in range(n_components):
            stat, p = ks_2samp(ref_projected[:, dim], cur_projected[:, dim])
            ks_stats.append(stat)
            p_values.append(p)
        
        # Overall drift score: mean KS statistic across dimensions
        mean_ks = float(np.mean(ks_stats))
        n_significant_dims = sum(1 for p in p_values if p < 0.05)
        
        # Centroid shift
        ref_centroid = ref_projected.mean(axis=0)
        cur_centroid = cur_projected.mean(axis=0)
        centroid_distance = float(np.linalg.norm(cur_centroid - ref_centroid))
        
        drift_detected = mean_ks > 0.10 or n_significant_dims > n_components * 0.3
        
        return {
            "drift_detected": drift_detected,
            "mean_ks_statistic": mean_ks,
            "n_significant_dimensions": n_significant_dims,
            "centroid_distance": centroid_distance,
            "severity": (
                "high" if mean_ks > 0.20
                else "medium" if mean_ks > 0.10
                else "low"
            )
        }
    
    def find_new_topics(
        self,
        reference_queries: list[str],
        current_queries: list[str],
        n_clusters: int = 20
    ) -> list[dict]:
        """
        Identify query topics that are new or much more frequent in current window.
        """
        from sklearn.cluster import KMeans
        
        ref_embs = self.embed_queries(reference_queries)
        cur_embs = self.embed_queries(current_queries)
        
        all_embs = np.vstack([ref_embs, cur_embs])
        
        # Cluster all queries
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_embs)
        
        ref_labels = labels[:len(reference_queries)]
        cur_labels = labels[len(reference_queries):]
        
        # Compare cluster frequencies
        new_topics = []
        for cluster_id in range(n_clusters):
            ref_freq = (ref_labels == cluster_id).mean()
            cur_freq = (cur_labels == cluster_id).mean()
            
            if cur_freq > ref_freq * 2 and cur_freq > 0.02:
                # Cluster is significantly more common in current window
                # Find representative queries from this cluster
                cur_cluster_queries = [
                    current_queries[i] for i, l in enumerate(cur_labels) 
                    if l == cluster_id
                ][:5]
                
                new_topics.append({
                    "cluster_id": cluster_id,
                    "reference_frequency": ref_freq,
                    "current_frequency": cur_freq,
                    "growth_factor": cur_freq / max(ref_freq, 0.001),
                    "example_queries": cur_cluster_queries
                })
        
        return sorted(new_topics, key=lambda x: x["growth_factor"], reverse=True)
```

**Topic-level frequency monitoring:**

```python
async def monitor_topic_frequencies(
    query_logs: list[dict],
    llm_client,
    time_windows: list[tuple]  # [(start, end), ...]
) -> dict:
    """
    Track how topic frequencies change across time windows.
    """
    
    # Classify each query into a topic
    async def classify_topic(query: str) -> str:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Classify this query into a topic category (max 3 words): {query}\nRespond with only the category."
            }],
            max_tokens=10,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    
    # Sample classification (expensive to do for all queries)
    sample_queries = query_logs[-1000:]  # Recent 1000 queries
    topics = {}
    
    for qlog in sample_queries:
        topic = await classify_topic(qlog["query"])
        topics[topic] = topics.get(topic, 0) + 1
    
    # Sort by frequency
    sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "top_topics": sorted_topics[:20],
        "unique_topics": len(topics),
        "total_queries_sampled": len(sample_queries)
    }
```

### Response to Query Distribution Shift

**Immediate action:** Identify which new topics are not well-covered by the current corpus. These are likely the source of increased IDK responses or hallucination.

**Short-term:** Ingest documents covering the new topics. Update the indexing pipeline if new document types are involved.

**Medium-term:** Add representative samples of new topic queries to the evaluation dataset. Re-run offline evaluation to measure coverage of new topics.

**Long-term:** If the query distribution has fundamentally changed, consider re-fine-tuning the embedding model on the new query distribution.

---

## Type 2 — Corpus Drift

### What It Is

The content of indexed documents becomes stale relative to ground truth. Policies change but old chunks persist. New documents are added but not indexed promptly. Documents are deleted but their chunks remain.

We covered the mechanics of data freshness in Lesson 2.6. Here we focus on detecting corpus drift at a systemic level — not whether a specific document was updated, but whether the overall quality of the index is degrading.

### Detection

**Staleness rate monitoring:**

```python
async def compute_corpus_staleness(
    vector_db,
    source_document_store,
    registry
) -> dict:
    """
    Measure what fraction of indexed chunks are from stale documents.
    """
    
    # Get all indexed documents and their index timestamps
    indexed_docs = await registry.get_all_active()
    
    stale_count = 0
    stale_docs = []
    
    for doc in indexed_docs:
        # Get the actual last-modified date of the source document
        source_modified = await source_document_store.get_modified_date(
            doc["source_path"]
        )
        
        if source_modified is None:
            continue  # Document may have been deleted
        
        indexed_at = doc["indexed_at"]
        days_stale = (source_modified - indexed_at).days
        
        if days_stale > 1:  # Modified after last index
            stale_count += 1
            stale_docs.append({
                "doc_id": doc["doc_id"],
                "source_path": doc["source_path"],
                "days_stale": days_stale
            })
    
    total = len(indexed_docs)
    staleness_rate = stale_count / total if total > 0 else 0
    
    return {
        "staleness_rate": staleness_rate,
        "stale_doc_count": stale_count,
        "total_indexed_docs": total,
        "most_stale_docs": sorted(stale_docs, key=lambda x: x["days_stale"], reverse=True)[:10],
        "alert": staleness_rate > 0.05  # Alert if >5% of docs are stale
    }
```

**Content contradiction detection:**

When a policy document is updated, both the old and new versions may exist in the index. The system then retrieves conflicting information for the same query. This is detectable by looking at retrieved chunks for the same query and checking whether they contradict each other.

```python
async def detect_content_contradictions(
    queries: list[str],
    retriever,
    llm_client,
    sample_rate: float = 0.01  # Check 1% of queries
) -> dict:
    """
    Sample queries and check whether retrieved chunks contradict each other.
    """
    import random
    
    sampled = random.sample(queries, max(1, int(len(queries) * sample_rate)))
    
    contradictions_found = []
    
    for query in sampled:
        results = await retriever.retrieve(query, k=5)
        
        if len(results) < 2:
            continue
        
        # Check top chunks for contradictions
        context_parts = [r["text"][:500] for r in results[:3]]
        
        prompt = f"""Do any of these retrieved document sections contradict each other 
on the topic of this query?

Query: {query}

Sections:
{chr(10).join(f"[{i+1}] {text}" for i, text in enumerate(context_parts))}

Respond with JSON:
{{
    "contradiction_found": true/false,
    "description": "what contradicts what (if found)"
}}"""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        if result.get("contradiction_found"):
            contradictions_found.append({
                "query": query,
                "description": result.get("description"),
                "chunk_ids": [r["chunk_id"] for r in results[:3]]
            })
    
    return {
        "contradiction_rate": len(contradictions_found) / len(sampled),
        "contradictions": contradictions_found,
        "queries_sampled": len(sampled),
        "alert": len(contradictions_found) / len(sampled) > 0.05
    }
```

### Response to Corpus Drift

**Immediate:** Prioritize re-indexing of documents flagged as stale by the staleness monitor.

**Conflict resolution:** When contradictions are detected between old and new versions of a document, use the `document_status` field to mark old versions as `superseded` so they are excluded from retrieval by default.

**Prevention:** Tighten the freshness SLA (maximum acceptable staleness) for high-importance document categories. Implement webhook-based change detection for critical documents.

---

## Type 3 — Embedding Drift

### What It Is

As the corpus grows and changes, the effective distribution of vectors in the index shifts. New document types may cluster in regions of embedding space not well-covered by the HNSW graph built on older data. Retrieval quality degrades because the graph structure no longer matches the current distribution.

Also occurs explicitly when you update your embedding model — all old vectors are now incompatible with new query embeddings.

### Detection

**ANN recall degradation monitoring:**

```python
async def monitor_ann_quality(
    vector_db,
    embedding_model,
    test_query_embeddings: list[np.ndarray],  # Pre-computed test query vectors
    ground_truth_chunk_ids: list[list[str]],   # True neighbors for each test query
    k: int = 10
) -> dict:
    """
    Periodically measure ANN recall to detect index quality degradation.
    Run weekly or after large batch indexing.
    """
    
    recalls = []
    
    for query_emb, true_neighbors in zip(test_query_embeddings, ground_truth_chunk_ids):
        results = await vector_db.search(
            collection="documents",
            query_vector=query_emb.tolist(),
            limit=k
        )
        
        retrieved_ids = set(r.id for r in results)
        true_ids = set(true_neighbors[:k])
        
        recall = len(retrieved_ids & true_ids) / len(true_ids) if true_ids else 0
        recalls.append(recall)
    
    mean_recall = float(np.mean(recalls))
    
    return {
        "mean_recall_at_k": mean_recall,
        "k": k,
        "n_test_queries": len(test_query_embeddings),
        "alert": mean_recall < 0.85,  # Alert if recall drops below 85%
        "recommendation": (
            "Consider rebuilding HNSW index" if mean_recall < 0.80
            else "Monitor closely" if mean_recall < 0.85
            else "Healthy"
        )
    }
```

**Embedding space coverage monitoring:**

As the corpus grows, check whether new document embeddings fall outside the density regions of the existing index — an indicator that the graph structure needs updating.

```python
def monitor_embedding_coverage(
    existing_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    coverage_threshold: float = 0.95
) -> dict:
    """
    Check whether new embeddings are well-covered by the existing index structure.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit KNN on existing embeddings
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(existing_embeddings)
    
    # Find nearest neighbors for new embeddings in existing space
    distances, _ = knn.kneighbors(new_embeddings)
    
    # Average distance to nearest existing neighbor
    avg_nearest_distance = float(distances[:, 0].mean())
    
    # Fraction of new embeddings with close existing neighbors
    close_threshold = 0.15  # Cosine distance < 0.15 means very similar
    pct_well_covered = float((distances[:, 0] < close_threshold).mean())
    
    return {
        "avg_nearest_distance": avg_nearest_distance,
        "pct_well_covered": pct_well_covered,
        "needs_index_rebuild": pct_well_covered < coverage_threshold,
        "severity": (
            "high" if pct_well_covered < 0.85
            else "medium" if pct_well_covered < 0.95
            else "low"
        )
    }
```

### Response to Embedding Drift

**HNSW index rebuild:** When ANN recall drops significantly, rebuild the HNSW index from scratch. This is expensive (requires re-indexing all vectors) but restores graph quality. Use blue-green indexing (Lesson 2.6) to avoid downtime.

**Incremental HNSW optimization:** Some vector databases support periodic index optimization without full rebuild. Qdrant's optimizer, for example, can improve graph quality for recently added vectors without a full rebuild.

**Embedding model migration:** When migrating to a new embedding model, re-embed the entire corpus and build a new index. Use blue-green switching to migrate without downtime. Track embedding model version in metadata.

---

## The Drift Monitoring Dashboard

A production RAG system needs a unified view of all drift indicators. Here is the monitoring schema:

```python
class DriftMonitor:
    def __init__(
        self,
        query_drift_detector: QueryDriftDetector,
        registry,
        vector_db,
        embedding_model,
        alert_service
    ):
        self.query_drift = query_drift_detector
        self.registry = registry
        self.vector_db = vector_db
        self.embedder = embedding_model
        self.alerts = alert_service
    
    async def run_weekly_drift_report(self) -> dict:
        """
        Comprehensive weekly drift assessment.
        """
        from datetime import datetime, timedelta
        
        now = datetime.utcnow()
        reference_window_start = now - timedelta(days=60)
        reference_window_end = now - timedelta(days=30)
        current_window_start = now - timedelta(days=7)
        
        # 1. Query distribution drift
        reference_queries = await fetch_query_logs(reference_window_start, reference_window_end)
        current_queries = await fetch_query_logs(current_window_start, now)
        
        query_drift = self.query_drift.detect_distribution_shift(
            reference_queries=[q["query"] for q in reference_queries],
            current_queries=[q["query"] for q in current_queries]
        )
        
        new_topics = self.query_drift.find_new_topics(
            reference_queries=[q["query"] for q in reference_queries],
            current_queries=[q["query"] for q in current_queries]
        )
        
        # 2. Corpus staleness
        corpus_staleness = await compute_corpus_staleness(
            self.vector_db, None, self.registry
        )
        
        # 3. ANN quality (use stored test queries)
        test_data = await load_test_query_embeddings()
        ann_quality = await monitor_ann_quality(
            self.vector_db,
            self.embedder,
            test_data["embeddings"],
            test_data["ground_truth"],
            k=10
        )
        
        # Compile report
        report = {
            "week_ending": now.isoformat(),
            "query_drift": query_drift,
            "new_emerging_topics": new_topics[:5],
            "corpus_staleness": corpus_staleness,
            "ann_quality": ann_quality,
            "overall_health": self._assess_overall_health(
                query_drift, corpus_staleness, ann_quality
            )
        }
        
        # Send alerts for high-severity issues
        if query_drift.get("severity") == "high":
            await self.alerts.send({
                "severity": "high",
                "type": "query_drift",
                "details": query_drift
            })
        
        if corpus_staleness.get("alert"):
            await self.alerts.send({
                "severity": "medium",
                "type": "corpus_staleness",
                "staleness_rate": corpus_staleness["staleness_rate"]
            })
        
        if ann_quality.get("alert"):
            await self.alerts.send({
                "severity": "high",
                "type": "ann_quality_degradation",
                "recall": ann_quality["mean_recall_at_k"]
            })
        
        return report
    
    def _assess_overall_health(
        self,
        query_drift: dict,
        corpus_staleness: dict,
        ann_quality: dict
    ) -> str:
        
        issues = 0
        
        if query_drift.get("severity") in ["high", "medium"]:
            issues += 1
        if corpus_staleness.get("staleness_rate", 0) > 0.05:
            issues += 1
        if ann_quality.get("mean_recall_at_k", 1.0) < 0.85:
            issues += 1
        
        if issues == 0:
            return "healthy"
        elif issues == 1:
            return "warning"
        else:
            return "degraded"
```

---

## Drift Response Playbook

When drift is detected, use this decision tree:

```
Drift detected
    │
    ├── Query distribution shift
    │   ├── New topics emerging
    │   │   └── → Ingest documents for new topics
    │   │       → Add new topic queries to eval set
    │   │       → Consider embedding model fine-tuning
    │   │
    │   └── Existing topics declining
    │       └── → Monitor — may be seasonal or temporary
    │           → Check if corpus for declining topics is still current
    │
    ├── Corpus staleness
    │   ├── Specific documents stale (< 10% of corpus)
    │   │   └── → Prioritize re-indexing flagged documents
    │   │       → Check incremental indexing pipeline health
    │   │
    │   └── Widespread staleness (> 10% of corpus)
    │       └── → Investigate indexing pipeline failure
    │           → Trigger full re-indexing
    │           → Root cause and fix before next cycle
    │
    └── Embedding / ANN drift
        ├── HNSW recall degrading (> 5% drop)
        │   └── → Rebuild HNSW index (blue-green)
        │       → Increase ef parameter as short-term fix
        │
        └── Embedding model change
            └── → Re-embed full corpus
                → Blue-green index switch
                → Validate recall after switch
```

---

## Summary

- RAG systems face three types of drift: query distribution shift (users ask different questions), corpus drift (indexed documents become stale or contradicted), and embedding drift (index quality degrades as corpus grows or model changes).
- Query distribution shift is detected by comparing embedding distributions of recent vs. historical queries using KS statistics and PCA projection. New emerging topics are identified by cluster frequency comparison.
- Corpus staleness is detected by comparing source document modification times against index timestamps. Content contradictions are detected by sampling retrieved chunk pairs for semantic conflicts.
- Embedding/ANN drift is detected by periodically measuring recall@K on test queries against exact nearest neighbors. Coverage monitoring flags new embeddings that fall outside the existing index's dense regions.
- Each drift type has a specific response: corpus ingestion for query drift, re-indexing for corpus staleness, HNSW rebuild for ANN drift.
- Run weekly drift reports combining all three drift types. Alert on severity thresholds. Use a unified dashboard to track drift trends over time.
- Drift monitoring completes the production quality loop — it detects degradation that happens between deployments, not caused by code changes.

---

## What's Next

Part 6 is complete. Part 7 begins with Lesson 7.1 — systematic debugging framework: how to isolate retrieval vs. generation failures and build the diagnostic tooling needed to debug RAG systems at scale.