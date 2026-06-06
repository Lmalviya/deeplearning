# Lesson 7.4 — Retrieval Accuracy Degradation: Embedding Drift, Query Distribution Evolution, and Long-Term Maintenance

---

## What This Lesson Covers

Lesson 7.2 covered five root causes of accuracy degradation related to corpus scale: HNSW degradation, BM25 IDF drift, embedding space crowding, filter selectivity, and precision dilution. All five are about what happens as your corpus grows.

This lesson covers a different dimension: what happens as your system operates over time regardless of corpus size. Two forces drive this:

**Embedding drift:** Your embedding model becomes misaligned with your corpus and queries — either because the model is updated, the domain vocabulary evolves, or the mismatch between training distribution and your domain grows as your use case specializes.

**Query distribution evolution:** Your user base changes. New user segments arrive with different query styles. Power users develop new question patterns. External events shift what users ask. The queries your system encounters diverge from the distribution it was built for.

Both forces are invisible — no error is thrown, no alert fires — and both degrade retrieval quality silently over time.

---

## Embedding Drift: The Three Causes

### Cause 1 — Embedding Model Update

You update your embedding model (to a newer version, a better model, or a domain-fine-tuned one). All new queries will use the new model. All existing index vectors were built with the old model. They live in incompatible embedding spaces.

This is the most obvious cause because it requires a deliberate action — you chose to update the model. But the consequences are often underestimated. Even a "minor" model update (v2 to v3 of the same model family) can shift the embedding space enough to degrade retrieval significantly.

**Detection:** Query with the new model against the old index and measure recall@K against your ground truth evaluation set. If recall drops more than 3%, the model change has broken retrieval.

```python
async def measure_model_compatibility(
    old_model_name: str,
    new_model_name: str,
    test_queries: list[str],
    ground_truth: list[list[str]],  # True relevant chunk IDs per query
    vector_db,
    k: int = 10
) -> dict:
    """
    Measure how much retrieval quality degrades when switching embedding models.
    Existing index vectors were built with old_model.
    New queries will use new_model.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    old_model = SentenceTransformer(old_model_name)
    new_model = SentenceTransformer(new_model_name)
    
    recalls_old = []
    recalls_new = []
    
    for query, true_ids in zip(test_queries, ground_truth):
        true_set = set(true_ids)
        
        # Query with old model (baseline — what the system currently does)
        old_emb = old_model.encode(query, normalize_embeddings=True)
        old_results = await vector_db.search(query_vector=old_emb.tolist(), limit=k)
        old_recall = len(set(r.id for r in old_results) & true_set) / len(true_set)
        recalls_old.append(old_recall)
        
        # Query with new model (what will happen after update — vectors not re-embedded)
        new_emb = new_model.encode(query, normalize_embeddings=True)
        new_results = await vector_db.search(query_vector=new_emb.tolist(), limit=k)
        new_recall = len(set(r.id for r in new_results) & true_set) / len(true_set)
        recalls_new.append(new_recall)
    
    recall_drop = float(np.mean(recalls_old)) - float(np.mean(recalls_new))
    
    return {
        "old_model_recall": float(np.mean(recalls_old)),
        "new_model_recall_against_old_index": float(np.mean(recalls_new)),
        "recall_drop": recall_drop,
        "requires_full_reembedding": recall_drop > 0.03,  # 3% drop threshold
        "recommendation": (
            "Full corpus re-embedding required before deploying new model"
            if recall_drop > 0.03
            else "New model compatible — recall within acceptable range"
        )
    }
```

**Fix:** Never deploy a new embedding model without re-embedding the entire corpus first. Use blue-green indexing to make the transition without downtime.

```python
async def migrate_embedding_model(
    old_collection: str,
    new_collection: str,
    new_model_name: str,
    vector_db,
    registry
) -> dict:
    """
    Safely migrate to a new embedding model using blue-green approach.
    """
    from sentence_transformers import SentenceTransformer
    
    new_model = SentenceTransformer(new_model_name)
    
    # Step 1: Create new collection with correct vector dimensions
    new_dim = new_model.get_sentence_embedding_dimension()
    await vector_db.create_collection(
        collection_name=new_collection,
        vector_size=new_dim
    )
    
    # Step 2: Re-embed all chunks and insert into new collection
    all_chunks = await fetch_all_chunks_from_registry(registry)
    
    batch_size = 256
    total_migrated = 0
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        
        # Embed with new model
        new_embeddings = new_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32
        )
        
        # Insert into new collection
        points = [
            {
                "id": chunk["chunk_id"],
                "vector": embedding.tolist(),
                "payload": {
                    **chunk["metadata"],
                    "embedding_model": new_model_name
                }
            }
            for chunk, embedding in zip(batch, new_embeddings)
        ]
        
        await vector_db.upsert(collection=new_collection, points=points)
        total_migrated += len(batch)
    
    # Step 3: Verify new collection quality
    validation = await validate_collection_recall(
        collection=new_collection,
        model=new_model,
        vector_db=vector_db
    )
    
    if validation["recall_at_10"] >= 0.85:
        # Step 4: Switch traffic to new collection
        await switch_active_collection(new_collection)
        await vector_db.delete_collection(old_collection)
        
        return {"status": "success", "total_migrated": total_migrated}
    else:
        # Rollback — keep old collection
        await vector_db.delete_collection(new_collection)
        return {
            "status": "failed",
            "reason": f"New collection recall too low: {validation['recall_at_10']:.2f}"
        }
```

### Cause 2 — Domain Vocabulary Evolution

Your domain's vocabulary changes over time. New technical terms, products, regulations, and acronyms enter the language. Old terms fall out of use. An embedding model trained on data from 2022 may not represent a regulatory term introduced in 2024 well.

This is slower and more subtle than a model update. The gap between your embedding model's training distribution and your current domain widens gradually.

**Detection:** Track out-of-vocabulary (OOV) rate for new documents and queries. High OOV terms are typically tokenized into subwords, which often have poor semantic representations.

```python
def analyze_vocabulary_coverage(
    texts: list[str],
    embedding_model,
    high_frequency_threshold: int = 5
) -> dict:
    """
    Analyze how well the embedding model's vocabulary covers your corpus.
    Looks for terms that appear frequently in your corpus but likely
    as rare or OOV terms in the model's training data.
    """
    from collections import Counter
    import re
    
    # Count term frequencies in your corpus
    all_terms = []
    for text in texts:
        terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]+\b', text.lower())
        all_terms.extend(terms)
    
    term_freq = Counter(all_terms)
    
    # Sample high-frequency terms and check if the model represents them well
    # A proxy: terms that are only 1 token in the tokenizer are better represented
    # than terms that split into many subword tokens
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model.model_card_data.model_name if hasattr(embedding_model, 'model_card_data') else 'bert-base-uncased')
    
    domain_specific_terms = {
        term: freq
        for term, freq in term_freq.items()
        if freq >= high_frequency_threshold and len(term) > 4
    }
    
    fragmented_terms = {}
    for term, freq in domain_specific_terms.items():
        tokens = tokenizer.tokenize(term)
        if len(tokens) > 2:  # Term splits into 3+ subwords = likely OOV
            fragmented_terms[term] = {
                "frequency": freq,
                "n_subword_tokens": len(tokens),
                "subwords": tokens
            }
    
    fragmentation_rate = len(fragmented_terms) / len(domain_specific_terms) if domain_specific_terms else 0
    
    return {
        "total_domain_terms": len(domain_specific_terms),
        "fragmented_terms": len(fragmented_terms),
        "fragmentation_rate": fragmentation_rate,
        "high_fragmentation": fragmentation_rate > 0.20,
        "examples": dict(list(fragmented_terms.items())[:10])
    }
```

**Fix:** When vocabulary evolution is detected, fine-tune the embedding model on recent domain data. Use the techniques from Lesson 2.2: generate (query, relevant_chunk) pairs from recent documents and fine-tune with contrastive loss. Even a small fine-tuning dataset (1,000-5,000 pairs) on recent domain-specific content can significantly improve representation of new vocabulary.

### Cause 3 — Document Style Shift

As your corpus grows, it may include new document types or writing styles not well-represented in your original indexing. Technical papers added to a primarily policy document corpus, or marketing materials added to an engineering knowledge base, create style mismatches.

**Detection:** Track retrieval quality separately by document type. If accuracy degrades for specific document types but not others, style shift is a likely cause.

```python
async def retrieval_quality_by_document_type(
    eval_dataset: list[dict],  # Must include doc_type per item
    retriever
) -> dict:
    """
    Measure recall@K broken down by document type.
    Identifies which document types have poor retrieval quality.
    """
    import numpy as np
    from collections import defaultdict
    
    results_by_type = defaultdict(list)
    
    for item in eval_dataset:
        doc_type = item.get("relevant_doc_type", "unknown")
        
        results = await retriever.retrieve(item["query"], k=10)
        retrieved_ids = set(r["chunk_id"] for r in results)
        relevant_ids = set(item["relevant_chunk_ids"])
        
        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0
        results_by_type[doc_type].append(recall)
    
    return {
        doc_type: {
            "recall_at_10": float(np.mean(recalls)),
            "n_queries": len(recalls)
        }
        for doc_type, recalls in results_by_type.items()
    }
```

---

## Query Distribution Evolution

### How Queries Change Over Time

Your initial user base typically consists of early adopters — often technical users who know how to phrase queries effectively. As adoption grows, new user segments arrive:

- Less technical users who use colloquial language.
- Users from different departments with different domain vocabulary.
- Users with very different task types (the system was built for Q&A but users start asking for summaries, comparisons, or action plans).
- External users (if you open the system to customers or partners).

Each new segment has a different query style, vocabulary, and expectation. A system tuned for early technical users may perform poorly for the new majority.

### Detecting Query Style Shift

```python
async def analyze_query_style_evolution(
    historical_queries: list[dict],  # Queries from N months ago
    current_queries: list[dict],     # Recent queries
    llm_client
) -> dict:
    """
    Identify how query style has changed over time.
    """
    
    def compute_style_features(queries: list[str]) -> dict:
        import re
        import numpy as np
        
        features = {
            "avg_word_count": np.mean([len(q.split()) for q in queries]),
            "pct_questions": np.mean([1.0 if q.strip().endswith("?") else 0.0 for q in queries]),
            "pct_with_technical_terms": np.mean([
                1.0 if re.search(r'\b[A-Z]{2,}\b|\b\w+(?:API|SDK|ML|AI|RAG)\b', q) else 0.0
                for q in queries
            ]),
            "pct_casual_language": np.mean([
                1.0 if re.search(r'\b(how do i|can i|what if|help me|tell me)\b', q.lower()) else 0.0
                for q in queries
            ]),
            "avg_specificity": np.mean([
                # Proxy: queries with numbers or quoted terms are more specific
                1.0 if re.search(r'\d+|"[^"]+"', q) else 0.0
                for q in queries
            ])
        }
        return features
    
    historical_texts = [q["query"] for q in historical_queries[:500]]
    current_texts = [q["query"] for q in current_queries[:500]]
    
    historical_features = compute_style_features(historical_texts)
    current_features = compute_style_features(current_texts)
    
    # Compute feature deltas
    feature_changes = {
        feature: current_features[feature] - historical_features[feature]
        for feature in historical_features
    }
    
    significant_changes = {
        feature: change
        for feature, change in feature_changes.items()
        if abs(change) > 0.1
    }
    
    return {
        "historical_style": historical_features,
        "current_style": current_features,
        "significant_changes": significant_changes,
        "style_shifted": len(significant_changes) > 0,
        "direction": {
            "more_casual": feature_changes.get("pct_casual_language", 0) > 0.1,
            "more_technical": feature_changes.get("pct_with_technical_terms", 0) > 0.1,
            "shorter_queries": feature_changes.get("avg_word_count", 0) < -2,
            "more_specific": feature_changes.get("avg_specificity", 0) > 0.1
        }
    }
```

### Responding to Query Style Shift

**Adjust query understanding:** If queries become more casual, strengthen the query rewriting step. If queries become shorter, use HyDE or step-back prompting to expand the retrieval signal.

**Update the evaluation dataset:** Add examples of the new query style. If your eval set consists only of technical queries but 60% of production queries are now casual, your offline evaluation is no longer predictive of online performance.

**Fine-tune the embedding model on new query patterns:** Use recent user query logs as training queries. Pair them with the chunks that received positive feedback when retrieved for those queries.

```python
async def generate_fine_tuning_data_from_logs(
    query_logs: list[dict],  # {query, retrieved_chunks, user_feedback}
    min_positive_feedback_threshold: float = 0.7
) -> list[dict]:
    """
    Generate embedding model fine-tuning pairs from query logs.
    Uses user feedback (thumbs up, positive ratings) to identify positive pairs.
    """
    
    fine_tuning_pairs = []
    
    for log in query_logs:
        query = log["query"]
        feedback = log.get("user_feedback")
        retrieved_chunks = log.get("retrieved_chunks", [])
        
        if not retrieved_chunks:
            continue
        
        # Queries with positive feedback on the top chunk
        top_chunk = retrieved_chunks[0] if retrieved_chunks else None
        
        if (feedback == "thumbs_up" or 
            (isinstance(feedback, (int, float)) and feedback >= min_positive_feedback_threshold)):
            
            if top_chunk:
                fine_tuning_pairs.append({
                    "query": query,
                    "positive_chunk": top_chunk["text"],
                    "positive_chunk_id": top_chunk["chunk_id"]
                })
    
    # Deduplicate by query
    seen_queries = set()
    unique_pairs = []
    for pair in fine_tuning_pairs:
        if pair["query"] not in seen_queries:
            seen_queries.add(pair["query"])
            unique_pairs.append(pair)
    
    return unique_pairs
```

---

## Long-Term Maintenance Plan

A RAG system requires ongoing maintenance to preserve quality over time. Without it, accuracy will drift downward as corpus, queries, and models all evolve.

### Monthly Tasks

```python
MONTHLY_MAINTENANCE_TASKS = [
    {
        "task": "Measure recall@K on evaluation set",
        "purpose": "Detect early degradation",
        "action_threshold": "recall drops > 3% from baseline",
        "action": "Run root cause analysis (Lesson 7.2 diagnostic)"
    },
    {
        "task": "Rebuild BM25 index",
        "purpose": "Correct IDF drift",
        "action_threshold": "Always",
        "action": "Scheduled rebuild, no manual trigger needed"
    },
    {
        "task": "Check corpus staleness",
        "purpose": "Detect indexing pipeline failures",
        "action_threshold": "staleness rate > 5%",
        "action": "Investigate indexing pipeline, trigger re-index of stale docs"
    },
    {
        "task": "Review query distribution shift",
        "purpose": "Detect evolving user base",
        "action_threshold": "distribution shift score > 0.1",
        "action": "Add new query style examples to eval set, consider fine-tuning"
    }
]
```

### Quarterly Tasks

```python
QUARTERLY_MAINTENANCE_TASKS = [
    {
        "task": "Rebuild HNSW index",
        "purpose": "Restore graph quality after incremental insertions",
        "action": "Blue-green rebuild on scheduled maintenance window"
    },
    {
        "task": "Run corpus conflict audit",
        "purpose": "Find version and content conflicts before they reach users",
        "action": "Review audit report, supersede outdated documents"
    },
    {
        "task": "Evaluate embedding model for updates",
        "purpose": "New embedding models may improve quality",
        "action": "Test new model on eval set, migrate if improvement > 5%"
    },
    {
        "task": "Refresh evaluation dataset",
        "purpose": "Keep eval set representative of current query patterns",
        "action": "Sample recent query logs, annotate 100-200 new examples"
    }
]
```

### Automated Quality Gates

Build automated quality gates that block deployments if retrieval quality degrades:

```python
async def pre_deployment_quality_gate(
    new_system_config: dict,
    eval_dataset: list[dict],
    retriever,
    quality_thresholds: dict
) -> dict:
    """
    Run quality checks before deploying any system change.
    Block deployment if quality drops below thresholds.
    """
    
    results = await run_retrieval_evaluation(eval_dataset, retriever)
    
    failures = []
    
    for metric, threshold in quality_thresholds.items():
        current_value = results.get(metric, 0)
        if current_value < threshold:
            failures.append({
                "metric": metric,
                "current": current_value,
                "threshold": threshold,
                "gap": threshold - current_value
            })
    
    passed = len(failures) == 0
    
    return {
        "passed": passed,
        "results": results,
        "failures": failures,
        "recommendation": (
            "Deployment approved — all quality gates passed"
            if passed
            else f"Deployment blocked — {len(failures)} quality gate(s) failed"
        )
    }

# Standard thresholds for a production RAG system
QUALITY_THRESHOLDS = {
    "hit_rate@5": 0.88,
    "recall@10": 0.80,
    "mrr": 0.70,
    "precision@5": 0.60
}
```

---

## The Accuracy Maintenance Dashboard

Track all accuracy-related signals in a single view:

```python
class AccuracyMaintenanceDashboard:
    """
    Tracks all signals relevant to long-term retrieval accuracy.
    """
    
    def __init__(self, metrics_store):
        self.metrics = metrics_store
    
    async def generate_report(self) -> dict:
        """Weekly accuracy maintenance report."""
        
        latest_eval = await self.metrics.get_latest("retrieval_evaluation")
        baseline_eval = await self.metrics.get_baseline("retrieval_evaluation")
        
        return {
            "retrieval_quality": {
                "current_recall@10": latest_eval.get("recall@10"),
                "baseline_recall@10": baseline_eval.get("recall@10"),
                "trend": "degrading" if latest_eval.get("recall@10", 0) < baseline_eval.get("recall@10", 0) * 0.97 else "stable",
            },
            "corpus_health": {
                "staleness_rate": await self.metrics.get_latest_value("corpus_staleness_rate"),
                "index_coverage": await self.metrics.get_latest_value("index_coverage"),
                "version_conflicts_open": await self.metrics.count("open_version_conflicts")
            },
            "embedding_health": {
                "ann_recall": await self.metrics.get_latest_value("ann_recall@10"),
                "embedding_model_version": await self.metrics.get_config("embedding_model"),
                "last_index_rebuild": await self.metrics.get_latest_value("last_hnsw_rebuild")
            },
            "query_evolution": {
                "distribution_shift_score": await self.metrics.get_latest_value("query_drift_score"),
                "new_topics_detected": await self.metrics.get_latest_value("new_topic_count"),
                "vocabulary_coverage": await self.metrics.get_latest_value("vocab_fragmentation_rate")
            },
            "actions_needed": await self._compute_needed_actions()
        }
```

---

## Summary

- Retrieval accuracy degrades over time from two forces independent of corpus size: embedding drift (model-corpus misalignment) and query distribution evolution (user base change).
- Three causes of embedding drift: model updates (immediate, measurable), domain vocabulary evolution (slow, subtle), and document style shift (detectable by type-specific quality tracking).
- Always measure compatibility before deploying a new embedding model against the existing index. A >3% recall drop requires full corpus re-embedding first.
- Query distribution evolution means the queries users ask today differ from when the system was built. Detect it with style feature analysis and embedding distribution comparison.
- Respond to query evolution by strengthening query understanding, updating the evaluation dataset, and fine-tuning the embedding model on recent query-feedback pairs.
- Long-term maintenance requires scheduled tasks: monthly recall measurement, BM25 rebuild, staleness check; quarterly HNSW rebuild, conflict audit, embedding model evaluation, eval dataset refresh.
- Automated quality gates in the deployment pipeline prevent degraded versions from reaching production.

---

## What's Next

Lesson 7.5 covers tracing and observability — building a RAG trace that captures every decision from query to response, how to use traces for debugging, and what production observability infrastructure looks like for RAG systems.