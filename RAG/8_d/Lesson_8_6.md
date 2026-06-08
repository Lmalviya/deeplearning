# Lesson 8.6 — Scaling the Retrieval Layer: Read Replicas, Sharding, and Caching

---

## The Retrieval Bottleneck at Scale

As query volume grows, the retrieval layer becomes the primary bottleneck. Unlike stateless API pods that scale horizontally with ease, vector databases are stateful — scaling them requires careful design.

At 10 QPS, a single Qdrant node handles retrieval comfortably. At 1,000 QPS, you need an architecture that distributes search load across multiple nodes while keeping indexes consistent.

Three scaling strategies work independently and in combination:
1. **Read replicas:** Multiple copies of the same index serve read (search) traffic. Writes go to one primary and replicate to others.
2. **Sharding:** Split the corpus across multiple nodes. Each node holds a fraction of the vectors. Queries fan out to all shards and results are merged.
3. **Caching:** Serve repeated or semantically similar queries from cache instead of hitting the vector database.

---

## Strategy 1: Read Replicas

Read replicas are the simplest scaling pattern. The primary node handles all writes (indexing new chunks). Read replicas copy the index and serve search queries.

```
[Indexing Workers] → [Primary Qdrant] → [Replica 1]
                                       → [Replica 2]
                                       → [Replica 3]

[RAG API] → [Load Balancer] → [Replica 1]
                             → [Replica 2]
                             → [Replica 3]
```

In Qdrant's cluster mode, this is implemented as a collection with multiple replicas:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(
    url="http://qdrant-cluster:6333"  # Points to cluster entry point
)

# Create collection with 1 shard and 3 replicas (1 primary + 2 replicas)
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    shard_number=1,        # One shard (all data on each node)
    replication_factor=3   # 3 total copies
)
```

**Read vs. write routing:**

```python
# For search (read): can hit any replica
search_results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=50,
    # Qdrant automatically routes to a healthy replica
)

# For write (upsert): automatically routed to shard leader
client.upsert(
    collection_name="documents",
    points=[...],
    # Write goes to leader, propagates to replicas
)
```

**Scaling read throughput:** With 3 replicas, you triple the maximum search QPS. With 5 replicas, 5×. Read replicas scale search throughput linearly with replica count.

**Limitations of replicas:** Every replica holds the full index. If the index is 200GB, each replica needs 200GB of RAM. Replicas scale QPS but not corpus size.

---

## Strategy 2: Sharding

Sharding splits the corpus across multiple nodes. Each node holds a subset of vectors. A search query fans out to all shards in parallel, and results are merged.

```
                [Search Query]
                      │
           ┌──────────┴──────────┐
           ↓                     ↓
     [Shard 1]              [Shard 2]
   Chunks 1-5M             Chunks 5M-10M
           │                     │
           └──────────┬──────────┘
                      ↓
              [Merge Top-K]
                      ↓
              [Final Top-10]
```

```python
# Create a sharded collection (Qdrant distributes shards automatically)
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    shard_number=4,         # 4 shards distributed across cluster nodes
    replication_factor=2    # Each shard has 2 copies (for fault tolerance)
)

# Search is automatically distributed — same API as single-node search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=50  # Qdrant fetches top-50 per shard, merges, returns global top-50
)
```

**The merge math:** When searching top-50 across 4 shards, Qdrant retrieves top-50 from each shard (200 total candidates), then merges to find the true global top-50. This works correctly for unfiltered searches.

**With metadata filters and sharding:** Pre-filtering must happen on each shard independently. If your filter is highly selective (matches 0.1% of vectors), most shards may return zero results — this is fine, the merge handles it. But very selective filters on a sharded index can cause the same filter selectivity problem described in Lesson 7.2.

**Custom sharding for tenant isolation:**

```python
# Assign chunks to specific shards based on tenant
# Useful for multi-tenant RAG where each tenant's data lives on one shard

client.upsert(
    collection_name="documents",
    points=[...],
    shard_key_selector="tenant_123"  # Route to tenant's dedicated shard
)

# Search only within a specific shard
client.search(
    collection_name="documents",
    query_vector=query_embedding,
    shard_key_selector="tenant_123",  # Only search this tenant's shard
    limit=50
)
```

Custom sharding with tenant-based shard keys ensures tenant data isolation at the storage level and improves search performance by eliminating cross-tenant noise.

---

## Strategy 3: Query Result Caching

Caching is the highest-leverage optimization for RAG systems where queries repeat. Three cache layers work at different granularities.

### Cache Layer 1: Exact Query Cache

```python
import hashlib
import json
import redis.asyncio as aioredis
from datetime import timedelta

class ExactQueryCache:
    def __init__(self, redis_url: str, ttl_seconds: int = 3600):
        self.redis = aioredis.from_url(redis_url)
        self.ttl = ttl_seconds
    
    def _make_key(self, query: str, filters: dict, k: int) -> str:
        """Deterministic cache key from query parameters."""
        payload = json.dumps({
            "query": query.lower().strip(),
            "filters": filters,
            "k": k
        }, sort_keys=True)
        return f"exact:{hashlib.sha256(payload.encode()).hexdigest()}"
    
    async def get(self, query: str, filters: dict, k: int) -> list[dict] | None:
        key = self._make_key(query, filters, k)
        cached = await self.redis.get(key)
        if cached:
            await self.redis.incr(f"cache_hits:{key[:16]}")  # Track hits
            return json.loads(cached)
        return None
    
    async def set(
        self,
        query: str,
        filters: dict,
        k: int,
        results: list[dict],
        ttl_override: int = None
    ):
        key = self._make_key(query, filters, k)
        await self.redis.setex(
            key,
            ttl_override or self.ttl,
            json.dumps(results)
        )
    
    async def invalidate_by_doc(self, doc_id: str):
        """
        Invalidate all cached results that contain chunks from this document.
        Called when a document is updated or deleted.
        """
        # In practice, tag cache entries with doc_ids at write time
        # Then use Redis SET operations to find and delete related entries
        # Simplified version: flush all (safe but inefficient)
        # Production: use Redis SCAN with pattern matching or tag-based invalidation
        pass
```

### Cache Layer 2: Semantic Similarity Cache

```python
import numpy as np

class SemanticCache:
    """
    Cache retrieval results for queries that are semantically similar to previous queries.
    Uses a small vector index of past queries.
    """
    
    def __init__(
        self,
        vector_db_client,
        embedding_model,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        ttl_hours: int = 4
    ):
        self.vdb = vector_db_client
        self.embedder = embedding_model
        self.threshold = similarity_threshold
        self.max_size = max_cache_size
        self.ttl = ttl_hours
    
    async def lookup(self, query: str, query_embedding: list[float]) -> dict | None:
        """
        Find a semantically similar cached query.
        Returns cached result if found, None if cache miss.
        """
        try:
            results = await self.vdb.search(
                collection_name="query_cache",
                query_vector=query_embedding,
                limit=1,
                score_threshold=self.threshold,
                with_payload=True
            )
            
            if results and results[0].score >= self.threshold:
                cached_payload = results[0].payload
                
                # Check TTL
                import datetime
                cached_at = datetime.datetime.fromisoformat(cached_payload["cached_at"])
                age_hours = (datetime.datetime.utcnow() - cached_at).total_seconds() / 3600
                
                if age_hours <= self.ttl:
                    return {
                        "results": cached_payload["results"],
                        "cache_score": results[0].score,
                        "original_query": cached_payload["original_query"]
                    }
        except Exception:
            pass  # Cache miss on error
        
        return None
    
    async def store(
        self,
        query: str,
        query_embedding: list[float],
        results: list[dict]
    ):
        """Store retrieval results in the semantic cache."""
        import uuid, datetime
        
        cache_id = str(uuid.uuid4())
        
        await self.vdb.upsert(
            collection_name="query_cache",
            points=[{
                "id": cache_id,
                "vector": query_embedding,
                "payload": {
                    "original_query": query,
                    "results": results[:20],  # Store top-20 for re-ranking flexibility
                    "cached_at": datetime.datetime.utcnow().isoformat(),
                    "hit_count": 0
                }
            }]
        )
```

### Cache Layer 3: Embedding Cache

The embedding computation is expensive and often redundant — the same query appears thousands of times. Cache query embeddings:

```python
class EmbeddingCache:
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def get_or_compute(
        self,
        text: str,
        embedding_fn,
        ttl_seconds: int = 86400
    ) -> list[float]:
        """Get embedding from cache or compute and cache it."""
        
        cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
        
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        embedding = await embedding_fn(text)
        
        await self.redis.setex(
            cache_key,
            ttl_seconds,
            json.dumps(embedding)
        )
        
        return embedding
```

### Putting Caching Together

```python
async def retrieve_with_full_caching(
    query: str,
    filters: dict,
    exact_cache: ExactQueryCache,
    semantic_cache: SemanticCache,
    embedding_cache: EmbeddingCache,
    vector_db,
    embedding_model,
    reranker,
    k: int = 10
) -> dict:
    
    # Layer 1: Exact cache
    exact_hit = await exact_cache.get(query, filters, k)
    if exact_hit:
        return {"results": exact_hit, "source": "exact_cache"}
    
    # Layer 2: Embedding cache (expensive computation)
    query_embedding = await embedding_cache.get_or_compute(
        text=query,
        embedding_fn=embedding_model.embed
    )
    
    # Layer 3: Semantic cache
    semantic_hit = await semantic_cache.lookup(query, query_embedding)
    if semantic_hit:
        # Re-rank the cached results for this specific query
        # (The semantically similar query may have retrieved slightly different docs)
        reranked = reranker.rerank(query, semantic_hit["results"])[:k]
        return {"results": reranked, "source": "semantic_cache", "cache_score": semantic_hit["cache_score"]}
    
    # Cache miss — run full retrieval
    results = await vector_db.search(
        query_vector=query_embedding,
        filter=filters,
        limit=50
    )
    
    reranked = reranker.rerank(query, results)[:k]
    
    # Store in caches for future queries
    await exact_cache.set(query, filters, k, reranked, ttl_override=1800)
    await semantic_cache.store(query, query_embedding, results)
    
    return {"results": reranked, "source": "live_retrieval"}
```

---

## Cache Invalidation: The Hard Part

The famous quote: "There are only two hard things in Computer Science: cache invalidation and naming things."

For RAG caches, invalidation is triggered by:

**Document updates:** When a document is updated or deleted, all cached results that include chunks from that document should be invalidated.

**Index rebuilds:** When the HNSW index is rebuilt (Lesson 7.2), cached embeddings remain valid, but cached retrieval results may change (different ANN results after rebuild).

**Embedding model updates:** When the embedding model changes, all cached embeddings are invalid. Invalidate the entire embedding cache.

```python
class CacheInvalidator:
    def __init__(self, redis, semantic_cache_collection: str):
        self.redis = redis
        self.semantic_collection = semantic_cache_collection
    
    async def on_document_updated(self, doc_id: str):
        """Called when a document is re-indexed."""
        
        # Invalidate exact cache entries that reference this doc
        # (Requires tagging cache entries with doc_ids at write time)
        tag_key = f"doc_cache_tag:{doc_id}"
        cache_keys = await self.redis.smembers(tag_key)
        
        if cache_keys:
            await self.redis.delete(*cache_keys)
            await self.redis.delete(tag_key)
        
        # For semantic cache: cannot efficiently find specific entries
        # Options:
        # 1. Accept temporary staleness (TTL will expire stale entries)
        # 2. Clear entire semantic cache (expensive but safe)
        # 3. Tag semantic cache entries with doc_ids and delete by tag
        
        # For most use cases, option 1 is acceptable (TTL = 1-4 hours)
        # For real-time accuracy requirements, option 2 or 3
    
    async def on_embedding_model_update(self):
        """Called when the embedding model is changed."""
        # All embedding caches are invalid
        await self.redis.flushdb()  # Nuclear option — clears all Redis data
        # Better: use key prefixes and delete by pattern
        async for key in self.redis.scan_iter("emb:*"):
            await self.redis.delete(key)
```

---

## Monitoring Cache Effectiveness

```python
async def collect_cache_metrics(redis, period_minutes: int = 60) -> dict:
    """
    Compute cache performance metrics for the monitoring dashboard.
    """
    
    exact_hits = int(await redis.get("metrics:exact_cache_hits") or 0)
    semantic_hits = int(await redis.get("metrics:semantic_cache_hits") or 0)
    live_retrievals = int(await redis.get("metrics:live_retrievals") or 0)
    
    total = exact_hits + semantic_hits + live_retrievals
    
    if total == 0:
        return {"no_data": True}
    
    return {
        "total_queries": total,
        "exact_cache_hit_rate": exact_hits / total,
        "semantic_cache_hit_rate": semantic_hits / total,
        "live_retrieval_rate": live_retrievals / total,
        "overall_cache_hit_rate": (exact_hits + semantic_hits) / total,
        
        # Cost savings (approximate)
        "queries_served_without_qdrant_search": exact_hits + semantic_hits,
        "estimated_cost_saved_usd": (exact_hits + semantic_hits) * 0.002  # $0.002 per search
    }
```

---

## When Each Strategy Applies

| Scenario | Strategy |
|---|---|
| < 500 QPS | Single Qdrant node, no special scaling needed |
| 500-2000 QPS, same corpus | Read replicas (2-4 replicas) |
| > 100M vectors | Sharding (4-8 shards) |
| High QPS with repetitive queries | Caching (exact + semantic) |
| Multi-tenant with strict isolation | Custom sharding by tenant |
| Very high QPS (> 5000) | Replicas + sharding + caching combined |

---

## Summary

- Read replicas distribute search load across multiple nodes, each holding a full copy of the index. Scale QPS linearly with replica count. Does not reduce memory requirements per node.
- Sharding distributes the corpus across nodes. Each node holds a fraction of vectors. Search fans out to all shards and results are merged. Scales both corpus size and QPS.
- Three cache layers: exact query cache (Redis, 35%+ hit rate for support bots), semantic similarity cache (vector index of past query embeddings, 20%+ additional), embedding cache (avoid recomputing same query embeddings).
- Cache invalidation is the hard part. Tag cache entries with source doc_ids at write time to enable efficient invalidation when documents update.
- Monitor cache hit rate, live retrieval rate, and estimated cost savings as standard metrics.

---

## What's Next

Lesson 8.7 covers serving LLMs: self-hosted (vLLM, Ollama) vs. API, the latency-cost trade-off, and how to make the deployment decision for your specific use case.