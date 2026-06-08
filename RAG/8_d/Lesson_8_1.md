# Lesson 8.1 — Vector Database Landscape: Qdrant, Pinecone, Weaviate, pgvector, Milvus

---

## Why the Choice Matters

The vector database is the backbone of your retrieval pipeline. It determines your query latency, your maximum corpus size, your filtering capabilities, your operational complexity, and your cost at scale. Choosing wrong is expensive to fix — migrating a 10M-vector production index from one database to another requires re-embedding the corpus, rebuilding indexes, and migrating metadata.

This lesson gives you the decision framework for choosing the right vector database, grounded in the trade-offs each option makes.

---

## The Five Dimensions to Compare

Every vector database makes trade-offs across five dimensions:

**1. Performance:** Query latency, throughput (QPS), and how both degrade at scale.
**2. Filtering capability:** How well it handles metadata filters alongside vector search.
**3. Operational model:** Self-hosted vs. fully managed, and the ops burden of each.
**4. Ecosystem fit:** What other tools, languages, and frameworks it integrates with.
**5. Cost:** Infrastructure cost (self-hosted) or API pricing (managed).

---

## Qdrant

Qdrant is written in Rust. It is fast, memory-efficient, and built specifically for production RAG and semantic search use cases.

**Strengths:**

**Named vector spaces:** Store multiple embedding models' outputs per document in one collection. This enables hybrid retrieval (dense + sparse), multilingual search (separate vector per language), and model migration (gradually switch from old to new model).

```python
client.create_collection(
    collection_name="docs",
    vectors_config={
        "dense_en": VectorParams(size=1024, distance=Distance.COSINE),
        "dense_de": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse_bm25": SparseVectorParams()
    }
)
```

**Payload indexing:** Create typed indexes on metadata fields for efficient pre-filtering. Keyword, integer, float, datetime, and geo types are all supported.

**Scalar and product quantization:** Built-in quantization reduces memory requirements without requiring a separate pipeline.

**On-disk indexing:** HNSW graph can be stored on disk (SSD) rather than RAM, enabling very large corpora on smaller machines.

**Weaknesses:** Smaller ecosystem than Elasticsearch or Weaviate. Fewer managed deployment options. Documentation can lag feature releases.

**Best for:** Teams building production RAG systems who want a dedicated vector database with hybrid search built in. The sweet spot is 100K to 100M vectors in a self-hosted or cloud-hosted environment.

---

## Pinecone

Pinecone is a fully managed vector database with no self-hosting option. You use it through an API.

**Strengths:**

**Zero operational overhead:** No infrastructure to manage. No HNSW parameter tuning. No index rebuilds. Just put vectors in and query them.

**Automatic scaling:** Pinecone manages horizontal scaling transparently. You pay for what you use.

**Namespaces:** Logical separation of vector groups within one index. Useful for multi-tenancy without separate indexes.

**Weaknesses:**

**No self-hosting:** Data must leave your infrastructure. This blocks regulated industries (healthcare, finance, defense) with strict data residency requirements.

**Limited filtering:** Pinecone's metadata filtering, while functional, is less flexible than Qdrant's or Weaviate's for complex filter expressions.

**Cost at scale:** At 100M+ vectors with high QPS, Pinecone's managed pricing can be significantly more expensive than self-hosted alternatives.

**Vendor lock-in:** Migrating away from Pinecone requires re-uploading all vectors to a new system.

**Best for:** Teams that prioritize time-to-production over cost and control. Startups and prototypes. Organizations without strict data residency requirements and without large-scale vector needs.

---

## Weaviate

Weaviate takes a different architectural philosophy: it is not just a vector database but an "AI-native database" with built-in modules for text vectorization, generative search, and classification.

**Strengths:**

**Built-in modules:** Weaviate can call embedding models automatically at indexing and query time. You pass raw text; Weaviate vectorizes it. This reduces pipeline complexity for teams that want tight integration.

**GraphQL query interface:** Flexible, expressive queries that combine vector search with structured filters and relationship traversal.

**Multi-modal support:** Native support for text, image, and multi-modal objects in the same database.

**Hybrid search:** BM25 + dense retrieval built in, merged with reciprocal rank fusion.

**Weaknesses:**

**Complexity:** Weaviate's schema system is more opinionated than Qdrant's. Setting up modules, schemas, and cross-references has a steeper learning curve.

**Resource requirements:** Weaviate's full feature set requires more resources than a minimal Qdrant deployment.

**Best for:** Teams that want a higher-level abstraction over the vector database — less custom pipeline code, more built-in functionality. Good fit when multi-modal or generative search is a primary use case.

---

## pgvector (PostgreSQL Extension)

pgvector adds vector storage and similarity search to PostgreSQL. It is not a dedicated vector database — it is an extension to a relational database.

**Strengths:**

**Zero new infrastructure:** If you already run PostgreSQL, pgvector adds vector search without a new system. No new operational knowledge needed.

**SQL joins:** You can join vector search results with relational data in a single query. This is powerful for structured + semantic hybrid queries.

```sql
-- Join vector search results with structured data in one query
SELECT 
    d.chunk_id,
    d.text,
    d.doc_title,
    m.metric_value,  -- From a metrics table
    1 - (d.embedding <=> $1::vector) AS similarity
FROM document_chunks d
JOIN financial_metrics m ON d.doc_id = m.doc_id
WHERE d.embedding <=> $1::vector < 0.4  -- cosine distance threshold
AND d.document_status = 'active'
ORDER BY d.embedding <=> $1::vector
LIMIT 10;
```

**ACID transactions:** Vector operations participate in PostgreSQL transactions. Index an embedding and update metadata atomically.

**Weaknesses:**

**Performance ceiling:** pgvector's HNSW implementation is slower and has higher memory requirements than dedicated vector databases at equivalent scale. Above ~1M vectors with high QPS requirements, performance degrades significantly.

**Limited quantization:** No built-in product quantization. Memory requirements grow linearly with corpus size.

**No sparse vector support:** Cannot do BM25-style sparse retrieval natively. Must combine with a separate text search system.

**Best for:** Teams with existing PostgreSQL infrastructure and modest vector search needs (< 1M vectors, < 50 QPS). Small teams that cannot afford additional infrastructure. Use cases where SQL joins between vector search and relational data are critical.

---

## Milvus

Milvus is a distributed vector database designed for massive scale — billions of vectors. It is the underlying technology behind Zilliz Cloud (the managed offering).

**Strengths:**

**Billion-scale:** The only open-source vector database purpose-built for 100M+ to 10B+ vector workloads. Uses IVF-based indexes with GPU acceleration.

**Distributed architecture:** True distributed design with separate components for storage, indexing, and query execution. Each layer scales independently.

**Multiple index types:** HNSW, IVF, IVF-PQ, SCANN, and disk-based indexes. Choose based on your memory/accuracy/speed trade-off.

**Weaknesses:**

**Operational complexity:** Milvus requires deploying and managing multiple components (etcd, MinIO/S3, message queue, index nodes, query nodes). Heavy infrastructure investment.

**Overkill for small corpora:** For < 10M vectors, Milvus's complexity is unjustified. Qdrant or pgvector serve those use cases with far less overhead.

**Slower metadata filtering:** Complex metadata filters are less performant than Qdrant's payload index approach at medium scale.

**Best for:** Enterprise use cases with hundreds of millions to billions of vectors. Large e-commerce (product image search), large-scale content recommendation, genomics, and similar domains where vector count truly demands distributed architecture.

---

## ChromaDB

A lightweight, open-source vector database primarily used for development and small-scale production.

**Strengths:** Extremely simple API, in-memory and persistent modes, Python-native.

**Weaknesses:** Not production-grade at scale. No GPU acceleration, no quantization, no advanced filtering. Memory-limited.

**Best for:** Local development, prototyping, early evaluation. Replace with Qdrant or another production database before going to production.

---

## FAISS (Facebook AI Similarity Search)

FAISS is not a database — it is a library for efficient similarity search. It has no server, no API, no persistence layer. You embed it directly in your application.

**Strengths:** The fastest ANN library available. Supports GPU-accelerated search. Supports HNSW, IVF, IVF-PQ, and many other index types. The gold standard for offline batch similarity search.

**Weaknesses:** No persistence (you manage serialization yourself). No API server. No metadata storage. No concurrent access support. You build everything around it.

**Best for:** Offline batch similarity search pipelines. Embedding similarity in ML training pipelines. Cases where you control all infrastructure and need maximum raw performance. Not for production RAG serving.

---

## Decision Framework

Work through these questions in order:

**1. Do you have strict data residency requirements?**
Yes → Self-hosted only (Qdrant, Weaviate, Milvus, pgvector). Cross out Pinecone.

**2. How many vectors do you need to store?**
< 100K → Any option works. pgvector or ChromaDB for simplicity.
100K–10M → Qdrant is the sweet spot.
10M–100M → Qdrant or Weaviate.
> 100M → Milvus or Zilliz Cloud.

**3. Do you need hybrid search (dense + sparse)?**
Yes → Qdrant (sparse vector support) or Weaviate (built-in BM25). pgvector requires a separate text search system.

**4. Do you have an existing PostgreSQL infrastructure?**
Yes, and needs are modest → pgvector. Keep the stack simple.

**5. Do you need zero operational overhead?**
Yes → Pinecone (managed) or Zilliz Cloud (managed Milvus).

**6. Do you need complex metadata filtering?**
Yes → Qdrant or Weaviate. Both have strong payload/property index support.

**Quick reference:**

| | Qdrant | Pinecone | Weaviate | pgvector | Milvus |
|---|---|---|---|---|---|
| Self-hosted | ✓ | ✗ | ✓ | ✓ | ✓ |
| Managed | ✓ | ✓ | ✓ | RDS | ✓ (Zilliz) |
| Hybrid search | ✓ | Partial | ✓ | ✗ | Partial |
| Max scale | 100M+ | 100M+ | 100M+ | ~1M | 10B+ |
| Metadata filtering | Excellent | Good | Excellent | Via SQL | Good |
| Operational complexity | Low | None | Medium | Low (existing) | High |
| Data residency | ✓ | Partial | ✓ | ✓ | ✓ |

---

## Lesson Learned: Start With Qdrant

For the vast majority of RAG projects, Qdrant is the right starting point. It handles 95% of production use cases, has excellent hybrid search support, reasonable operational requirements, good documentation, and a generous free tier for cloud deployment.

The cases where you should deviate:
- **Pinecone** if time-to-production is more important than cost and you have no data residency constraints.
- **pgvector** if you already run PostgreSQL and your needs are modest.
- **Milvus** if your vector count genuinely exceeds 100M.
- **Weaviate** if you want an all-in-one AI application layer rather than a pure vector store.

---

## What's Next

Lesson 8.2 covers self-hosted vs. managed trade-offs — the cost model, ops burden, and decision criteria for running your own infrastructure vs. using a managed service.