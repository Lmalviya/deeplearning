# Lesson 3.1 — Dense Retrieval Internals: HNSW, IVF, Product Quantization, and ANN Search

---

## Why You Need to Understand the Index Internals

Most RAG tutorials treat the vector database as a black box: put vectors in, get similar vectors out. This works until you need to:

- Tune retrieval performance when accuracy degrades at scale.
- Understand why recall drops from 95% at 10K documents to 70% at 5M documents.
- Configure index parameters for your specific latency vs. accuracy trade-off.
- Debug why filtered search is suddenly 10x slower than unfiltered search.
- Design a system that scales from prototype to millions of vectors without a full rebuild.

None of these is possible if the index is a black box. This lesson opens that box.

---

## The Exact Nearest Neighbor Problem

The fundamental retrieval operation is: given a query vector `q` and a corpus of `N` vectors, find the K vectors most similar to `q`.

If you do this naively — compute cosine similarity between `q` and every vector in the corpus — this is O(N × D) where D is the vector dimension. For:

- N = 1,000,000 vectors
- D = 1,536 dimensions
- Each comparison requires 1,536 multiply-add operations

Total: 1.5 billion floating point operations per query. On modern hardware this takes roughly 1–3 seconds. For a system handling 100 queries per second, that is 150–300 billion operations per second — not feasible.

Exact nearest neighbor search does not scale. Every vector index structure is an approximation that trades a small amount of accuracy for orders-of-magnitude speed improvement. This is why they are called **Approximate Nearest Neighbor (ANN)** algorithms.

The trade-off is measured by **recall@K**: what fraction of the true top-K nearest neighbors does the approximate algorithm find? A recall@10 of 0.95 means the ANN algorithm finds 9.5 out of the true 10 nearest neighbors on average. The missing 0.5 is the cost of approximation.

---

## HNSW — Hierarchical Navigable Small World

HNSW (Malkov & Yashunin, 2018) is the dominant ANN algorithm in production vector databases today. Qdrant, Weaviate, and most modern systems use HNSW as their primary index. Understanding it well is important.

### The Intuition: Navigable Small World Graphs

The Navigable Small World (NSW) insight comes from graph theory and social networks. In any large network, you can get from any node to any other node in a surprisingly small number of hops — this is the "six degrees of separation" phenomenon.

NSW builds a graph where each vector is a node, and each node is connected to its approximate nearest neighbors. To find the nearest neighbor of a query vector, you start at an entry point node and greedily traverse the graph: at each step, move to whichever connected neighbor is closest to the query. Stop when no neighbor is closer than the current node.

This greedy traversal is fast but can get stuck in local minima — you reach a node where all its neighbors are farther from the query than the node itself, but the true nearest neighbor is elsewhere in the graph.

**HNSW fixes this** with a hierarchical layer structure.

### The Hierarchical Layer Structure

HNSW builds multiple layers of the graph, from sparse (top) to dense (bottom):

```
Layer 2 (sparse):    O ————————————————— O
                      \                 /
Layer 1 (medium):    O — O ———— O — O — O
                      \   \   /   \ | /
Layer 0 (dense):     O—O—O—O—O—O—O—O—O—O—O  (all vectors)
```

- **Layer 0** contains all vectors, connected to their nearest neighbors.
- **Higher layers** contain a random subset of vectors with longer-range connections.
- Each vector exists in layer 0, and may probabilistically be promoted to higher layers.

### How Search Works

```
1. Start at the entry point in the highest layer (Layer 2 in example above).
2. Greedily traverse Layer 2 toward the query — take the neighbor closest to q.
3. When no progress can be made in Layer 2, descend to Layer 1 at the current best node.
4. Greedily traverse Layer 1 toward the query.
5. Descend to Layer 0 and traverse to find the final nearest neighbors.
```

The sparse upper layers enable large jumps across the vector space, escaping local minima. The dense lower layers provide precise local search. This two-scale navigation is what makes HNSW both fast and accurate.

### How Insertion Works

When a new vector is added:
1. Randomly assign its maximum layer (most vectors land in layer 0, fewer in layer 1, very few in layer 2+).
2. Starting from the top layer, search for the insertion point.
3. In each layer, find the M nearest neighbors of the new vector.
4. Add bidirectional edges between the new vector and its M nearest neighbors.
5. If any existing node now has too many connections (> M_max), prune the weakest edges.

This is why HNSW supports **efficient incremental insertion** — you can add new vectors without rebuilding the entire index. This is critical for incremental indexing (Lesson 2.6).

### Key HNSW Parameters

These are the parameters you tune when configuring a vector database:

**`M` (number of connections per node):**
- Controls how many bidirectional edges each node maintains in the graph.
- Typical values: 16–64.
- Higher M → better recall (more paths to explore), more memory (each connection requires storage), slower insertion.
- Lower M → less memory, faster insertion, lower recall.
- Rule of thumb: M = 16 for memory-constrained environments, M = 32–64 for high-recall requirements.

**`ef_construction` (search width during index building):**
- Controls how thoroughly the algorithm searches for neighbors when inserting a new vector.
- Typical values: 100–500.
- Higher ef_construction → better graph quality (more accurate neighbor connections), slower index build time.
- Does not affect query speed or memory — only affects build time and final index quality.
- Rule of thumb: ef_construction = 2 × M minimum; higher values improve recall on difficult queries.

**`ef` (search width at query time, sometimes called `ef_search`):**
- Controls how many candidate nodes are explored during search.
- Typical values: 50–500.
- Higher ef → better recall, slower query.
- Can be set at query time without rebuilding the index — this is the primary runtime tuning knob.
- Rule of thumb: ef ≥ K (number of results requested). Start at ef = 100 and adjust based on recall measurements.

```python
# Qdrant HNSW configuration
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,                    # connections per node
        ef_construct=200,        # build-time search width
        full_scan_threshold=10000  # use brute force below this count
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000  # start building HNSW after this many vectors
    )
)

# At query time, specify ef for this specific search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    search_params=SearchParams(hnsw_ef=128)  # runtime ef parameter
)
```

### HNSW Memory Requirements

HNSW stores the graph structure in RAM for fast traversal. Memory consumption:

```
Memory ≈ N × (D × 4 bytes + M × 2 × 8 bytes)
       = N × (D × 4 + M × 16) bytes
```

For N = 1,000,000 vectors, D = 1536 dimensions, M = 16:
```
Memory ≈ 1,000,000 × (1536 × 4 + 16 × 16)
       ≈ 1,000,000 × (6,144 + 256)
       ≈ 1,000,000 × 6,400
       ≈ 6.4 GB
```

For 10M vectors: ~64 GB. For 100M vectors: ~640 GB. HNSW memory requirements become a hard constraint at very large scale.

---

## IVF — Inverted File Index

IVF (Inverted File Index) takes a fundamentally different approach. Instead of building a navigable graph, it clusters the vector space and at query time only searches within the nearest clusters.

### How IVF Works

**Building the index (offline, requires training step):**

1. Run k-means clustering on all vectors to create `nlist` cluster centroids.
2. Assign each vector to its nearest centroid.
3. Store vectors grouped by their cluster assignment.

```
Centroid 1: [vectors 23, 891, 4521, 9023, ...]
Centroid 2: [vectors 7, 145, 2890, ...]
Centroid 3: [vectors 56, 788, 3401, ...]
...
Centroid nlist: [vectors ...]
```

**At query time:**

1. Compute the distance from the query vector to all `nlist` centroids.
2. Select the `nprobe` nearest centroids (nprobe << nlist).
3. Search exhaustively through only the vectors in those `nprobe` clusters.
4. Return the K nearest vectors found.

The speed gain comes from searching a fraction of the corpus. If nlist = 1000 and nprobe = 10, you search only 1% of clusters.

### Key IVF Parameters

**`nlist` (number of clusters):**
- More clusters → smaller clusters → faster search but requires higher nprobe for good recall.
- Rule of thumb: `nlist = sqrt(N)` for N vectors. For 1M vectors, nlist ≈ 1000.
- At minimum, each cluster should contain ~39 vectors on average (FAISS recommendation).

**`nprobe` (number of clusters to search at query time):**
- Higher nprobe → better recall, slower query.
- This is the runtime tuning knob (like ef in HNSW).
- Rule of thumb: start with nprobe = nlist / 10. Increase until recall target is met.

**`nprobe` vs. recall trade-off example:**

| nprobe | Recall@10 | Query time (1M vectors) |
|--------|-----------|------------------------|
| 1 | ~0.50 | 0.5ms |
| 10 | ~0.85 | 2ms |
| 50 | ~0.95 | 8ms |
| 100 | ~0.98 | 15ms |
| nlist | ~1.00 | 150ms (exact search) |

### IVF Training Requirement

IVF requires a training step before you can add vectors. The k-means clustering needs representative data to create good cluster centroids. This means:

- You cannot build an IVF index from scratch with zero vectors.
- Adding many new vectors after training may fall outside the original cluster structure, degrading recall.
- If the vector distribution changes significantly (new document types, new embedding model), you need to retrain.

This is IVF's main weakness compared to HNSW: **IVF is not well-suited to dynamic corpora** where vectors are continuously added. HNSW handles incremental insertion natively. IVF works best for static or slowly changing corpora.

```python
# FAISS IVF example (lower level than Qdrant)
import faiss
import numpy as np

d = 1536      # vector dimension
nlist = 1000  # number of clusters

# Train the index (needs representative data)
quantizer = faiss.IndexFlatL2(d)  # exact search for cluster assignment
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Training data (needs at least 39 * nlist vectors)
training_vectors = np.random.randn(50000, d).astype('float32')
index.train(training_vectors)

# Add vectors
corpus_vectors = np.random.randn(1000000, d).astype('float32')
index.add(corpus_vectors)

# Search
index.nprobe = 50  # runtime parameter
query = np.random.randn(1, d).astype('float32')
distances, indices = index.search(query, k=10)
```

---

## Product Quantization (PQ) — Reducing Memory

Both HNSW and IVF store full-precision float32 vectors. For large corpora this is expensive:
- 1M vectors × 1536 dims × 4 bytes = 6 GB just for the raw vectors.
- 100M vectors = 600 GB.

Product Quantization compresses vectors to a fraction of their original size with acceptable accuracy loss.

### How PQ Works

1. Split each D-dimensional vector into `M` sub-vectors of D/M dimensions each.
2. For each sub-vector position, run k-means with 256 clusters (fits in 1 byte).
3. Replace each sub-vector with the index (0–255) of its nearest cluster centroid.

A 1536-dimensional vector split into 96 sub-vectors of 16 dimensions each:
- Original: 1536 × 4 bytes = 6,144 bytes per vector.
- After PQ: 96 × 1 byte = 96 bytes per vector. **64× compression.**

At search time, distances are approximated using precomputed lookup tables between query sub-vectors and centroids.

```python
# FAISS IVF with Product Quantization
import faiss

d = 1536      # dimension
nlist = 1000  # clusters
M = 96        # number of sub-vectors (d must be divisible by M)
nbits = 8     # bits per sub-vector code (256 centroids)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# Training requires more data with PQ
index.train(training_vectors)
index.add(corpus_vectors)
```

**PQ trade-offs:**
- Memory: 64× reduction is typical.
- Recall: 3–10% recall drop compared to exact distance computation.
- Speed: Similar to IVF or faster (compressed distance computation).

**IVFPQ** (IVF + PQ) is the standard combination for very large corpora where memory is the binding constraint.

### Scalar Quantization (SQ)

A simpler alternative to PQ: quantize each dimension from float32 to int8 (or int4). 4× memory reduction with very small accuracy loss.

Qdrant supports scalar quantization natively:

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # clip outliers at 99th percentile
            always_ram=True       # keep quantized vectors in RAM
        )
    )
)
```

Scalar quantization is a good first step when memory is a concern — 4× savings with minimal quality impact. Use PQ when you need more than 4× compression.

---

## HNSW vs. IVF: When to Use Which

| | HNSW | IVF |
|---|---|---|
| **Recall quality** | Higher (same parameters) | Slightly lower |
| **Memory** | Higher (graph overhead) | Lower |
| **Query speed** | Fast | Fast (with good nprobe) |
| **Build speed** | Slower (graph construction) | Faster |
| **Incremental inserts** | Native support | Degrades without retraining |
| **Dynamic corpora** | Excellent | Poor |
| **Static corpora** | Good | Excellent |
| **Typical use case** | Most production RAG systems | Very large static corpora |

**For RAG systems:** HNSW is almost always the right choice. RAG document corpora are dynamic — documents are added, updated, and deleted continuously. HNSW handles this natively. IVF is the right choice when you have a corpus of 100M+ vectors that rarely changes and memory is a hard constraint.

---

## Metadata Filtering and Its Impact on ANN Search

This is where many teams discover a painful performance cliff.

When you add a metadata filter to a vector search, the ANN algorithm must find K nearest neighbors within only the subset of vectors that pass the filter. If the filtered subset is small, the graph traversal (in HNSW) may need to explore many more nodes to find K qualifying vectors, degrading both speed and recall.

### The Small Filter Problem

Consider: you have 1M vectors, HNSW with ef=128. A query without filtering explores ~128 candidate nodes and returns with excellent recall in 5ms.

Now add a filter that matches only 1% of vectors (10,000 out of 1M). The HNSW graph was built for the full 1M vectors. During traversal, 99% of visited nodes fail the filter and are discarded. The algorithm needs to explore far more nodes to find K qualifying results. Some databases handle this by automatically falling back to brute force search over the filtered subset.

**Solutions:**

**1. Partition by high-cardinality filter fields.** If you always filter by `department`, create one collection per department. Each collection is smaller, graph traversal is more efficient.

```python
# Instead of one collection with department filter:
# collection: "documents" + filter: {"department": "engineering"}

# Use separate collections:
# collection: "documents_engineering"
# collection: "documents_hr"
# collection: "documents_finance"
```

**2. Use payload indexes.** Qdrant and Weaviate build separate inverted indexes for metadata fields. When a filter is applied, the database first identifies qualifying vector IDs using the inverted index, then performs ANN search only within those IDs. This is more efficient than post-filtering.

**3. Increase ef for heavily filtered queries.** A higher ef compensates for the higher miss rate on filtered nodes by exploring more of the graph.

**4. Use HNSW with filterable HNSW.** Qdrant's filtered HNSW builds separate graph structures that account for common filter patterns, maintaining recall quality even with aggressive filtering.

> **Interview note:** "What happens to retrieval performance when you add metadata filters?" — the answer interviewers want: it depends on filter selectivity. High selectivity (filters out most vectors) degrades ANN accuracy because the graph was built for the full distribution. Solutions are: partition collections by filter field, use payload indexes, or increase ef for filtered queries.

---

## Measuring and Monitoring ANN Quality

Never assume your ANN index is performing well — measure it.

### Recall@K Measurement

```python
import numpy as np

def measure_recall_at_k(
    index,             # your ANN index
    exact_index,       # brute force index (faiss.IndexFlatL2)
    test_queries: np.ndarray,
    k: int = 10
) -> float:
    """
    Measure recall@K: fraction of true top-K neighbors found by ANN.
    """
    n_queries = len(test_queries)
    
    # Get exact top-K (ground truth)
    _, exact_indices = exact_index.search(test_queries, k)
    
    # Get ANN top-K
    _, ann_indices = index.search(test_queries, k)
    
    # Compute recall for each query
    recalls = []
    for i in range(n_queries):
        true_set = set(exact_indices[i])
        ann_set = set(ann_indices[i])
        recall = len(true_set & ann_set) / k
        recalls.append(recall)
    
    return np.mean(recalls)

# Run periodically in production
recall = measure_recall_at_k(production_index, exact_index, sample_queries, k=10)
print(f"Recall@10: {recall:.3f}")

if recall < 0.90:
    alert("ANN recall below threshold — consider rebuilding index or tuning ef")
```

### Latency Percentiles

Track P50, P95, and P99 query latency separately:

```python
import time
import numpy as np

def benchmark_index(index, queries: np.ndarray, k: int = 10) -> dict:
    latencies = []
    
    for query in queries:
        start = time.perf_counter()
        index.search(query.reshape(1, -1), k)
        latencies.append(time.perf_counter() - start)
    
    latencies = sorted(latencies)
    n = len(latencies)
    
    return {
        "p50_ms": latencies[int(n * 0.50)] * 1000,
        "p95_ms": latencies[int(n * 0.95)] * 1000,
        "p99_ms": latencies[int(n * 0.99)] * 1000,
        "mean_ms": np.mean(latencies) * 1000
    }
```

P99 latency is what your worst-case users experience. It is often 3–5× the P50. Always optimize for P99, not mean.

---

## Putting It Together: Index Configuration for Production

A practical checklist for configuring a vector index for a RAG system:

**1. Estimate corpus size and growth rate.**
How many vectors today? In 6 months? In 2 years? Choose an index type that handles your 2-year projection.

**2. Set vector dimension based on your embedding model.**
Do not configure a 768-dim index for a 1536-dim model — this is a silent bug that produces garbage results.

**3. Choose HNSW for dynamic corpora (almost always the right choice for RAG).**
Start with M=16, ef_construction=200. These are safe defaults.

**4. Configure ef at query time.**
Start at ef=128. Measure recall@10. If below 0.95, increase ef. If latency is too high, decrease ef or reduce M.

**5. Add scalar quantization if memory is a concern.**
INT8 quantization gives 4× memory reduction with < 1% recall loss. Enable `always_ram=True` for quantized vectors to keep them in memory.

**6. Create payload indexes for every metadata field you filter on.**
Unindexed filters = full scans = slow.

**7. Set a recall measurement job to run daily.**
Corpora grow and distributions shift. What worked at 10K vectors may not work at 500K.

---

## Summary

- Exact nearest neighbor search is O(N × D) — too slow for large corpora. ANN algorithms trade a small accuracy loss for orders-of-magnitude speed improvement.
- HNSW builds a hierarchical navigable graph. Key parameters: M (connections), ef_construction (build quality), ef (query time recall/speed trade-off). Best for dynamic corpora with frequent updates.
- IVF clusters the vector space and searches only the nearest clusters. Key parameters: nlist (clusters), nprobe (clusters searched at query time). Best for large static corpora.
- Product Quantization compresses vectors 32–64× by replacing sub-vectors with cluster assignments. Combines with IVF as IVFPQ for very large corpora.
- Scalar quantization (INT8) gives 4× compression with minimal accuracy loss — a good first step before PQ.
- Metadata filtering with small filter selectivity degrades ANN recall. Solutions: partition collections, use payload indexes, or increase ef for filtered queries.
- Always measure recall@K and P99 latency in production. Never assume the index is performing well just because queries complete.

---

## What's Next

Lesson 3.2 covers sparse retrieval in depth — BM25 internals, SPLADE and learned sparse models, and how to implement the keyword search side of hybrid retrieval.