# Lesson 3.2 — Sparse Retrieval: BM25, SPLADE, and Learned Sparse Models

---

## What Sparse Retrieval Is and Why It Still Matters

Sparse retrieval predates neural embeddings by decades. BM25, the dominant sparse retrieval algorithm, was formalized in the 1990s. Yet it remains a critical component in state-of-the-art RAG systems in 2024–2025. The reason is simple: sparse and dense retrieval fail on completely different types of queries, and you need both.

"Sparse" refers to the representation. A sparse vector has most of its values as zero — only the dimensions corresponding to terms that actually appear in the text have non-zero values. A vocabulary of 100,000 terms produces a 100,000-dimensional vector, but a typical document chunk uses only 50–200 unique terms, so 99.9%+ of dimensions are zero.

Dense vectors (from embedding models) are the opposite — every dimension has a non-zero value encoding some aspect of semantic meaning.

The practical implication: sparse retrieval is exact keyword matching with statistical weighting. Dense retrieval is semantic similarity. They are not competing alternatives — they are complementary tools.

---

## BM25 — The Algorithm Behind Keyword Retrieval

BM25 stands for Best Match 25. It is the 25th iteration of a family of probabilistic retrieval models developed at the University of London through the 1970s–90s. It is still the default retrieval function in Elasticsearch, OpenSearch, Solr, and most search engines.

### The Intuition

BM25 answers: "How relevant is document D to query Q, based on the words they share?"

Three factors determine the score:

**1. Term Frequency (TF):** If the query term "refund" appears 5 times in a document, that document is probably more about refunds than a document where it appears once. But the benefit of additional occurrences diminishes — 10 occurrences is not 10× better than 1 occurrence. BM25 uses a saturating term frequency.

**2. Inverse Document Frequency (IDF):** The word "the" appears in almost every document — it carries no information. The word "indemnification" appears in very few documents — finding it is significant. IDF weights rare terms higher and common terms lower.

**3. Document Length Normalization:** A 10,000-word document will naturally contain the query term more times than a 200-word document just by virtue of being longer. BM25 normalizes for document length so longer documents do not get an unfair advantage.

### The Formula

For query Q containing terms q₁, q₂, ..., qₙ and document D:

```
BM25(D, Q) = Σᵢ IDF(qᵢ) × [TF(qᵢ, D) × (k1 + 1)] / [TF(qᵢ, D) + k1 × (1 - b + b × |D|/avgdl)]
```

Where:
- `TF(qᵢ, D)` = frequency of term qᵢ in document D
- `|D|` = length of document D (in terms)
- `avgdl` = average document length across the corpus
- `k1` = term frequency saturation parameter (typically 1.2–2.0)
- `b` = length normalization parameter (typically 0.75)
- `IDF(qᵢ)` = log((N - df(qᵢ) + 0.5) / (df(qᵢ) + 0.5) + 1)
  - N = total number of documents
  - df(qᵢ) = number of documents containing term qᵢ

### Parameter Intuition

**k1 (term frequency saturation):**
- Controls how quickly the benefit of additional term occurrences diminishes.
- k1 = 0: term frequency is completely ignored (just IDF matching).
- k1 = large: term frequency has increasing importance, less saturation.
- k1 = 1.2–2.0 works well for most text. For shorter chunks (RAG chunks are shorter than full documents), k1 = 1.2 is often better — chunks are short enough that raw frequency difference is meaningful.

**b (length normalization):**
- b = 0: no length normalization (longer chunks always win on frequency).
- b = 1: full length normalization (scores normalized to chunk length).
- b = 0.75 is the standard. For RAG where chunk sizes are relatively consistent, lower b (0.5–0.6) is sometimes better — you want a chunk with high term frequency to win even if it is slightly longer.

```python
from rank_bm25 import BM25Okapi
import nltk

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """Build a BM25 index over a list of text chunks."""
    
    # Tokenize each chunk
    tokenized_chunks = [
        simple_tokenize(chunk) for chunk in chunks
    ]
    
    # Build BM25 index (rank_bm25 uses BM25Okapi which is the standard variant)
    return BM25Okapi(tokenized_chunks, k1=1.5, b=0.75)

def simple_tokenize(text: str) -> list[str]:
    """Basic tokenization: lowercase, remove punctuation, split on whitespace."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    tokens = text.split()
    return tokens

def bm25_search(bm25_index: BM25Okapi, query: str, k: int = 20) -> list[tuple[int, float]]:
    """Search BM25 index. Returns (chunk_index, score) pairs."""
    query_tokens = simple_tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    
    # Return top-k (index, score) pairs
    top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return top_k
```

### Tokenization Matters

BM25 operates on tokens, not characters. The quality of your tokenizer significantly affects BM25 quality.

**Minimal tokenizer (lowercase + split):** Fast, simple. Misses stemming ("running" and "runs" are different tokens) and does not handle compound words.

**Stemming tokenizer:** Reduces words to their root form. "running" → "run", "policies" → "polic". Increases recall but reduces precision (all stemmed variants are treated as the same term).

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def stemming_tokenize(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    # Filter stopwords and stem
    stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'is', 'are', 'was', 'were'}
    stemmed = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stopwords]
    return stemmed
```

**Stopword removal:** Removing common words ("the", "is", "at") reduces index size and improves IDF weighting of meaningful terms. But be careful — in some domains, words that look like stopwords carry meaning. "not" is critical for legal negation.

**For domain-specific content:**
- Medical: preserve compound terms ("blood pressure", "type 2 diabetes") — split into individual tokens loses the compound meaning.
- Legal: preserve exact phrases ("force majeure", "indemnify and hold harmless").
- Code: camelCase splitting ("getUserById" → ["get", "user", "by", "id"]) improves retrieval.

### What BM25 Cannot Do

BM25 has no understanding of meaning. It works purely on token overlap.

- "heart attack" and "myocardial infarction" → zero BM25 similarity (no shared tokens).
- "the policy was not approved" and "the policy was approved" → BM25 may give them similar scores (they share most tokens, "not" is often a stopword).
- Synonyms, paraphrases, translations → all invisible to BM25.

This is precisely why BM25 alone is insufficient and must be combined with dense retrieval.

---

## BM25 at Scale: Inverted Indexes

The rank_bm25 library works fine for tens of thousands of chunks. At millions of chunks, you need a proper inverted index.

### The Inverted Index Structure

An inverted index maps each term to a list of documents (and positions) containing that term:

```
"refund"       → [(doc_42, tf=3), (doc_891, tf=1), (doc_2301, tf=5), ...]
"policy"       → [(doc_7, tf=2), (doc_42, tf=4), (doc_156, tf=1), ...]
"termination"  → [(doc_156, tf=2), (doc_891, tf=3), ...]
```

At query time, for each query term, fetch its posting list, compute BM25 contribution, merge across terms. Only documents containing at least one query term are scored — the rest of the corpus is never touched.

This is why BM25 search is fast: it processes only the fraction of the corpus that shares vocabulary with the query.

**Production inverted index options:**

**Elasticsearch / OpenSearch:** Battle-tested, distributed, supports BM25 natively (it is the default). Handles billions of documents. Supports filtering alongside BM25 search. Good choice when you need both keyword search and can tolerate the operational overhead of running an Elasticsearch cluster.

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# Index a chunk
es.index(
    index="rag_chunks",
    id=chunk_id,
    document={
        "text": chunk_text,
        "doc_id": doc_id,
        "section": section,
        "effective_date": effective_date
    }
)

# BM25 search with metadata filter
response = es.search(
    index="rag_chunks",
    body={
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "text": {
                            "query": query_text,
                            "operator": "or"  # match any query term
                        }
                    }
                },
                "filter": [
                    {"term": {"document_status": "active"}},
                    {"range": {"effective_date": {"gte": "2024-01-01"}}}
                ]
            }
        },
        "size": 20
    }
)
```

**Qdrant sparse vectors:** Qdrant supports storing sparse vectors (BM25 or SPLADE sparse representations) alongside dense vectors in the same collection, enabling true single-system hybrid search without running a separate Elasticsearch cluster.

```python
from qdrant_client.models import SparseVector, NamedSparseVector

# Encode chunk as sparse vector
def text_to_sparse_vector(text: str, bm25_vocab: dict) -> SparseVector:
    """Convert text to a sparse vector using BM25 term weights."""
    tokens = simple_tokenize(text)
    term_counts = {}
    for token in tokens:
        if token in bm25_vocab:
            term_counts[token] = term_counts.get(token, 0) + 1
    
    indices = [bm25_vocab[term] for term in term_counts]
    values = [float(count) for count in term_counts.values()]
    
    return SparseVector(indices=indices, values=values)

# Upsert with both dense and sparse vectors
client.upsert(
    collection_name="documents",
    points=[PointStruct(
        id=chunk_id,
        vector={
            "dense": dense_embedding,      # named dense vector
            "sparse": sparse_vector        # named sparse vector
        },
        payload=chunk_metadata
    )]
)

# Hybrid search using both
client.search(
    collection_name="documents",
    query_vector=NamedVector(name="dense", vector=query_dense_embedding),
    query_sparse_vector=NamedSparseVector(name="sparse", vector=query_sparse_vector),
    limit=20
)
```

---

## SPLADE — Learned Sparse Models

BM25 is fast and interpretable but has no semantic understanding. Dense models have semantic understanding but no exact keyword matching. SPLADE sits in between: it uses a neural model to produce sparse vectors that combine the advantages of both.

### What SPLADE Does

SPLADE (SParse Lexical AnD Expansion) uses a BERT-based model to:

1. **Expand terms:** Add terms that are semantically related to the text even if they do not appear in it. A chunk about "myocardial infarction" gets "heart" and "attack" added to its sparse vector — terms a user might search for.

2. **Weight terms:** Unlike BM25 (which uses frequency-based weights), SPLADE learns neural weights for each term that better reflect its importance for retrieval.

The result is a sparse vector over the full vocabulary where:
- High values indicate terms that are important and present (or semantically implied).
- Zero values indicate terms that are irrelevant.
- Most values are zero (sparse) — typically 99%+ of vocabulary dimensions are zero.

### How SPLADE Vectors Look

For the text "The patient was diagnosed with myocardial infarction":

**BM25 sparse vector (simplified):**
```
patient: 0.82
diagnosed: 0.45
myocardial: 1.23
infarction: 1.18
```
(Only terms present in text, weighted by TF-IDF)

**SPLADE sparse vector (simplified):**
```
patient: 0.91
diagnosed: 0.52
myocardial: 1.31
infarction: 1.25
heart: 0.78        ← expanded (not in original text)
cardiac: 0.71      ← expanded
attack: 0.63       ← expanded
coronary: 0.55     ← expanded
disease: 0.41      ← expanded
treatment: 0.38    ← expanded
```

The expansion terms allow a user query "heart attack treatment" to match the chunk about "myocardial infarction" through the sparse vector — something BM25 cannot do.

### SPLADE vs. Dense Embeddings

Both SPLADE and dense models understand semantics. The difference is in how they represent and retrieve:

| | SPLADE | Dense Embeddings |
|---|---|---|
| **Representation** | Sparse over vocabulary | Dense over latent dimensions |
| **Interpretability** | High (terms are human-readable) | Low (dimensions are latent) |
| **Exact term matching** | Yes (terms are preserved) | No |
| **Semantic expansion** | Yes (via learned expansion) | Yes (via latent space) |
| **Storage** | Sparse (compressed) | Dense (full vectors) |
| **Index structure** | Inverted index | ANN graph |
| **Query speed** | Very fast (inverted index lookup) | Fast (ANN search) |

SPLADE tends to outperform BM25 significantly. It often matches or exceeds dense retrieval on benchmarks, especially for English-language retrieval tasks with keyword-heavy queries.

### Using SPLADE in Practice

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class SPLADEEncoder:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> dict[int, float]:
        """
        Returns a sparse vector as {token_id: weight} dict.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # SPLADE aggregation: max pooling over tokens, then ReLU and log
        # This produces the sparse representation
        aggregated = torch.max(
            torch.log(1 + torch.relu(logits)) * inputs['attention_mask'].unsqueeze(-1),
            dim=1
        ).values.squeeze()
        
        # Convert to sparse dict (only non-zero terms)
        sparse_dict = {}
        nonzero_indices = aggregated.nonzero().squeeze()
        
        for idx in nonzero_indices:
            idx = idx.item()
            weight = aggregated[idx].item()
            if weight > 0:
                sparse_dict[idx] = weight
        
        return sparse_dict
    
    def decode_sparse_vector(self, sparse_dict: dict[int, float], 
                              top_k: int = 20) -> dict[str, float]:
        """Convert token IDs to readable terms."""
        top_items = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {
            self.tokenizer.decode([token_id]): weight
            for token_id, weight in top_items
        }

# Usage
encoder = SPLADEEncoder()

# Encode a chunk
chunk_sparse = encoder.encode("The employee is entitled to 16 weeks parental leave")
# Returns: {token_id: weight, ...}

# Encode a query  
query_sparse = encoder.encode("maternity leave duration")
# Returns: {token_id: weight, ...}

# Compute similarity (dot product of sparse vectors)
def sparse_dot_product(vec1: dict, vec2: dict) -> float:
    shared_keys = set(vec1.keys()) & set(vec2.keys())
    return sum(vec1[k] * vec2[k] for k in shared_keys)

similarity = sparse_dot_product(chunk_sparse, query_sparse)
```

### SPLADE Variants

**SPLADE-v2:** Improved training, better balance between sparsity and retrieval quality.

**SPLADE-CoCondenser:** Fine-tuned on MS MARCO with co-condenser pretraining. Best general performance.

**SPLADEv3:** Latest variant with improved efficiency and quality.

**DistilSPLADE:** Smaller, faster model with acceptable quality for latency-sensitive applications.

The model choice follows the same principle as dense embedding models: evaluate on your domain. SPLADE models trained on MS MARCO (web search queries + web documents) may not perform optimally on legal documents or medical records.

---

## When to Use BM25 vs. SPLADE

| Situation | Recommendation |
|---|---|
| General-purpose RAG | BM25 as the sparse component of hybrid search |
| High-quality retrieval requirement | SPLADE instead of BM25 |
| Already running Elasticsearch | BM25 (built-in, no extra infrastructure) |
| Using Qdrant for vector DB | Qdrant sparse vectors — BM25 or SPLADE both work |
| Domain with lots of exact codes/IDs | BM25 (SPLADE may expand codes incorrectly) |
| Domain with synonym-heavy queries | SPLADE (handles expansion better than BM25) |
| Latency critical | BM25 (simpler computation, faster) |

In most production RAG systems, BM25 is the right sparse component for hybrid retrieval. SPLADE makes sense when you have evaluated retrieval quality and found BM25 insufficient, and you can absorb the compute overhead of neural sparse encoding.

---

## Building a Complete Hybrid Retrieval System

Putting BM25 and dense retrieval together into a production-ready hybrid system:

```python
class HybridRetriever:
    def __init__(
        self,
        vector_db_client,
        bm25_index: BM25Okapi,
        chunk_id_list: list[str],   # maps BM25 result index to chunk_id
        embedding_model,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.vector_db = vector_db_client
        self.bm25 = bm25_index
        self.chunk_ids = chunk_id_list
        self.embedder = embedding_model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    async def retrieve(
        self,
        query: str,
        k_dense: int = 50,
        k_sparse: int = 50,
        k_final: int = 20,
        metadata_filter: dict = None
    ) -> list[dict]:
        
        # Run dense and sparse retrieval in parallel
        dense_task = self._dense_retrieve(query, k_dense, metadata_filter)
        sparse_task = self._sparse_retrieve(query, k_sparse)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Merge using RRF (score-agnostic, no need to normalize)
        merged = self._reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=60  # RRF constant
        )
        
        return merged[:k_final]
    
    async def _dense_retrieve(self, query: str, k: int, 
                               metadata_filter: dict) -> list[dict]:
        query_embedding = await self.embedder.embed(query)
        
        results = await self.vector_db.search(
            collection="documents",
            query_vector=query_embedding,
            filter=metadata_filter,
            limit=k
        )
        
        return [{"chunk_id": r.id, "score": r.score, "payload": r.payload}
                for r in results]
    
    async def _sparse_retrieve(self, query: str, k: int) -> list[dict]:
        top_k_indices = self.bm25.get_top_n(
            simple_tokenize(query),
            list(range(len(self.chunk_ids))),
            n=k
        )
        
        return [{"chunk_id": self.chunk_ids[idx], "score": score}
                for idx, score in top_k_indices]
    
    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[dict]],
        k: int = 60
    ) -> list[dict]:
        scores = {}
        
        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                chunk_id = result["chunk_id"]
                if chunk_id not in scores:
                    scores[chunk_id] = {"score": 0.0, "payload": result.get("payload")}
                scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
        
        merged = [
            {"chunk_id": cid, "rrf_score": data["score"], "payload": data["payload"]}
            for cid, data in scores.items()
        ]
        
        return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)
```

---

## Summary

- BM25 is a probabilistic keyword retrieval algorithm with three components: term frequency (saturating), inverse document frequency (rare terms weighted higher), and document length normalization.
- Key BM25 parameters: k1 controls TF saturation (1.2–2.0), b controls length normalization (0.75 standard, lower for consistent-length chunks).
- Tokenization quality directly affects BM25 quality. Domain-specific tokenization (stemming, compound term preservation, stopword tuning) is worth investing in.
- At scale, BM25 requires an inverted index — Elasticsearch/OpenSearch for existing infrastructure, or Qdrant sparse vectors for a single-system solution.
- SPLADE uses a neural model to produce learned sparse vectors with term expansion. It understands semantics while maintaining sparse, interpretable representations. Outperforms BM25 but has higher compute cost.
- BM25 is the right default sparse component for most production RAG systems. SPLADE is worth evaluating when BM25 quality is insufficient and you need synonym handling.
- Hybrid retrieval runs dense and sparse search in parallel and merges with RRF. Parallelism is important — do not run them sequentially.

---

## What's Next

Lesson 3.3 covers hybrid search design in depth — how dense and sparse retrieval complement each other at the failure mode level, RRF vs. score-based fusion in detail, and how to tune the balance between them.