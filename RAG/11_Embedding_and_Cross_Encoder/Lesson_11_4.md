# Lesson A.4 — Matryoshka Representation Learning: Internals and Production Use

---

## The Problem: Fixed-Dimension Embeddings Are Inflexible

Standard embedding models produce a fixed-dimension vector — always 768 or always 1536 dimensions. This creates a rigid trade-off at deployment time:

- Use full dimensions → maximum quality, maximum storage, slower search.
- Switch to a smaller model → lower quality, less storage, faster search.

There is no middle ground. If you need to reduce costs by cutting storage from 1536 to 256 dimensions, you cannot just truncate the vector — the last 1280 dimensions of a standard model contain meaningful information that is interleaved throughout the full vector. Truncation destroys quality.

This is the problem Matryoshka Representation Learning (MRL, Kusupati et al., 2022) solves. MRL trains a model so that the first M dimensions of any vector are a complete, high-quality representation at dimension M — enabling arbitrary truncation without catastrophic quality loss.

The name comes from Russian Matryoshka nesting dolls: each smaller doll is complete in itself, nested inside larger ones. Similarly, MRL embeddings contain a complete 64-dim representation inside a 128-dim representation, inside a 256-dim representation, and so on up to the full dimension.

---

## How Standard Embedding Training Distributes Information

In a standard embedding model, information is distributed across all dimensions by the training process. There is no ordering to the dimensions — dimension 1 is not more important than dimension 768. The model learns to use all dimensions to minimize its loss, and different dimensions encode different aspects of meaning.

If you take a 1536-dim vector and keep only the first 64 dimensions, you are keeping an arbitrary subset of the model's representational capacity. Some important information might be in dimension 800. Some might be in dimension 1400. The truncated 64-dim vector is garbage — it does not represent the input well because it has access to only a random fraction of the model's learned features.

---

## How MRL Forces Information Hierarchy

MRL solves this with a modified training objective that explicitly encourages the model to pack the most important information into the earliest dimensions, with later dimensions providing refinement.

The key mechanism: **multi-scale loss**.

During training, MRL computes the contrastive loss at multiple dimension levels simultaneously and adds them all together:

```
Total Loss = λ₁ × Loss(first 64 dims) + λ₂ × Loss(first 128 dims) + λ₃ × Loss(first 256 dims) + λ₄ × Loss(first 512 dims) + λ₅ × Loss(full 1536 dims)
```

Where `λ₁ > λ₂ > λ₃ > λ₄ > λ₅` (earlier dimensions get higher weight in the loss).

```python
import torch
import torch.nn.functional as F

class MatryoshkaLoss(torch.nn.Module):
    def __init__(
        self,
        dimensions: list[int],  # [64, 128, 256, 512, 1536]
        weights: list[float] = None
    ):
        super().__init__()
        self.dimensions = sorted(dimensions)
        
        if weights is None:
            # Default: higher weight for smaller dimensions
            # Forces the model to make small dimensions useful
            self.weights = [1.0 / d for d in self.dimensions]
        else:
            self.weights = weights
        
        # Normalize weights to sum to 1
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def info_nce_at_dim(
        self,
        embeddings: torch.Tensor,  # (2N, D) — first N are anchors, last N are positives
        dim: int,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute InfoNCE loss using only the first `dim` dimensions."""
        
        # Truncate to first `dim` dimensions
        truncated = embeddings[:, :dim]
        
        # Normalize the truncated vectors
        truncated = F.normalize(truncated, dim=-1)
        
        N = truncated.shape[0] // 2
        anchors = truncated[:N]    # First N
        positives = truncated[N:]  # Second N
        
        # Similarity matrix
        sim_matrix = torch.matmul(anchors, positives.T) / temperature
        
        # Labels: diagonal is the true positive
        labels = torch.arange(N, device=truncated.device)
        
        return F.cross_entropy(sim_matrix, labels)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for dim, weight in zip(self.dimensions, self.weights):
            loss_at_dim = self.info_nce_at_dim(embeddings, dim)
            total_loss += weight * loss_at_dim
        
        return total_loss
```

---

## What the Multi-Scale Loss Forces the Model to Do

Here is the key insight. Consider the gradient flow during training.

For the 64-dim loss term to decrease, the model must make the first 64 dimensions sufficient to rank the true positive above all negatives. This is a strong constraint — 64 dimensions must capture the most essential semantic content.

For the 128-dim loss term to decrease, dimensions 65-128 must add additional discriminative information beyond what 64 dimensions capture.

For the 256-dim loss term to decrease, dimensions 129-256 must add more refinement still.

And so on. Each scale compels the model to:
1. Front-load the most important semantic information into early dimensions.
2. Add progressively finer distinctions in later dimensions.

The result: **the first M dimensions of a Matryoshka vector are the best possible M-dimensional representation of that text** — not a random subset, but the information-theoretically optimal first M dimensions.

This is fundamentally different from taking a standard 1536-dim model and keeping only the first 64 dims. The standard model was never trained to make the first 64 dims sufficient. The Matryoshka model was explicitly trained to do so.

---

## Information Distribution: What Each Dimension Band Captures

After training, what does each dimension range encode?

Empirically and theoretically, the pattern is:

- **First 64-128 dims:** Coarse semantic content. Topic, domain, general intent. "This is about parental leave policy" vs. "This is about financial regulations."
- **Dims 129-256:** More specific content. Within parental leave, distinguishing eligibility vs. duration vs. application process.
- **Dims 257-512:** Fine-grained distinctions. Specific conditions, exceptions, precise phrasing differences.
- **Dims 513-1536:** Very subtle distinctions. Stylistic differences, nuanced topic variants, rare vocabulary.

This matches the intuition: for rough retrieval (finding the right topic area), 64-128 dims is sufficient. For precise retrieval (finding the exact relevant clause), you need more dimensions.

---

## The Similarity Score Relationship Across Dimensions

An important property: the similarity score computed at a smaller dimension is a noisy but directionally correct version of the full-dimension similarity score.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# nomic-embed-text-v1.5 is a Matryoshka model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

texts = [
    "The employee parental leave policy allows 16 weeks of paid leave",
    "Workers are entitled to 4 months of fully compensated family leave",  # Semantically similar
    "The quarterly revenue report shows 15% growth",  # Unrelated
]

# Full 768-dim embeddings
full_embeddings = model.encode(texts, normalize_embeddings=True)

# Truncated and re-normalized embeddings
def truncate_and_normalize(embs: np.ndarray, dim: int) -> np.ndarray:
    truncated = embs[:, :dim]
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    return truncated / norms

for dim in [64, 128, 256, 512, 768]:
    truncated = truncate_and_normalize(full_embeddings, dim)
    sim_similar = float(np.dot(truncated[0], truncated[1]))
    sim_unrelated = float(np.dot(truncated[0], truncated[2]))
    
    print(f"Dim {dim}: similar={sim_similar:.3f}, unrelated={sim_unrelated:.3f}, separation={sim_similar - sim_unrelated:.3f}")

# Expected output pattern:
# Dim 64:  similar=0.72, unrelated=0.31, separation=0.41
# Dim 128: similar=0.78, unrelated=0.22, separation=0.56
# Dim 256: similar=0.83, unrelated=0.18, separation=0.65
# Dim 512: similar=0.87, unrelated=0.14, separation=0.73
# Dim 768: similar=0.89, unrelated=0.12, separation=0.77
```

The pattern: at every dimension level, the model correctly identifies that the similar pair is more similar than the unrelated pair. Larger dimensions increase the margin (separation). This is the property that makes Matryoshka useful for tiered retrieval.

---

## Production Patterns: Tiered Retrieval

The primary production use case for Matryoshka embeddings is tiered retrieval — using small dimensions for fast candidate selection and large dimensions for precise re-ranking.

### Pattern 1 — Two-Stage with Dimension Escalation

```python
class MatryoshkaRetriever:
    def __init__(self, vector_db, embedding_model):
        self.vdb = vector_db
        self.model = embedding_model
    
    async def retrieve(
        self,
        query: str,
        first_stage_dim: int = 128,
        second_stage_dim: int = 768,
        first_stage_k: int = 100,
        final_k: int = 10
    ) -> list[dict]:
        """
        Two-stage retrieval exploiting Matryoshka dimension flexibility.
        """
        
        # Embed query at full dimension (we need both in one pass)
        full_query_emb = self.model.encode(query, normalize_embeddings=True)
        
        # Stage 1: Fast retrieval at small dimension
        # Truncate and re-normalize for 128-dim search
        small_query_emb = self._truncate_normalize(full_query_emb, first_stage_dim)
        
        candidates = await self.vdb.search(
            collection="docs_128dim",  # Index built at 128 dims
            query_vector=small_query_emb.tolist(),
            limit=first_stage_k
        )
        
        candidate_ids = [c.id for c in candidates]
        
        # Stage 2: Precise re-ranking at full dimension
        # Fetch full-dim embeddings for candidates
        full_dim_embeddings = await self.vdb.get_vectors(
            collection="docs_768dim",
            ids=candidate_ids
        )
        
        # Compute precise similarity at full dimension
        large_query_emb = self._truncate_normalize(full_query_emb, second_stage_dim)
        
        precise_scores = [
            (cid, float(np.dot(large_query_emb, emb)))
            for cid, emb in full_dim_embeddings.items()
        ]
        
        precise_scores.sort(key=lambda x: x[1], reverse=True)
        
        return precise_scores[:final_k]
    
    def _truncate_normalize(self, embedding: np.ndarray, dim: int) -> np.ndarray:
        truncated = embedding[:dim]
        norm = np.linalg.norm(truncated)
        return truncated / norm if norm > 0 else truncated
```

### Pattern 2 — Storage-Cost Reduction with Quality Trade-off Analysis

```python
def analyze_matryoshka_quality_tradeoff(
    model,
    eval_queries: list[str],
    eval_relevant_chunks: list[list[str]],
    corpus_chunks: list[str],
    dimensions: list[int] = [64, 128, 256, 512, 768, 1536]
) -> dict:
    """
    Measure recall@10 at each dimension to find the 
    minimum dimension that meets your quality threshold.
    """
    import numpy as np
    
    results = {}
    
    # Embed everything at full dimension once
    all_texts = eval_queries + corpus_chunks
    all_embeddings = model.encode(all_texts, normalize_embeddings=True)
    
    query_embeddings = all_embeddings[:len(eval_queries)]
    corpus_embeddings = all_embeddings[len(eval_queries):]
    
    for dim in dimensions:
        # Truncate and re-normalize
        query_embs_dim = np.array([
            e[:dim] / np.linalg.norm(e[:dim]) for e in query_embeddings
        ])
        corpus_embs_dim = np.array([
            e[:dim] / np.linalg.norm(e[:dim]) for e in corpus_embeddings
        ])
        
        # Compute retrieval metrics
        recalls = []
        for i, (query_emb, relevant) in enumerate(zip(query_embs_dim, eval_relevant_chunks)):
            similarities = corpus_embs_dim @ query_emb
            top_10_indices = np.argsort(similarities)[-10:][::-1]
            retrieved = set(corpus_chunks[j] for j in top_10_indices)
            relevant_set = set(relevant)
            recall = len(retrieved & relevant_set) / len(relevant_set)
            recalls.append(recall)
        
        storage_per_million = dim * 4 / 1e6  # MB per million vectors (float32)
        
        results[dim] = {
            "recall@10": float(np.mean(recalls)),
            "storage_per_1M_vectors_MB": storage_per_million * 1000,  # GB for 1M vectors
            "relative_to_full": float(np.mean(recalls)) / results.get(max(dimensions), {}).get("recall@10", 1)
        }
    
    return results

# Example output interpretation:
# Dim 64:  recall@10 = 0.72 (storage: 0.24 GB/1M vectors) — 81% of full quality
# Dim 128: recall@10 = 0.79 (storage: 0.49 GB/1M vectors) — 89% of full quality  
# Dim 256: recall@10 = 0.84 (storage: 0.98 GB/1M vectors) — 94% of full quality
# Dim 512: recall@10 = 0.87 (storage: 1.95 GB/1M vectors) — 98% of full quality
# Full 768: recall@10 = 0.89 (storage: 2.93 GB/1M vectors) — 100%
# 
# Decision: Dim 256 gives 94% of quality at 33% of storage. Good trade-off.
```

### Pattern 3 — OpenAI API Dimension Parameter

OpenAI's `text-embedding-3-large` and `text-embedding-3-small` are Matryoshka models that support native dimension truncation via the API:

```python
from openai import OpenAI

client = OpenAI()

def embed_with_openai(text: str, dimensions: int = 1536) -> list[float]:
    """
    OpenAI Matryoshka embedding with specified dimensions.
    The API handles truncation and normalization internally.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dimensions  # Matryoshka truncation at API level
    )
    return response.data[0].embedding

# Fast, cheap first-stage retrieval
small_embedding = embed_with_openai(query, dimensions=256)

# Precise second-stage re-ranking
full_embedding = embed_with_openai(query, dimensions=3072)

# Cost note: OpenAI charges the same regardless of dimensions parameter
# The benefit is storage and search speed, not API cost
```

---

## When Matryoshka Does NOT Help

Matryoshka is not a universal win. Understand when it helps and when it does not:

**Helps when:**
- Storage cost is a constraint (millions or billions of vectors).
- You want tiered retrieval without maintaining two separate models.
- You need flexibility to adjust the speed-quality trade-off post-deployment.

**Does NOT help when:**
- You have plenty of storage and latency is not a concern.
- You need the absolute best quality at all costs — full-dimension non-Matryoshka models trained with that specific objective may slightly outperform Matryoshka at the same dimension.
- Your corpus is small (< 100K vectors) — the speed difference is negligible.

**The quality cost of Matryoshka:**

A Matryoshka model trained to be good at 64, 128, 256, 512, and 1536 dimensions must distribute its learning objective across all of these. A standard model trained only for 1536 dimensions can focus all its capacity on making 1536-dim representations perfect. At the full dimension, a well-trained Matryoshka model is typically within 1-2% recall of its non-Matryoshka equivalent — acceptable for most use cases.

---

## Summary

- Standard embedding models distribute information uniformly across all dimensions. Truncation destroys quality because important information is in non-first dimensions.
- MRL trains with a multi-scale loss that explicitly forces the model to pack the most important semantic information into the earliest dimensions.
- The first M dimensions of a Matryoshka vector are the best possible M-dimensional representation — not a random subset.
- Earlier dimensions capture coarse semantic content; later dimensions add fine-grained distinctions.
- Production patterns: two-stage tiered retrieval (small dims for candidates, full dims for re-ranking), storage reduction with measured quality trade-off, OpenAI's native dimensions API parameter.
- Quality cost at full dimension is typically 1-2% recall — acceptable for most use cases in exchange for the flexibility gained.

---

## What's Next

Lesson A.5 covers cross-encoder architecture and training in depth — how the joint encoding mechanism works mechanically, the three types of training loss (pointwise, pairwise, listwise), and why cross-encoders are fundamentally more accurate than bi-encoders for relevance scoring.