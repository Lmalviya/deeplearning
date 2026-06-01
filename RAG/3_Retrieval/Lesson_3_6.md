# Lesson 3.6 — Re-ranking: Cross-Encoders, LLM-as-Reranker, and Choosing the Right Strategy

---

## Why Retrieval Ranking Is Coarse

After hybrid retrieval and RRF fusion, you have a ranked list of candidate chunks — say, the top 50. The question is: is this the right ordering for the LLM to consume?

The answer is almost always no, for a specific reason. Every fast retrieval mechanism — ANN vector search, BM25, RRF — optimizes for speed by making architectural compromises that sacrifice ranking precision. They are designed to get the right chunks into the candidate set, not to order them perfectly.

- ANN search finds approximate nearest neighbors, not exact ones.
- BM25 scores on term statistics, not on whether the chunk actually answers the question.
- RRF merges ranks from multiple lists without understanding the content.

None of these mechanisms can answer: "Given this specific query, which of these 50 chunks most directly and completely answers it?" That question requires reading both the query and the chunk together — joint reasoning that fast retrieval systems explicitly avoid because it does not scale.

Re-ranking is the stage that does this joint reasoning. It takes the coarse candidate list from retrieval and produces a precise final ordering. The key insight is that re-ranking only needs to score 20–100 candidates, not millions — so it can afford to be much more computationally expensive per comparison.

---

## Cross-Encoder Architecture

A cross-encoder is a neural network that takes a (query, chunk) pair as a single input and outputs a relevance score. Unlike a bi-encoder (which encodes query and chunk independently), the cross-encoder sees both texts simultaneously and can model their interactions at the token level.

### How Cross-Encoders Work

```
Bi-encoder (retrieval):
  Query: "maternity leave eligibility"  → Encoder → q_vector [0.23, -0.11, ...]
  Chunk: "Employees are eligible..."    → Encoder → d_vector [0.19, -0.08, ...]
  Similarity: cosine(q_vector, d_vector) = 0.87

Cross-encoder (re-ranking):
  Input: [CLS] maternity leave eligibility [SEP] Employees are eligible... [SEP]
         ↓
  Transformer (all tokens attend to all other tokens)
         ↓
  Score: 0.94  ← single relevance score
```

In the cross-encoder, every query token attends to every chunk token through the transformer's self-attention mechanism. The model learns to recognize patterns like: "this chunk answers the question" vs. "this chunk mentions related terms but does not answer the question."

This is qualitatively more powerful than cosine similarity between independent embeddings. The cross-encoder can reason: "the query asks about eligibility, the chunk defines eligibility criteria, therefore highly relevant" — even if the specific words do not overlap much.

The cost is that you cannot pre-compute document representations. Every (query, chunk) pair requires a fresh forward pass. For 50 candidates, that is 50 forward passes. For millions of documents, that is infeasible — hence the two-stage pipeline.

### Implementation

```python
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int = 10
    ) -> list[dict]:
        """
        Re-rank candidates using cross-encoder scores.
        Returns top_k candidates in re-ranked order.
        """
        if not candidates:
            return []
        
        # Build (query, chunk_text) pairs
        pairs = [(query, c[text_key]) for c in candidates]
        
        # Score all pairs in one batch forward pass
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )
        
        # Attach scores to candidates
        scored = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        
        # Sort by re-rank score (descending)
        reranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]
```

### Choosing a Cross-Encoder Model

Cross-encoder models vary significantly in speed and quality. The trade-off is direct: larger models are more accurate but slower.

**Fast, good quality (production default):**

`cross-encoder/ms-marco-MiniLM-L-6-v2`
- 6-layer MiniLM, very fast (~5ms per 32 pairs on GPU, ~50ms on CPU)
- Trained on MS MARCO (web search relevance)
- Good general performance
- Best for latency-sensitive applications

`cross-encoder/ms-marco-MiniLM-L-12-v2`
- 12-layer, roughly 2× slower than L-6
- Meaningfully better accuracy than L-6
- Good default when you have moderate latency budget

**Slower, higher quality:**

`cross-encoder/ms-marco-electra-base`
- ELECTRA-based, stronger than MiniLM at higher latency
- Good for offline re-ranking or high-value queries

`BAAI/bge-reranker-large`
- Strong multilingual and domain performance
- Good for non-English or domain-specific corpora

**Managed API (no infrastructure):**

Cohere Rerank API (`rerank-english-v3.0`, `rerank-multilingual-v3.0`)
- Excellent quality
- Latency: ~100–300ms per re-ranking call depending on candidate count
- No GPU infrastructure needed
- Per-call cost at scale

```python
import cohere

co = cohere.Client(api_key="your-api-key")

def cohere_rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "text",
    top_k: int = 10
) -> list[dict]:
    
    documents = [c[text_key] for c in candidates]
    
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )
    
    reranked = []
    for result in response.results:
        candidate = candidates[result.index]
        reranked.append({
            **candidate,
            "rerank_score": result.relevance_score
        })
    
    return reranked
```

### Handling the 512-Token Limit

Most cross-encoder models have a 512-token input limit for the concatenated (query + chunk) pair. If your chunks are 500 tokens and your query is 50 tokens, you will exceed this limit.

Strategies:
- **Truncate the chunk:** Cut the chunk to fit within the token budget. The cross-encoder sees only the first N tokens of the chunk. Information in the truncated portion is invisible.
- **Use a long-context cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` supports up to 512 tokens, but newer models support longer inputs.
- **Use parent-child retrieval to pass smaller child chunks for re-ranking:** The child chunks (128–256 tokens) comfortably fit within 512 tokens with the query. After re-ranking, fetch the parent chunk for the LLM.

The last option — re-rank on child chunks, return parent chunks — is the cleanest architectural solution. It keeps the re-ranking model's input small while giving the LLM full context.

---

## LLM-as-Reranker

Instead of a specialized cross-encoder, use the LLM itself to score and rank candidate chunks. This is more expensive but potentially more accurate for nuanced relevance judgments.

### Pointwise Scoring

Score each chunk independently with a relevance prompt. The LLM outputs a score or label for each (query, chunk) pair.

```python
async def llm_pointwise_rerank(
    query: str,
    candidates: list[dict],
    llm_client,
    text_key: str = "text",
    top_k: int = 10
) -> list[dict]:
    """
    Score each candidate independently using the LLM.
    Requires N LLM calls for N candidates — expensive.
    """
    
    scoring_prompt_template = """Rate the relevance of the following passage to the query.

Query: {query}

Passage: {passage}

Rate relevance on a scale of 1-10 where:
1-3 = not relevant or only tangentially related
4-6 = somewhat relevant, partially addresses the query  
7-9 = highly relevant, directly addresses the query
10 = perfectly answers the query completely

Output only the number, nothing else."""
    
    # Score all candidates (parallelize to reduce latency)
    async def score_one(candidate: dict) -> float:
        prompt = scoring_prompt_template.format(
            query=query,
            passage=candidate[text_key][:1000]  # Truncate to avoid token waste
        )
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",  # Use smaller model to reduce cost
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(1.0, min(10.0, score))  # Clamp to [1, 10]
        except ValueError:
            return 5.0  # Default middle score on parse failure
    
    # Score all candidates in parallel
    scores = await asyncio.gather(*[score_one(c) for c in candidates])
    
    scored = [
        {**c, "llm_rerank_score": score}
        for c, score in zip(candidates, scores)
    ]
    
    return sorted(scored, key=lambda x: x["llm_rerank_score"], reverse=True)[:top_k]
```

**Cost:** N LLM API calls per query for N candidates. For 50 candidates with gpt-4o-mini at ~$0.00015 per 1K tokens and ~200 tokens per call: 50 × 200 × $0.00015/1000 ≈ $0.0015 per query. At 10,000 queries per day: ~$15/day just for re-ranking. Not prohibitive but not free.

### Listwise Re-ranking

Instead of scoring each chunk independently, show the LLM all candidates at once and ask it to output a ranked order. One LLM call for all candidates.

```python
async def llm_listwise_rerank(
    query: str,
    candidates: list[dict],
    llm_client,
    text_key: str = "text",
    top_k: int = 10
) -> list[dict]:
    """
    Rank all candidates in a single LLM call.
    More token-efficient than pointwise but context window limited.
    """
    
    # Build numbered list of passages
    passages_text = "\n\n".join([
        f"[{i+1}] {c[text_key][:500]}"  # Truncate each passage
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""Given the following query and passages, rank the passages from 
most relevant to least relevant for answering the query.

Query: {query}

Passages:
{passages_text}

Return ONLY a comma-separated list of passage numbers in order from most to 
least relevant. Example format: 3,1,7,2,5,4,6
Do not include any other text."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",  # Need capable model for this task
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0
    )
    
    raw_ranking = response.choices[0].message.content.strip()
    
    # Parse the ranking
    try:
        ranked_indices = [int(x.strip()) - 1 for x in raw_ranking.split(",")]
        # Filter valid indices and deduplicate
        valid_indices = []
        seen = set()
        for idx in ranked_indices:
            if 0 <= idx < len(candidates) and idx not in seen:
                valid_indices.append(idx)
                seen.add(idx)
        
        reranked = [candidates[i] for i in valid_indices[:top_k]]
        
        # Add synthetic scores based on rank position
        for rank, c in enumerate(reranked):
            c["llm_rerank_score"] = 1.0 - (rank / len(reranked))
        
        return reranked
    
    except (ValueError, IndexError):
        # Fallback to original order if parsing fails
        return candidates[:top_k]
```

**Context window limitation:** Showing 50 candidates, each 500 tokens, to the LLM requires 25,000 tokens of input. This exceeds many models' practical limits and is expensive. Limit listwise re-ranking to 10–20 candidates.

**Sliding window for large candidate sets:** Apply listwise re-ranking iteratively over windows:
1. Rank the first 20 candidates → get ranked top-10.
2. Replace bottom 5 of top-10 with next 5 candidates.
3. Re-rank the new 15-candidate set.
4. Repeat until all candidates are processed.

This is called **RankGPT's sliding window approach** — it applies listwise re-ranking over a moving window to handle more candidates than the context window allows.

### When LLM Re-ranking Justifies the Cost

LLM re-ranking beats cross-encoder re-ranking in specific scenarios:

- **Nuanced relevance:** The query requires inferential reasoning that the cross-encoder misses. "Which of these legal clauses is most favorable to the buyer?" requires understanding contract law, not just term overlap.
- **Long-form answer quality:** For generation tasks where you want the most comprehensive chunk rather than the most similar one.
- **Domain-specific judgment:** When your domain requires expertise the cross-encoder model was not trained on.

For most production RAG systems, a fast cross-encoder provides 95% of the quality benefit at 10% of the cost. Use LLM re-ranking for high-value queries or as an offline evaluation tool.

---

## ColBERT as a Re-ranker

ColBERT was introduced in Lesson 2.2 as a retrieval model. It also functions effectively as a re-ranker: use a fast bi-encoder for first-stage retrieval, then use ColBERT's MaxSim scoring to re-rank the candidate set.

```python
from ragatouille import RAGPretrainedModel

colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def colbert_rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "text",
    top_k: int = 10
) -> list[dict]:
    """
    Use ColBERT MaxSim scoring to re-rank candidates.
    """
    passages = [c[text_key] for c in candidates]
    
    # ColBERT scores using MaxSim over token embeddings
    scores = colbert.rerank(query=query, documents=passages, k=top_k)
    
    # scores is a list of (document, score) tuples ranked by relevance
    reranked = []
    for doc_text, score in scores:
        # Find the original candidate with this text
        for candidate in candidates:
            if candidate[text_key] == doc_text:
                reranked.append({**candidate, "colbert_score": score})
                break
    
    return reranked[:top_k]
```

ColBERT as a re-ranker is generally faster than cross-encoders of equivalent quality because the MaxSim operation is highly parallelizable, and document token embeddings can be partially pre-computed. The storage cost (all token embeddings) is paid at index time, not at re-ranking time.

**When to choose ColBERT over cross-encoder:**
- You already have ColBERT set up for retrieval.
- You need sub-100ms re-ranking latency.
- Your domain benefits from token-level interaction matching.

---

## Cascade Re-ranking

For high-quality production systems, use multiple re-ranking stages in cascade:

```
First retrieval: top-200 from hybrid search (fast)
    ↓
Stage 1 re-rank: fast cross-encoder (MiniLM L-6) → top-50 (moderate)
    ↓
Stage 2 re-rank: accurate cross-encoder or Cohere → top-10 (expensive)
    ↓
LLM generation with top-10 context
```

Each stage reduces the candidate set while using a progressively more accurate (and expensive) ranker. The expensive ranker only ever processes a small set, keeping overall latency manageable.

```python
class CascadeReranker:
    def __init__(self):
        self.fast_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.accurate_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text"
    ) -> list[dict]:
        
        # Stage 1: Fast re-ranking — narrow from 100 to 30
        if len(candidates) > 30:
            stage1_scores = self.fast_reranker.predict(
                [(query, c[text_key]) for c in candidates],
                batch_size=64
            )
            candidates = sorted(
                zip(candidates, stage1_scores),
                key=lambda x: x[1],
                reverse=True
            )
            candidates = [c for c, _ in candidates[:30]]
        
        # Stage 2: Accurate re-ranking — narrow from 30 to 10
        stage2_scores = self.accurate_reranker.predict(
            [(query, c[text_key]) for c in candidates],
            batch_size=32
        )
        
        reranked = sorted(
            [
                {**c, "final_score": float(s)}
                for c, s in zip(candidates, stage2_scores)
            ],
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        return reranked[:10]
```

---

## Re-ranking in the Full Pipeline

Re-ranking fits between retrieval and context assembly:

```
Hybrid retrieval (dense + sparse) → RRF → top-50 candidates
    ↓
Cross-encoder re-ranking → top-10 results
    ↓
Parent chunk lookup (if using parent-child)
    ↓
Context assembly (ordering, deduplication, length management)
    ↓
LLM generation
```

**Re-ranking should happen after parent expansion, not before.** If you use parent-child retrieval, expand child chunks to parent chunks before re-ranking. The cross-encoder should score the full context the LLM will receive — not the small child chunk used for initial retrieval.

Wait — actually this is a trade-off. Re-ranking small child chunks (128–256 tokens) is fast and stays within cross-encoder token limits. Re-ranking large parent chunks (500–2000 tokens) is slower and may exceed limits. A pragmatic approach: re-rank on child chunks to get the final ordering, then fetch parent chunks for the re-ranked top-K. The ordering determined by the child chunk re-ranking is usually a good proxy for the parent chunk relevance.

---

## Latency Benchmarks

Understanding what re-ranking actually costs in production:

| Re-ranker | Candidate count | GPU latency | CPU latency |
|---|---|---|---|
| MiniLM L-6 (batch 32) | 50 chunks | ~5ms | ~50ms |
| MiniLM L-12 (batch 32) | 50 chunks | ~10ms | ~100ms |
| BGE-reranker-large | 50 chunks | ~25ms | ~250ms |
| Cohere Rerank API | 50 chunks | ~150ms | N/A (API) |
| LLM pointwise (parallel) | 20 chunks | ~800ms | ~800ms |
| LLM listwise (single call) | 20 chunks | ~500ms | ~500ms |

For most production RAG systems targeting < 2s total response time, MiniLM L-6 or L-12 on GPU (or Cohere Rerank API) is the practical sweet spot.

> **Interview note:** "How do you handle re-ranking at scale?" — the answer: (1) two-stage pipeline where first-stage retrieval narrows to 50–100 candidates, (2) cross-encoder re-ranking on candidates only (not full corpus), (3) batch processing for GPU efficiency, (4) consider Cohere Rerank API to avoid GPU infrastructure, (5) cascade re-ranking if you need both speed and quality. Key insight to communicate: re-ranking is only feasible because it runs on a small candidate set, not the full corpus.

---

## Fine-Tuning a Re-ranker

When a general cross-encoder does not perform well enough on your domain, fine-tune it.

The training objective is the same as for bi-encoder fine-tuning but applied to cross-encoders:

- Positive pairs: (query, relevant_chunk) → label 1
- Negative pairs: (query, irrelevant_chunk) → label 0
- Hard negatives: (query, chunk_that_looks_relevant_but_is_not) → label 0

Hard negatives are critical. If all negatives are obviously irrelevant, the model learns to separate easy cases but struggles on hard ones (which is exactly what re-ranking needs to handle).

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Prepare training data
train_samples = [
    InputExample(texts=["query about refunds", "Our refund policy allows..."], label=1.0),
    InputExample(texts=["query about refunds", "Shipping takes 3-5 days..."], label=0.0),
    InputExample(texts=["query about refunds", "Returns are processed within 30 days..."], label=0.8),
    # ... thousands more
]

# Load base model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

# Fine-tune
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-reranker"
)
```

---

## Summary

- Re-ranking addresses the coarseness of first-stage retrieval. It scores (query, chunk) pairs jointly, enabling reasoning about relevance that fast retrieval cannot perform.
- Cross-encoders are the standard re-ranking approach. They encode query and chunk together, enabling fine-grained interaction modeling. The two-stage pipeline (fast retrieval → cross-encoder re-ranking) is the production standard.
- MiniLM L-6/L-12 cross-encoders are the practical default: fast, good quality, self-hosted. Cohere Rerank API is the managed alternative.
- LLM re-ranking is more expensive but handles nuanced relevance judgments better. Pointwise is simple but requires N LLM calls. Listwise uses one call but has context window limits.
- ColBERT doubles as a re-ranker using pre-computed token embeddings and MaxSim scoring. Fast and effective.
- Cascade re-ranking — fast ranker narrows to medium set, accurate ranker narrows to final set — balances quality and latency.
- Re-rank on the text that the LLM will actually receive. For parent-child systems, this means re-ranking after parent expansion, or accepting that child-chunk re-ranking is a good proxy.
- Fine-tune cross-encoders with hard negatives when general models underperform on your domain.

---

## What's Next

Lesson 3.7 covers contextual compression and context window packing — how to manage what goes into the LLM's context after retrieval and re-ranking to maximize answer quality while minimizing token waste.