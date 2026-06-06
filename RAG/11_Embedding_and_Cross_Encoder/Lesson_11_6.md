# Lesson A.6 — Cross-Encoder Practical Considerations: Truncation, Deployment, and Fine-Tuning

---

## The 512-Token Truncation Problem

This is the most commonly hit practical limitation of cross-encoders in RAG production systems.

Most cross-encoder models (MiniLM, BERT-base, ELECTRA) have a maximum input length of 512 tokens. This limit applies to the CONCATENATED input — query + separator + document. In practice:

```
Input format: [CLS] query_tokens [SEP] document_tokens [SEP]
Total budget: 512 tokens
Query typically uses: 10-50 tokens
Available for document: 512 - query - 3 special tokens ≈ 450-490 tokens
```

A chunk of 500 characters of plain text is roughly 125 tokens. A chunk of 500 characters with technical terminology, code, or legal language may be 150-200 tokens. If your chunks are 400-500 tokens (roughly 1,500-2,000 characters), they will be SILENTLY TRUNCATED at the 512-token input boundary.

**Why truncation is a serious problem:**

The cross-encoder sees only the first 450 tokens of the chunk. If the answer to the query is in token position 480 of the chunk, the cross-encoder never sees it. It assigns a low relevance score to the chunk not because it is irrelevant, but because the relevant content was cut off.

This creates a systematic failure mode: long chunks that contain the answer at the end get low scores and get pushed out of the top-K after re-ranking. The answer exists in the retrieved set but is dropped during the re-ranking stage.

---

## Diagnosing Truncation Failures

```python
import tiktoken

def diagnose_truncation_risk(
    cross_encoder_tokenizer,
    queries: list[str],
    chunks: list[str],
    max_length: int = 512
) -> dict:
    """
    Check what fraction of query-chunk pairs will be truncated.
    """
    
    truncated_count = 0
    total = len(queries) * len(chunks)
    
    truncation_examples = []
    
    for query in queries[:10]:  # Sample
        for chunk in chunks[:10]:
            tokens = cross_encoder_tokenizer(
                query, chunk,
                return_tensors="pt",
                truncation=False  # Don't truncate — measure the natural length
            )
            
            natural_length = tokens["input_ids"].shape[1]
            
            if natural_length > max_length:
                truncated_count += 1
                truncation_examples.append({
                    "query_preview": query[:50],
                    "natural_length": natural_length,
                    "truncated_tokens": natural_length - max_length,
                    "chunk_length_tokens": natural_length - len(cross_encoder_tokenizer(query)["input_ids"]) - 3
                })
    
    return {
        "truncation_rate": truncated_count / min(total, 100),
        "examples": truncation_examples[:5]
    }
```

If truncation rate is above 20%, you have a systematic problem affecting re-ranking quality.

---

## Solutions to the Truncation Problem

### Solution 1 — Use Parent-Child Chunking (Best for RAG)

The cleanest architectural solution: store small child chunks (128-256 tokens) for retrieval, but at re-ranking time, re-rank on the child chunks (which fit within 512 tokens), then fetch the parent chunk (500-2000 tokens) for the LLM.

```
Child chunk (128-256 tokens) → Re-ranking (fits in 512 tokens)
    ↓ Re-ranking selects top-K by child chunk score
    ↓ Fetch parent chunks for selected children
Parent chunk (500-2000 tokens) → LLM context
```

The ordering determined by re-ranking child chunks is a reliable proxy for parent chunk relevance. This is the architecture recommendation from Lesson 2.7, and it also elegantly solves the truncation problem.

### Solution 2 — Sliding Window Re-ranking

Split long chunks into overlapping windows, re-rank each window independently, and use the maximum score across all windows as the chunk's final score:

```python
def rerank_with_sliding_window(
    cross_encoder,
    query: str,
    chunk_text: str,
    window_size_tokens: int = 400,
    overlap_tokens: int = 100,
    tokenizer = None
) -> float:
    """
    Score a long chunk by sliding window re-ranking.
    Returns the maximum score across all windows.
    """
    
    # Tokenize the chunk
    chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
    
    if len(chunk_tokens) <= window_size_tokens:
        # Short enough — score directly
        return cross_encoder.predict([(query, chunk_text)])[0]
    
    # Create overlapping windows
    windows = []
    start = 0
    while start < len(chunk_tokens):
        end = min(start + window_size_tokens, len(chunk_tokens))
        window_tokens = chunk_tokens[start:end]
        window_text = tokenizer.decode(window_tokens)
        windows.append(window_text)
        
        if end == len(chunk_tokens):
            break
        start += window_size_tokens - overlap_tokens
    
    # Score all windows
    pairs = [(query, window) for window in windows]
    scores = cross_encoder.predict(pairs)
    
    # Return max score (the most relevant window)
    return float(max(scores))
```

**Trade-off:** This increases the number of cross-encoder forward passes proportionally to chunk length divided by window size. A 1,000-token chunk with 400-token windows and 100-token overlap requires ~3 forward passes instead of 1. Latency increases proportionally.

### Solution 3 — Use a Long-Context Cross-Encoder

Some cross-encoders are specifically trained with longer context windows:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`: 512 tokens (standard limit).
- `BAAI/bge-reranker-large`: supports longer inputs with attention scaling tricks.
- `mixedbread-ai/mxbai-rerank-large-v1`: supports up to 8192 tokens.

For corpora with inherently long chunks (legal documents, technical manuals), a long-context cross-encoder eliminates truncation at the cost of higher latency per forward pass.

### Solution 4 — Front-Load Relevant Content in Chunks

Design your chunking strategy so the most answer-relevant content appears in the first half of each chunk. This works because even when truncated, the cross-encoder sees the most important content.

For structured documents (policies, manuals), this aligns naturally with how content is written: topic sentences and key facts appear first, elaboration and examples later.

---

## Self-Hosted vs. Cohere Rerank API

Production teams face a clear choice: self-host a cross-encoder or use the Cohere Rerank API.

### Self-Hosted Cross-Encoders

```python
from sentence_transformers import CrossEncoder
import torch

class SelfHostedReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512
        )
        self.batch_size = batch_size
    
    def rerank(self, query: str, candidates: list[dict], k: int = 10) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        
        # Process in batches for GPU efficiency
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores.tolist())
        
        # Attach scores and sort
        scored = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, all_scores)
        ]
        
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:k]
```

**Self-hosted advantages:**
- No per-call API cost — only infrastructure cost.
- Data never leaves your infrastructure (critical for sensitive domains).
- Full control over model version, fine-tuning, batch size.
- Latency is predictable and does not depend on external API availability.
- Can fine-tune the model on domain-specific data.

**Self-hosted disadvantages:**
- Requires GPU infrastructure (A10G or similar: ~$2-4/hour).
- Operational overhead: model serving, version management, monitoring.
- Must handle scaling during peak load.

**Cost calculation (self-hosted):**

```
1 A10G GPU instance: ~$2/hour
At 100 queries/minute, 50 candidates each:
5,000 cross-encoder calls/minute
At 30ms/call on GPU: 150 seconds of compute per minute → need ~3 GPUs
3 GPUs × $2/hour = $6/hour → $144/day at constant load
Actual load is spiky → ~$50-80/day with auto-scaling
```

### Cohere Rerank API

```python
import cohere

co = cohere.Client(api_key="your-key")

def cohere_rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "text",
    k: int = 10,
    model: str = "rerank-english-v3.0"
) -> list[dict]:
    
    documents = [c[text_key] for c in candidates]
    
    response = co.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=k,
        return_documents=False  # Just scores and indices
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

**Cohere advantages:**
- No GPU infrastructure — zero operational overhead.
- Excellent quality — Cohere's rerank models are well-maintained.
- Scales automatically — no need to manage peak load.
- Available immediately — no setup required.

**Cohere disadvantages:**
- Per-call cost: ~$0.001 per 1K characters for reranking.
- Data goes to Cohere's servers — compliance issues for some domains.
- API latency adds ~100-300ms (network round-trip + processing).
- No customization — cannot fine-tune for your specific domain.
- Dependency on external service uptime.

**Cost calculation (Cohere):**

```
100 queries/minute, 50 candidates × 500 chars each = 25,000 chars/query
25,000 chars × $0.001/1000 = $0.025 per query
100 queries/minute × 60 minutes × 24 hours = 144,000 queries/day
144,000 × $0.025 = $3,600/day
```

At high QPS, self-hosted is dramatically cheaper. At low QPS (< 10 queries/minute), Cohere is cheaper because GPU instance costs exceed API costs.

**The break-even point:** For most teams, Cohere is cheaper below ~20 queries/minute. Self-hosted is cheaper above that. Calculate for your specific load.

---

## When to Choose Cross-Encoder vs. ColBERT

Both cross-encoder and ColBERT are used as re-rankers on top of bi-encoder retrieval. They are not alternatives for first-stage retrieval (too slow for that). The choice is about which re-ranking approach fits your constraints.

### Cross-Encoder

- **Scores:** Single relevance score per (query, document) pair.
- **Speed:** Requires one full forward pass per (query, doc) pair at re-ranking time. Cannot precompute document representations.
- **Accuracy:** Very high — full joint attention between query and document.
- **Latency for 50 candidates:** ~50 × 15ms (MiniLM L-6 GPU) = ~750ms.
- **Best when:** You can afford the latency, need the highest accuracy, and have diverse query patterns that benefit from full joint attention.

### ColBERT as Re-ranker

- **Scores:** MaxSim over per-token embeddings.
- **Speed:** Document token embeddings are pre-computed and stored. At re-ranking time, only query token embeddings need to be computed (one forward pass), then MaxSim is computed between query tokens and pre-stored document token embeddings.
- **Accuracy:** High — better than bi-encoder cosine similarity, slightly below full cross-encoder.
- **Latency for 50 candidates:** ~50ms for MaxSim computation (largely parallelizable).
- **Best when:** Latency is tight, you need sub-100ms re-ranking, and you can afford the storage cost of per-token embeddings.

```python
# ColBERT re-ranking latency advantage:
# Cross-encoder: 50 forward passes × 15ms = 750ms
# ColBERT: 1 query encoding + 50 MaxSim operations = ~50ms
# 15x faster at re-ranking time, at the cost of storage
```

**The storage trade-off:**

ColBERT stores N token embeddings per document (where N is document length in tokens). A 256-token chunk with 128-dim token embeddings requires 256 × 128 × 4 bytes = 131KB per chunk. At 100K chunks, that is 13GB — vs. 600MB for standard bi-encoder embeddings at 1536 dims.

**Decision rule:**

| Requirement | Choose |
|---|---|
| Highest accuracy, latency < 2s acceptable | Cross-encoder (DeBERTa or BERT-large) |
| Good accuracy, latency < 500ms | Cross-encoder (MiniLM-L-12) |
| Latency < 100ms, good accuracy | ColBERT |
| No GPU, managed service preferred | Cohere Rerank API |
| Domain-specific training required | Self-hosted cross-encoder + fine-tuning |

---

## Fine-Tuning a Cross-Encoder: The Complete Workflow

When a general cross-encoder underperforms on your domain, fine-tune it.

### Step 1 — Collect Training Data

```python
async def create_cross_encoder_training_dataset(
    eval_queries: list[str],
    retriever,
    ground_truth: dict,  # query -> list of relevant chunk_ids
    n_hard_negatives: int = 5
) -> list[dict]:
    """
    Build training data from existing retrieval system + ground truth.
    """
    training_data = []
    
    for query in eval_queries:
        relevant_ids = set(ground_truth.get(query, []))
        
        if not relevant_ids:
            continue
        
        # Retrieve candidates (these will be our negatives)
        candidates = await retriever.retrieve(query, k=20)
        
        # Get positive examples
        positives = [
            c for c in candidates
            if c["chunk_id"] in relevant_ids
        ]
        
        # Hard negatives: retrieved but not relevant
        hard_negatives = [
            c for c in candidates
            if c["chunk_id"] not in relevant_ids
        ][:n_hard_negatives]
        
        if not positives:
            continue  # Skip queries with no retrieved positives
        
        # Create training pairs
        for pos in positives[:2]:
            # Positive pair
            training_data.append({
                "query": query,
                "document": pos["text"],
                "label": 1.0
            })
            
            # Negative pairs
            for neg in hard_negatives:
                training_data.append({
                    "query": query,
                    "document": neg["text"],
                    "label": 0.0
                })
    
    return training_data


### Step 2 — Fine-Tune

```python
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

def fine_tune_cross_encoder(
    base_model: str,
    training_data: list[dict],
    output_path: str,
    epochs: int = 2,
    batch_size: int = 16,
    warmup_steps: int = 100
) -> CrossEncoder:
    """
    Fine-tune a cross-encoder on domain-specific training data.
    """
    
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    
    train_samples = [
        InputExample(
            texts=[d["query"], d["document"]],
            label=d["label"]
        )
        for d in training_data
    ]
    
    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size
    )
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        # Use BCEWithLogitsLoss for binary classification
        loss_fct=torch.nn.BCEWithLogitsLoss()
    )
    
    return model
```

### Step 3 — Evaluate and Compare

```python
async def compare_cross_encoders(
    baseline_model: CrossEncoder,
    fine_tuned_model: CrossEncoder,
    eval_queries: list[str],
    retriever,
    ground_truth: dict,
    k: int = 10
) -> dict:
    """
    Compare baseline vs. fine-tuned cross-encoder on NDCG@10.
    """
    import numpy as np
    
    baseline_ndcgs = []
    finetuned_ndcgs = []
    
    for query in eval_queries:
        candidates = await retriever.retrieve(query, k=50)
        relevant_ids = set(ground_truth.get(query, []))
        
        if not relevant_ids:
            continue
        
        # Score with baseline
        baseline_scores = baseline_model.predict(
            [(query, c["text"]) for c in candidates]
        )
        baseline_ranked = sorted(
            zip(candidates, baseline_scores),
            key=lambda x: x[1], reverse=True
        )[:k]
        
        # Score with fine-tuned
        finetuned_scores = fine_tuned_model.predict(
            [(query, c["text"]) for c in candidates]
        )
        finetuned_ranked = sorted(
            zip(candidates, finetuned_scores),
            key=lambda x: x[1], reverse=True
        )[:k]
        
        # Compute NDCG@K for each
        def ndcg(ranked: list, relevant: set, k: int) -> float:
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, (c, _) in enumerate(ranked[:k])
                if c["chunk_id"] in relevant
            )
            ideal_dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank in range(min(len(relevant), k))
            )
            return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        baseline_ndcgs.append(ndcg(baseline_ranked, relevant_ids, k))
        finetuned_ndcgs.append(ndcg(finetuned_ranked, relevant_ids, k))
    
    return {
        "baseline_ndcg@10": float(np.mean(baseline_ndcgs)),
        "finetuned_ndcg@10": float(np.mean(finetuned_ndcgs)),
        "improvement": float(np.mean(finetuned_ndcgs)) - float(np.mean(baseline_ndcgs)),
        "deploy_finetuned": float(np.mean(finetuned_ndcgs)) > float(np.mean(baseline_ndcgs)) + 0.01
    }
```

Expect 5-15% NDCG improvement from domain fine-tuning when the domain vocabulary or query style is significantly different from MS MARCO.

---

## Summary

- The 512-token truncation problem is the most common production cross-encoder failure. Diagnose it by measuring natural token counts before truncation. Fix with parent-child chunking (clean architecture), sliding window scoring (works for existing systems), or long-context cross-encoders (best quality, higher latency).
- Self-hosted cross-encoders are cheaper at > 20 queries/minute. Cohere Rerank is cheaper at lower QPS and requires zero infrastructure management. Calculate your actual load before deciding.
- Cross-encoder scores are for ranking within a model — not for thresholding or comparison across models. Calibrate thresholds on your own evaluation set.
- ColBERT re-ranking is 10-15× faster than cross-encoder re-ranking due to pre-computed document token embeddings. Choose ColBERT when latency is tight; choose cross-encoder when accuracy is paramount.
- Domain fine-tuning with hard negatives mined from your retrieval system produces 5-15% NDCG improvement. The training loop is straightforward; the data collection is the critical step.

---

## What's Next

Lesson A.7 (Optional) covers embedding model evaluation in depth — MTEB internals, what the benchmark actually measures, building your own domain evaluation, and why MTEB rankings diverge from production performance.

Lesson A.8 (Optional) covers the full embedding fine-tuning pipeline — dataset construction at scale, hard negative mining pipeline, LoRA for large models, evaluation after training.