# Lesson A.7 — Embedding Model Evaluation and MTEB [OPTIONAL — Cover Later]

> **Tag: OPTIONAL** — This lesson goes deep on evaluation methodology. Skip if time is short. Come back when you need to defend model selection decisions or build internal benchmarks.

---

## What MTEB Is

MTEB (Massive Text Embedding Benchmark, Muennighoff et al., 2022) is the standard benchmark for comparing embedding models. It evaluates models across 56 datasets and 8 task types:

- **Retrieval (15 datasets):** Given a query, find relevant documents. Uses NDCG@10.
- **Reranking (4 datasets):** Given a query and a list of candidates, order by relevance. Uses MAP.
- **Clustering (11 datasets):** Cluster texts by topic. Uses V-measure.
- **Classification (12 datasets):** Classify texts into categories. Uses accuracy.
- **Pair Classification (3 datasets):** Classify whether two texts are similar/duplicate. Uses AP.
- **Semantic Textual Similarity (10 datasets):** Score sentence pair similarity. Uses Spearman.
- **Summarization (1 dataset):** Score summary quality. Uses Spearman.
- **Bitext Mining (1 dataset):** Find parallel translations. Uses F1.

The leaderboard (huggingface.co/spaces/mteb/leaderboard) ranks models by average score across all tasks or filtered by task type.

---

## What MTEB Measures and What It Does Not

### What It Measures Well

- General English language understanding at the sentence level.
- Cross-task generalization — models that score well across many task types are robust.
- Relative ranking of models on English general-domain content.

### Critical Limitations for RAG Practitioners

**1. MTEB retrieval datasets are mostly web/Wikipedia content.**

The 15 retrieval datasets include:
- TREC-COVID (biomedical literature)
- ArguAna (debate arguments)
- MSMARCO (web search)
- NQ/TriviaQA (Wikipedia-based QA)

If your RAG corpus is legal contracts, financial filings, internal wikis, or code — none of these is in MTEB. A model that ranks 3rd on MTEB may rank 8th on your specific domain.

**2. MTEB queries are web-search-style.**

MTEB retrieval queries look like web search queries: short keyword phrases or question-style queries. If your users write very different queries (conversational, very technical, domain-jargon-heavy), the MTEB ranking may not predict your retrieval quality.

**3. MTEB does not measure latency or cost.**

A model that scores 72.5 on MTEB at 100ms/query may be worse for your production system than one that scores 70.0 at 20ms/query. MTEB has no speed dimension.

**4. MTEB uses fixed evaluation sets.**

The benchmark is static. Models trained after the benchmark was published have potentially seen similar data during training. More recent top-ranked models may be partially "overfit" to the benchmark.

---

## Reading MTEB Results Correctly

When you look at the MTEB leaderboard, filter by what matters for RAG:

```
Filter: Task Type = "Retrieval"
Sort by: Average Retrieval Score (NDCG@10)
Look at: Which datasets in the retrieval subset are similar to your domain
```

Do not look at the overall average — it includes clustering, classification, and STS tasks that are irrelevant for RAG retrieval quality.

For RAG, also look at the BEIR benchmark (subset of MTEB) — it specifically evaluates out-of-domain retrieval generalization, which is more relevant for enterprise RAG than in-distribution web search retrieval.

---

## Building Your Own Domain Evaluation

This is the most important evaluation skill. MTEB tells you which model to start with. Your domain evaluation tells you which model to deploy.

### Step 1 — Build a Domain Evaluation Set

```python
async def build_domain_eval_set(
    corpus_chunks: list[dict],
    llm_client,
    n_queries: int = 200
) -> list[dict]:
    """
    Generate domain-specific evaluation queries from your actual corpus.
    """
    import random
    
    # Sample diverse chunks
    sampled = random.sample(corpus_chunks, min(n_queries * 2, len(corpus_chunks)))
    
    eval_pairs = []
    
    for chunk in sampled[:n_queries]:
        prompt = f"""Generate ONE realistic user query that this document passage would answer.
The query should be phrased naturally, as a real user would type it.

Passage: {chunk['text'][:600]}

Return JSON: {{"query": "...", "difficulty": "easy|medium|hard"}}"""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.5
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        eval_pairs.append({
            "query": result["query"],
            "relevant_chunk_id": chunk["chunk_id"],
            "relevant_chunk_text": chunk["text"],
            "difficulty": result.get("difficulty", "medium")
        })
    
    return eval_pairs
```

### Step 2 — Evaluate Multiple Models

```python
def evaluate_model_on_domain(
    model_name: str,
    eval_pairs: list[dict],
    all_chunks: list[dict],
    k: int = 10
) -> dict:
    """
    Evaluate a candidate embedding model on your domain evaluation set.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer(model_name)
    
    # Check if model needs instruction prefix
    needs_query_prefix = "e5" in model_name.lower()
    needs_passage_prefix = "e5" in model_name.lower() and "intfloat" in model_name.lower()
    
    # Embed all corpus chunks
    chunk_texts = [
        f"passage: {c['text']}" if needs_passage_prefix else c['text']
        for c in all_chunks
    ]
    chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True, batch_size=64)
    chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}
    
    recalls = []
    mrrs = []
    
    for pair in eval_pairs:
        query_text = f"query: {pair['query']}" if needs_query_prefix else pair['query']
        query_embedding = model.encode(query_text, normalize_embeddings=True)
        
        # Compute similarities with all chunks
        similarities = chunk_embeddings @ query_embedding
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        retrieved_ids = [all_chunks[i]["chunk_id"] for i in top_k_indices]
        relevant_id = pair["relevant_chunk_id"]
        
        # Recall@K
        recall = 1.0 if relevant_id in retrieved_ids else 0.0
        recalls.append(recall)
        
        # MRR
        if relevant_id in retrieved_ids:
            rank = retrieved_ids.index(relevant_id) + 1
            mrrs.append(1.0 / rank)
        else:
            mrrs.append(0.0)
    
    return {
        "model": model_name,
        f"recall@{k}": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "n_queries": len(eval_pairs)
    }


# Compare multiple candidate models
candidates = [
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5"
]

results = []
for model_name in candidates:
    result = evaluate_model_on_domain(model_name, eval_pairs, corpus_chunks)
    results.append(result)
    print(f"{model_name}: Recall@10={result['recall@10']:.3f}, MRR={result['mrr']:.3f}")
```

### Step 3 — Compare to MTEB Rankings

After running your domain evaluation, compare to MTEB rankings:

```python
# Hypothetical results showing MTEB divergence:
# Model              | MTEB Rank | Domain Recall@10 | Domain MRR
# -------------------|-----------|-----------------|----------
# bge-large-en-v1.5  |    3rd    |    0.82         | 0.71
# e5-large-v2        |    5th    |    0.87         | 0.76  ← Best on domain
# all-mpnet-base-v2  |   12th    |    0.74         | 0.63
# nomic-embed-v1.5   |    7th    |    0.83         | 0.72

# e5-large-v2 ranks 5th on MTEB but best on your domain.
# Deploy e5-large-v2, not the MTEB top-3 model.
```

This kind of domain divergence from MTEB rankings is common — particularly for:
- Legal and financial domains (specialized vocabulary).
- Short document corpora (FAQ-style content).
- Non-English content (MTEB is heavily English-weighted).
- Query styles that differ significantly from web search.

---

## When to Trust MTEB and When to Override

**Trust MTEB when:**
- You do not have time to build a domain evaluation set yet.
- Your corpus is general English content (similar to web text).
- You are choosing between the top-3 models (they are usually close on any domain).
- You need a quick baseline to start from.

**Override MTEB with domain evaluation when:**
- Your corpus is domain-specific (legal, medical, financial, code).
- Your users write very different queries than web search style.
- You have time before deployment (domain eval takes ~1 day to build properly).
- The production quality difference between top MTEB models matters for your use case.

**The practical answer:** Use MTEB to create a shortlist of 3-5 candidate models. Run domain evaluation on that shortlist. Deploy the winner.

---

## Summary

- MTEB evaluates models on 56 datasets across 8 task types. For RAG, focus on the Retrieval subset and BEIR.
- MTEB has critical limitations: datasets are web/Wikipedia, queries are web-search-style, no latency measurement, static benchmark with potential training data contamination.
- Build a domain evaluation set from your actual corpus and query patterns. This is 30-minute investment that prevents deploying the wrong model.
- MTEB rankings commonly diverge from domain performance by 1-3 rank positions, especially for specialized domains.
- Use MTEB to shortlist → domain evaluation to decide → deploy winner.

---

## What's Next

Lesson A.8 (Optional) covers the full embedding model fine-tuning pipeline — dataset construction at scale, automated hard negative mining with a mining pipeline, LoRA for large models, and evaluation after fine-tuning.