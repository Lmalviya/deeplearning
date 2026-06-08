# Lesson 9.4 — Cost Management at Scale: Token Budgets, Caching, Batching, and Tiered Retrieval

---

## The Cost Problem at Scale

A RAG system that costs $50/month at 1,000 queries/day does not cost $50,000/month at 1,000,000 queries/day. With smart cost engineering, it costs $5,000-10,000/month. Without it, it costs $50,000+.

The difference is deliberate cost architecture — designing each component to minimize spend without sacrificing quality.

The three biggest cost drivers in a RAG system, in order:
1. **LLM generation tokens** (~60-70% of total cost)
2. **Embedding computation** (~15-20%)
3. **Vector database storage + compute** (~10-15%)

This lesson covers the specific techniques that reduce each.

---

## Cost Baseline: What You Are Actually Paying

Before optimizing, establish a per-query cost breakdown:

```python
def compute_per_query_cost(metrics: dict) -> dict:
    """
    Compute the fully-loaded cost per query.
    """
    
    # LLM generation cost
    avg_input_tokens = metrics["avg_input_tokens"]   # Typically 2000-5000
    avg_output_tokens = metrics["avg_output_tokens"] # Typically 200-500
    
    # GPT-4o pricing
    llm_cost_per_query = (
        avg_input_tokens * 5.00 / 1_000_000 +
        avg_output_tokens * 15.00 / 1_000_000
    )
    
    # Query rewrite LLM (gpt-4o-mini)
    rewrite_cost_per_query = (
        200 * 0.15 / 1_000_000 +   # ~200 tokens input
        50 * 0.60 / 1_000_000      # ~50 tokens output
    )
    
    # Embedding cost (or GPU amortized cost if self-hosted)
    embedding_cost_per_query = (
        avg_input_tokens * 0.13 / 1_000_000  # text-embedding-3-large
    )
    
    # Vector DB search (amortized infrastructure cost)
    qdrant_cost_per_query = 0.0002  # Estimated from instance cost / daily QPS
    
    total_cost_per_query = (
        llm_cost_per_query +
        rewrite_cost_per_query +
        embedding_cost_per_query +
        qdrant_cost_per_query
    )
    
    return {
        "llm_generation": llm_cost_per_query,
        "query_rewrite_llm": rewrite_cost_per_query,
        "embedding": embedding_cost_per_query,
        "vector_db": qdrant_cost_per_query,
        "total_per_query": total_cost_per_query,
        "monthly_at_100k_queries": total_cost_per_query * 100_000 * 30
    }
```

---

## Optimization 1: Context Token Budget

The single most impactful cost lever: how many tokens you send to the LLM. At $5/million input tokens (GPT-4o), reducing average context from 5,000 to 2,500 tokens cuts LLM input costs by 50%.

### Contextual Compression as a Cost Tool

```python
async def budget_aware_context_assembly(
    chunks: list[dict],
    query: str,
    token_budget: int,
    llm_client,
    compression_model: str = "gpt-4o-mini"  # Cheap model for compression
) -> tuple[str, int]:
    """
    Assemble context within a strict token budget.
    Use cheap compression rather than expensive context.
    """
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # First, check if chunks fit without compression
    full_context = "\n\n".join(c["text"] for c in chunks)
    full_tokens = len(enc.encode(full_context))
    
    if full_tokens <= token_budget:
        return full_context, full_tokens
    
    # Need to compress. Cost comparison:
    # Option A: Large context → GPT-4o at $5/M input tokens
    # Option B: Compress with gpt-4o-mini ($0.15/M) → smaller GPT-4o call
    
    # Compress each chunk to fit budget
    compressed_chunks = []
    tokens_used = 0
    tokens_per_chunk = token_budget // len(chunks)
    
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk["text"]))
        
        if chunk_tokens <= tokens_per_chunk:
            compressed_chunks.append(chunk["text"])
            tokens_used += chunk_tokens
        else:
            # Compress this chunk
            compressed = await compress_chunk_to_budget(
                chunk_text=chunk["text"],
                query=query,
                max_tokens=tokens_per_chunk,
                llm_client=llm_client,
                model=compression_model
            )
            compressed_chunks.append(compressed)
            tokens_used += len(enc.encode(compressed))
    
    return "\n\n".join(compressed_chunks), tokens_used


async def compress_chunk_to_budget(
    chunk_text: str,
    query: str,
    max_tokens: int,
    llm_client,
    model: str
) -> str:
    """
    Compress a chunk to fit within max_tokens while preserving
    information relevant to the query.
    
    Cost: ~$0.0001 per chunk with gpt-4o-mini
    Savings: reduce expensive gpt-4o input by 100-500 tokens per chunk
    """
    
    prompt = f"""Compress this text to under {max_tokens} tokens while keeping all information relevant to: "{query}"

Text:
{chunk_text}

Compressed (keep exact numbers, dates, names):"""
    
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    
    return response.choices[0].message.content
```

### Dynamic Token Budget by Query Type

Not all queries need the same context length. A simple factual lookup needs 500 tokens. A synthesis question needs 5,000 tokens.

```python
TOKEN_BUDGETS_BY_QUERY_TYPE = {
    "factual_lookup": 1_000,    # Short answer, precise context
    "procedural": 2_000,        # Step-by-step process
    "comparison": 3_000,        # Two things to compare
    "synthesis": 5_000,         # Multiple topics
    "analysis": 8_000,          # Deep analysis
    "default": 3_000
}

async def get_query_token_budget(query: str, llm_client) -> int:
    """Classify query and return appropriate token budget."""
    
    # Fast heuristic (no LLM call)
    word_count = len(query.split())
    
    compare_words = ["compare", "vs", "versus", "difference", "contrast"]
    synthesis_words = ["summarize", "overview", "all", "across", "throughout"]
    lookup_words = ["what is", "when did", "who is", "how many", "what date"]
    
    query_lower = query.lower()
    
    if any(w in query_lower for w in lookup_words) and word_count < 10:
        return TOKEN_BUDGETS_BY_QUERY_TYPE["factual_lookup"]
    
    if any(w in query_lower for w in compare_words):
        return TOKEN_BUDGETS_BY_QUERY_TYPE["comparison"]
    
    if any(w in query_lower for w in synthesis_words):
        return TOKEN_BUDGETS_BY_QUERY_TYPE["synthesis"]
    
    return TOKEN_BUDGETS_BY_QUERY_TYPE["default"]
```

---

## Optimization 2: Model Tiering

Use expensive models (GPT-4o) only when needed. Use cheaper models (GPT-4o-mini) for tasks where they are sufficient.

```python
MODEL_TIERS = {
    "premium": {
        "model": "gpt-4o",
        "cost_per_1m_input": 5.00,
        "cost_per_1m_output": 15.00,
        "use_for": "Complex analysis, legal/financial Q&A, multi-hop reasoning"
    },
    "standard": {
        "model": "gpt-4o-mini",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "use_for": "Most RAG Q&A, support chatbot, knowledge base search"
    },
    "utility": {
        "model": "gpt-4o-mini",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "use_for": "Query rewriting, IDK detection, compression, routing"
    }
}

# Cost comparison: 100K queries/day
# All GPT-4o (3000 input, 400 output): $1,500 + $600 = $2,100/day
# Tiered (70% gpt-4o-mini, 30% gpt-4o): ~$700/day (67% reduction)

async def route_to_model_tier(
    query: str,
    query_complexity: str,  # From classifier
    user_plan: str
) -> str:
    """Select appropriate model tier for this query."""
    
    # Enterprise users always get premium
    if user_plan == "enterprise":
        return "premium"
    
    # Complex queries need premium regardless of plan
    if query_complexity in ["complex", "multi_hop", "analytical"]:
        return "premium"
    
    # Most queries work fine with standard
    return "standard"
```

---

## Optimization 3: Batching for the OpenAI Batch API

For non-real-time tasks (document summarization, classification, synthetic QA generation, offline evaluation), use OpenAI's Batch API for a 50% cost reduction.

```python
import json
import time

async def batch_generate_document_summaries(
    chunks: list[dict],
    output_file: str
) -> dict:
    """
    Use OpenAI Batch API for 50% cost reduction on offline tasks.
    Latency: up to 24 hours (acceptable for offline processing).
    """
    
    client = AsyncOpenAI()
    
    # Prepare batch requests
    batch_requests = []
    
    for i, chunk in enumerate(chunks):
        request = {
            "custom_id": f"chunk-{chunk['chunk_id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Summarize in 2 sentences:\n\n{chunk['text'][:2000]}"
                    }
                ]
            }
        }
        batch_requests.append(request)
    
    # Write requests to JSONL file
    with open("/tmp/batch_requests.jsonl", "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    
    # Upload batch file
    with open("/tmp/batch_requests.jsonl", "rb") as f:
        batch_file = await client.files.create(file=f, purpose="batch")
    
    # Submit batch job
    batch_job = await client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print(f"Batch job submitted: {batch_job.id}")
    print(f"Cost: 50% of standard pricing")
    print(f"Estimated completion: up to 24 hours")
    
    return {"batch_id": batch_job.id, "request_count": len(batch_requests)}


async def retrieve_batch_results(batch_id: str) -> dict[str, str]:
    """Poll for batch completion and retrieve results."""
    
    client = AsyncOpenAI()
    
    while True:
        batch = await client.batches.retrieve(batch_id)
        
        if batch.status == "completed":
            break
        elif batch.status == "failed":
            raise Exception(f"Batch failed: {batch.errors}")
        
        print(f"Batch status: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total})")
        await asyncio.sleep(60)  # Check every minute
    
    # Download results
    output_file = await client.files.content(batch.output_file_id)
    
    results = {}
    for line in output_file.text.split("\n"):
        if not line:
            continue
        result = json.loads(line)
        chunk_id = result["custom_id"].replace("chunk-", "")
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        results[chunk_id] = content
    
    return results
```

---

## Optimization 4: Tiered Retrieval (Cheap-to-Expensive Pipeline)

Not every query needs the full pipeline. A tiered approach runs cheaper stages first and escalates only when needed.

```python
class TieredRetriever:
    """
    Three-tier retrieval that escalates cost only when lower tiers are insufficient.
    
    Tier 1: Exact cache lookup (< 1ms, $0)
    Tier 2: Semantic cache + BM25 only (no GPU embedding, 10ms, ~$0.00001)
    Tier 3: Full pipeline with re-ranking (200ms, ~$0.003)
    """
    
    def __init__(self, exact_cache, semantic_cache, full_retriever, cheap_embedder):
        self.exact = exact_cache
        self.semantic = semantic_cache
        self.full = full_retriever
        self.cheap_embedder = cheap_embedder  # Smaller, faster embedding model
    
    async def retrieve(self, query: str, user_context: dict) -> dict:
        
        # Tier 1: Exact cache (free)
        exact_hit = await self.exact.get(query, user_context, k=10)
        if exact_hit:
            return {"chunks": exact_hit, "tier": 1, "cost": 0}
        
        # Tier 2: Cheap embedding + BM25 + semantic cache (very cheap)
        cheap_embedding = await self.cheap_embedder.embed(query)  # all-MiniLM: free self-hosted
        
        semantic_hit = await self.semantic.lookup(query, cheap_embedding)
        if semantic_hit and semantic_hit["cache_score"] > 0.95:
            return {"chunks": semantic_hit["results"], "tier": 2, "cost": 0.00001}
        
        # BM25 retrieval (no GPU needed)
        bm25_results = await bm25_search(query, k=10)
        
        if bm25_results and bm25_results[0].get("score", 0) > 0.8:
            # BM25 found a high-confidence match — use it
            return {"chunks": bm25_results, "tier": 2, "cost": 0.00002}
        
        # Tier 3: Full pipeline (expensive but comprehensive)
        chunks = await self.full.retrieve(query, user_context)
        return {"chunks": chunks, "tier": 3, "cost": 0.003}
```

**Tier 2 hit rate in practice:** For support RAG systems with repetitive queries, Tier 1+2 handles 50-60% of traffic. Tier 3 handles only the remaining 40-50%. Cost reduction: ~55%.

---

## Optimization 5: Generation Cost by Answer Length

Control output token count based on query type:

```python
MAX_OUTPUT_TOKENS_BY_TYPE = {
    "yes_no": 50,          # "Yes, employees are eligible after 6 months."
    "factual_short": 100,  # Single fact with brief context
    "factual_medium": 300, # Fact with explanation
    "procedural": 500,     # Step-by-step process
    "synthesis": 800,      # Multi-point analysis
    "summary": 1000        # Document summary
}

async def type_aware_generation(
    query: str,
    context: str,
    llm_client
) -> str:
    """Use appropriate max_tokens based on expected answer length."""
    
    query_type = classify_expected_answer_length(query)
    max_tokens = MAX_OUTPUT_TOKENS_BY_TYPE.get(query_type, 400)
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=max_tokens,
        temperature=0.1
    )
    
    return response.choices[0].message.content


def classify_expected_answer_length(query: str) -> str:
    """Classify expected answer length from query characteristics."""
    
    query_lower = query.lower()
    
    if any(query_lower.startswith(x) for x in ["is ", "can ", "do ", "does ", "are ", "should "]):
        return "yes_no"
    
    if any(x in query_lower for x in ["what is the", "when", "where", "who"]):
        return "factual_short"
    
    if "how do i" in query_lower or "how to" in query_lower:
        return "procedural"
    
    if any(x in query_lower for x in ["summarize", "overview", "explain", "describe"]):
        return "summary"
    
    return "factual_medium"
```

---

## Cost Dashboard: Tracking Spend Per Component

```python
class CostTracker:
    def __init__(self, metrics_store):
        self.metrics = metrics_store
    
    async def track_query_cost(
        self,
        trace_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        tier: int,
        cache_hit: bool
    ):
        """Record cost for a single query."""
        
        # Compute token costs
        pricing = {
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
        
        llm_cost = (
            input_tokens * model_pricing["input"] / 1_000_000 +
            output_tokens * model_pricing["output"] / 1_000_000
        )
        
        await self.metrics.record({
            "trace_id": trace_id,
            "llm_cost_usd": llm_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "retrieval_tier": tier,
            "cache_hit": cache_hit,
            "timestamp": datetime.utcnow()
        })
    
    async def daily_cost_report(self) -> dict:
        """Aggregate daily costs and identify top spenders."""
        
        today_queries = await self.metrics.get_today_queries()
        
        total_llm_cost = sum(q["llm_cost_usd"] for q in today_queries)
        cache_hits = sum(1 for q in today_queries if q["cache_hit"])
        
        return {
            "total_llm_cost_today": total_llm_cost,
            "query_count": len(today_queries),
            "avg_cost_per_query": total_llm_cost / len(today_queries) if today_queries else 0,
            "cache_hit_rate": cache_hits / len(today_queries) if today_queries else 0,
            "estimated_monthly_cost": total_llm_cost * 30,
            "cost_by_model": {
                model: sum(q["llm_cost_usd"] for q in today_queries if q["model"] == model)
                for model in set(q["model"] for q in today_queries)
            },
            "tier_distribution": {
                f"tier_{tier}": sum(1 for q in today_queries if q["retrieval_tier"] == tier) / len(today_queries)
                for tier in [1, 2, 3]
            }
        }
```

---

## Cost Reduction Summary Table

| Technique | Implementation Effort | Cost Reduction | Notes |
|---|---|---|---|
| Query+semantic caching | Medium | 40-60% | Highest ROI for repetitive queries |
| Context compression | Medium | 20-40% | Depends on chunk verbosity |
| Model tiering | Low | 30-50% | 70% mini + 30% GPT-4o |
| Dynamic token budgets | Low | 10-25% | Type-aware max_tokens |
| Batch API for offline | Low | 50% | Latency tradeoff |
| Tiered retrieval | Medium | 10-30% | Skips GPU on cached paths |
| Answer length control | Low | 10-20% | Prevents padding |

**Realistic combined savings:** Implementing all techniques reduces a $50,000/month spend to $12,000-18,000/month. The largest savings come from caching (most impactful, least quality impact) and model tiering.

---

## Summary

- LLM generation tokens are 60-70% of total cost. Reducing context size is the highest-leverage optimization.
- Context compression with gpt-4o-mini is cost-effective: spend $0.0001 compressing a chunk to save $0.005 in GPT-4o input tokens — 50× ROI.
- Model tiering: use gpt-4o-mini for 70%+ of queries (most Q&A, support, routing tasks). Use GPT-4o only for complex analysis.
- Batch API provides 50% discount for offline tasks (document summarization, evaluation, synthetic QA generation). Use it whenever latency tolerance allows.
- Tiered retrieval escalates cost only when lower tiers are insufficient. Tier 1+2 (cache + BM25) handles 50-60% of traffic for free.
- Track cost per query, per model, per tier. Alert when daily cost exceeds budget. Identify the top 10% of expensive queries for targeted optimization.

---

## What's Next

Lesson 9.5 covers debugging in production at scale — distributed tracing, log aggregation, and how to diagnose failures when you have millions of traces and no reproduction steps.