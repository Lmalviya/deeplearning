# Case Study 3 — Customer Support RAG: High QPS, Freshness Requirements, and Feedback Loops

---

## Problem Statement

A SaaS company with 50,000 business customers wants to replace their tiered support model with an AI-first support system. Currently:
- Tier 1 (80% of tickets): simple how-to questions answered from knowledge base articles.
- Tier 2 (15%): product configuration and troubleshooting.
- Tier 3 (5%): bugs and complex edge cases requiring engineering.

The goal: an AI assistant handles Tier 1 and most of Tier 2, escalating only when confidence is low or the issue requires system access.

The corpus:
- 3,200 knowledge base articles (product documentation, how-to guides, FAQs).
- 180 product release notes (updated weekly).
- 45 product feature specifications.
- Historical resolved tickets (500,000+ archived).
- Real-time product status feed (incidents, maintenance windows).

The operational requirements:
- **QPS:** 2,000 queries per minute at peak (Monday mornings, post-release days).
- **Latency:** p50 < 1.5 seconds, p95 < 3 seconds. Users wait on a chat widget — every second matters.
- **Freshness:** Release notes and product changes reflected within 15 minutes.
- **Escalation:** When confidence is below threshold, route to human agent with full context.
- **Feedback loop:** Every resolved/rejected answer feeds back to improve the system.

This case study is primarily about operational engineering — how to build a RAG system that is fast, fresh, and reliable at scale, not just accurate.

---

## Architecture Design Decisions

### Decision 1 — Latency Budget: Every Millisecond Allocated

At p95 < 3 seconds, there is no room for unplanned latency. Every component must be budgeted before building.

```
Target p95: 3,000ms
  ├── Query understanding + rewrite: 150ms (gpt-4o-mini, async)
  ├── Dense retrieval (cached query embedding): 20ms
  ├── Sparse retrieval (BM25): 10ms
  ├── RRF fusion: 1ms
  ├── Cross-encoder re-rank (MiniLM L-6 on GPU): 30ms
  ├── Context assembly: 5ms
  ├── LLM generation (gpt-4o-mini streaming): 800ms first token, 1500ms full
  └── Response post-processing: 5ms
Total budget: ~2,000ms (33% headroom for variance)
```

Key decisions driven by this budget:
- **gpt-4o-mini for generation**, not gpt-4o. At p95 = 1.5s vs. 2.5s for gpt-4o, the tradeoff is worth it. Quality is monitored closely.
- **MiniLM L-6 for re-ranking**, not L-12 or a larger model. 30ms vs. 80ms.
- **Streaming responses** — user sees first tokens at 800ms even if full response takes 1.5s.
- **Query embedding cache** — repeated or similar queries use cached embeddings.

### Decision 2 — High QPS Architecture

2,000 queries per minute = ~33 queries per second on average, with peaks at 100+ QPS.

**Embedding serving:**

```python
# GPU embedding server: batching strategy
class BatchEmbeddingServer:
    def __init__(self, model, max_batch_size: int = 64, max_wait_ms: int = 10):
        self.model = model
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms
        self.pending = asyncio.Queue()
    
    async def embed(self, text: str) -> list[float]:
        """
        Submit text for embedding. Returns when batch is processed.
        """
        future = asyncio.Future()
        await self.pending.put((text, future))
        return await future
    
    async def _process_batches(self):
        """
        Continuously process pending embedding requests in batches.
        """
        while True:
            batch = []
            
            # Collect items for up to max_wait_ms or max_batch_size
            deadline = asyncio.get_event_loop().time() + self.max_wait / 1000
            
            while len(batch) < self.max_batch:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.pending.get(), timeout=timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                texts = [item[0] for item in batch]
                futures = [item[1] for item in batch]
                
                # Single GPU batch forward pass
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                
                for future, embedding in zip(futures, embeddings):
                    future.set_result(embedding.tolist())
```

This batching strategy dramatically improves GPU utilization — instead of 33 individual embedding calls per second (wasting GPU capacity), you batch them into groups of 16-64 and do one efficient forward pass.

**Connection pooling for vector DB:**

```python
from qdrant_client import QdrantClient, AsyncQdrantClient

# Async client with connection pool
qdrant = AsyncQdrantClient(
    url="http://qdrant:6333",
    grpc_port=6334,
    prefer_grpc=True,  # gRPC is faster than HTTP for high QPS
    timeout=1.0        # Fail fast — better IDK than slow
)
```

**Horizontal scaling topology:**

```
[Load Balancer]
    ├── RAG API Pod 1 (2 CPU, 4GB RAM, shared GPU access)
    ├── RAG API Pod 2
    ├── RAG API Pod 3
    └── ... (auto-scales to 10 pods at peak)
         │
         ├── GPU Embedding Server (2× A10G, shared across all pods)
         ├── Qdrant (3-node cluster, 64GB RAM each)
         ├── Redis Cache (query embedding + result cache)
         └── LLM API (OpenAI, with request pool management)
```

### Decision 3 — Multi-Layer Caching

Customer support queries are highly repetitive. "How do I reset my password?" is asked thousands of times. Caching at multiple levels dramatically reduces cost and latency.

**Layer 1: Exact query cache (Redis TTL=1h)**

```python
import hashlib
import json
import redis.asyncio as aioredis

redis = aioredis.from_url("redis://redis:6379")

async def get_or_compute_answer(query: str, user_context: dict) -> dict:
    """
    Check exact match cache before running the full pipeline.
    """
    # Cache key: hash of query + relevant context (not user-specific context)
    cache_key = hashlib.sha256(
        f"{query.lower().strip()}:{user_context.get('product_tier', 'standard')}".encode()
    ).hexdigest()
    
    cached = await redis.get(f"answer:{cache_key}")
    if cached:
        result = json.loads(cached)
        result["from_cache"] = True
        return result
    
    # Cache miss — run full pipeline
    result = await run_rag_pipeline(query, user_context)
    
    # Cache the result (only for high-confidence non-IDK answers)
    if result.get("confidence", 0) > 0.85 and not result.get("is_idk"):
        await redis.setex(
            f"answer:{cache_key}",
            3600,  # 1 hour TTL
            json.dumps(result)
        )
    
    return result
```

**Layer 2: Semantic similarity cache (vector similarity)**

```python
async def check_semantic_cache(
    query: str,
    query_embedding: list[float],
    cache_collection: str = "query_cache",
    similarity_threshold: float = 0.92
) -> dict | None:
    """
    Find semantically similar cached queries and return their cached answers.
    Threshold of 0.92 means very similar queries (e.g., minor wording differences).
    """
    results = await qdrant.search(
        collection_name=cache_collection,
        query_vector=query_embedding,
        limit=1,
        score_threshold=similarity_threshold
    )
    
    if results and results[0].score >= similarity_threshold:
        cached_answer = results[0].payload.get("cached_answer")
        if cached_answer:
            return {**cached_answer, "from_semantic_cache": True, "cache_score": results[0].score}
    
    return None

async def store_in_semantic_cache(
    query: str,
    query_embedding: list[float],
    answer: dict
):
    """Store a high-quality answer in the semantic cache."""
    cache_id = str(uuid.uuid4())
    
    await qdrant.upsert(
        collection_name="query_cache",
        points=[{
            "id": cache_id,
            "vector": query_embedding,
            "payload": {
                "original_query": query,
                "cached_answer": answer,
                "cached_at": datetime.utcnow().isoformat(),
                "cache_hits": 0
            }
        }]
    )
```

**Layer 3: Query embedding cache**

```python
# The most expensive part per query is embedding generation
# Cache query embeddings in Redis (TTL=24h)

async def get_query_embedding_cached(query: str) -> list[float]:
    cache_key = f"qemb:{hashlib.md5(query.encode()).hexdigest()}"
    cached = await redis.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    embedding = await embedding_server.embed(query)
    await redis.setex(cache_key, 86400, json.dumps(embedding))
    return embedding
```

**Cache hit rates in production:**
- Exact cache: ~35% hit rate (very common questions).
- Semantic cache (threshold 0.92): additional ~20% hit rate.
- Embedding cache: ~45% hit rate across a day.

Net effect: ~55% of queries are fully served from cache. Cost reduction ~55%, latency for cached queries < 100ms.

### Decision 4 — Freshness: Near-Real-Time for Release Notes

Product releases break existing answers. If the system answers "click the Settings gear icon" but the UI was redesigned last week, users get confused.

**Trigger-based immediate re-indexing:**

```python
# Release notes webhook handler
async def handle_release_note_webhook(event: dict):
    """
    Called when a new release note is published.
    Immediately re-indexes the document AND invalidates related caches.
    """
    doc_id = event["document_id"]
    doc_url = event["document_url"]
    affected_features = event.get("affected_features", [])
    
    # Step 1: Re-index the new release note immediately (bypass normal queue)
    await index_document_urgent(doc_url, doc_id, priority="high")
    
    # Step 2: Invalidate cache entries related to affected features
    for feature in affected_features:
        # Find and delete cached answers about this feature
        await invalidate_cache_by_topic(feature)
    
    # Step 3: Mark related old knowledge base articles for review
    related_kb_articles = await find_related_articles(affected_features)
    for article_id in related_kb_articles:
        await registry.flag_for_review(article_id, reason=f"New release: {doc_id}")
    
    # Step 4: Alert content team if articles may need updating
    if len(related_kb_articles) > 0:
        await notify_content_team(
            f"Release {doc_id} may affect {len(related_kb_articles)} KB articles"
        )

async def invalidate_cache_by_topic(topic: str):
    """
    Invalidate all cached answers that mention a topic.
    Uses Redis pattern matching.
    """
    # Exact cache: scan and delete entries whose queries mention the topic
    # (This is an approximation — full scan is expensive at scale)
    # Better: store topic tags alongside cache entries and delete by tag
    
    pattern = f"answer:*"  # In production, tag caches by topic at write time
    keys = await redis.keys(pattern)
    
    # For each cached entry, check if it's related to this topic
    # (simplified — in production use topic-tagged cache entries)
    for key in keys:
        cached = await redis.get(key)
        if cached and topic.lower() in json.loads(cached).get("query", "").lower():
            await redis.delete(key)
```

### Decision 5 — Confidence Scoring and Escalation

The system must know when to escalate to a human agent. Routing a low-confidence answer to a customer is worse than routing it to a human.

```python
async def compute_answer_confidence(
    query: str,
    retrieved_chunks: list[dict],
    generated_answer: str,
    llm_client
) -> dict:
    """
    Multi-signal confidence score for escalation decisions.
    """
    
    signals = {}
    
    # Signal 1: Retrieval confidence (top rerank score)
    if retrieved_chunks:
        signals["retrieval_score"] = retrieved_chunks[0].get("rerank_score", 0)
    else:
        signals["retrieval_score"] = 0
    
    # Signal 2: Answer contains IDK language
    idk_phrases = [
        "i don't have information", "not covered in", "cannot find",
        "please contact support", "i'm not sure", "you may want to check"
    ]
    signals["idk_language"] = any(
        phrase in generated_answer.lower() for phrase in idk_phrases
    )
    
    # Signal 3: Answer length (very short answers are often IDK or low-quality)
    word_count = len(generated_answer.split())
    signals["adequate_length"] = word_count >= 30
    
    # Signal 4: Query matches known escalation triggers
    escalation_keywords = [
        "billing", "invoice", "charge", "refund", "cancel",
        "data loss", "breach", "outage", "urgent", "critical",
        "legal", "compliance", "gdpr", "contract"
    ]
    signals["escalation_keyword"] = any(
        kw in query.lower() for kw in escalation_keywords
    )
    
    # Signal 5: LLM self-assessed confidence (fast check)
    confidence_response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate your confidence in this answer on a scale 1-5.
Query: {query}
Answer: {generated_answer[:200]}
Retrieved context quality: {'good' if signals['retrieval_score'] > 0.6 else 'poor'}

Score (1=very uncertain, 5=very confident): (respond with just the number)"""
        }],
        max_tokens=5,
        temperature=0.0
    )
    
    try:
        llm_confidence = int(confidence_response.choices[0].message.content.strip()) / 5.0
    except ValueError:
        llm_confidence = 0.5
    
    signals["llm_self_confidence"] = llm_confidence
    
    # Compute overall confidence score
    if signals["escalation_keyword"]:
        # Always escalate for billing, legal, data loss
        overall_confidence = 0.0
    elif signals["idk_language"]:
        overall_confidence = 0.2
    else:
        overall_confidence = (
            signals["retrieval_score"] * 0.4 +
            signals["llm_self_confidence"] * 0.4 +
            (0.1 if signals["adequate_length"] else 0.0) +
            0.1  # Base confidence
        )
    
    return {
        "confidence": overall_confidence,
        "signals": signals,
        "should_escalate": overall_confidence < 0.6,
        "escalation_reason": (
            "Escalation keyword detected" if signals["escalation_keyword"]
            else "Low retrieval confidence" if signals["retrieval_score"] < 0.4
            else "LLM uncertainty" if llm_confidence < 0.5
            else None
        )
    }


async def handle_escalation(
    query: str,
    answer: str,
    confidence_result: dict,
    session_context: dict
) -> dict:
    """
    Route to human agent with full context for takeover.
    """
    
    escalation_package = {
        "customer_query": query,
        "ai_attempted_answer": answer,
        "confidence_score": confidence_result["confidence"],
        "escalation_reason": confidence_result["escalation_reason"],
        "session_id": session_context["session_id"],
        "customer_id": session_context["customer_id"],
        "product_tier": session_context["product_tier"],
        "previous_interactions": session_context.get("history", [])[-5:],
        "retrieved_context": [
            {
                "title": c.get("metadata", {}).get("doc_title"),
                "section": c.get("metadata", {}).get("section"),
                "text_preview": c.get("text", "")[:300]
            }
            for c in session_context.get("last_retrieved_chunks", [])
        ]
    }
    
    # Route to support queue
    ticket_id = await create_support_ticket(escalation_package)
    
    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "user_message": (
            "I've connected you with a support specialist who can help with this. "
            f"Your ticket ID is {ticket_id}. Expected response: 2-4 hours."
        )
    }
```

### Decision 6 — Feedback Loop: Resolved Tickets → Training Data

Every resolved ticket is a potential training signal. When a human agent provides the correct answer, that (query, correct_answer, retrieved_context) triple becomes a fine-tuning candidate.

```python
async def process_resolved_ticket(ticket: dict):
    """
    When a human agent resolves a ticket, extract training signal.
    """
    
    human_answer = ticket["agent_resolution"]
    original_query = ticket["customer_query"]
    ai_attempted_answer = ticket.get("ai_attempted_answer")
    retrieved_context = ticket.get("retrieved_context", [])
    
    # Case 1: AI gave wrong answer — agent correction is a negative example
    if ai_attempted_answer and ticket.get("ai_answer_wrong"):
        await store_negative_example({
            "query": original_query,
            "wrong_answer": ai_attempted_answer,
            "correct_answer": human_answer,
            "correct_context": retrieved_context
        })
    
    # Case 2: AI said IDK but agent found the answer — retrieval gap
    if ticket.get("ai_said_idk") and human_answer:
        # The agent knows where the answer came from
        source_article = ticket.get("agent_source_article")
        
        if source_article:
            # Check if this article is indexed
            indexed = await check_if_indexed(source_article)
            
            if not indexed:
                # Indexing gap — trigger indexing of this article
                await trigger_indexing(source_article)
            else:
                # Article indexed but retrieval missed it — potential embedding issue
                await log_retrieval_miss({
                    "query": original_query,
                    "correct_chunk_id": await find_chunk_for_article(source_article),
                    "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_context]
                })
    
    # Case 3: Successful answer — positive training signal
    if ticket.get("customer_satisfied") and not ticket.get("ai_answer_wrong"):
        await store_positive_example({
            "query": original_query,
            "good_answer": human_answer,
            "supporting_context": retrieved_context
        })
    
    # Weekly: use accumulated training signals to fine-tune embedding model
    # (Batched, not per-ticket)
```

**Weekly embedding model fine-tuning cycle:**

```python
async def weekly_fine_tuning_cycle():
    """
    Use accumulated feedback to improve the embedding model.
    Runs every Sunday night when traffic is lowest.
    """
    
    # Gather fine-tuning data from the week
    positive_pairs = await get_positive_training_examples(days=7)
    retrieval_misses = await get_retrieval_misses(days=7)
    
    if len(positive_pairs) < 500:
        print(f"Insufficient training data ({len(positive_pairs)} pairs) — skipping fine-tuning")
        return
    
    # Convert to fine-tuning format
    training_data = [
        InputExample(
            texts=[pair["query"], pair["supporting_context"][0]["text"] if pair["supporting_context"] else pair["good_answer"]]
        )
        for pair in positive_pairs
    ]
    
    # Fine-tune with current model as base
    await fine_tune_embedding_model(
        base_model="current_production_model",
        training_data=training_data,
        output_path="./models/weekly-finetuned"
    )
    
    # Evaluate on hold-out eval set
    evaluation = await evaluate_model("./models/weekly-finetuned")
    
    if evaluation["recall@10"] > current_model_recall + 0.01:
        # Improvement threshold met — re-embed corpus and deploy
        await trigger_corpus_reembedding("./models/weekly-finetuned")
        print(f"Fine-tuned model deployed. Recall: {evaluation['recall@10']:.3f}")
    else:
        print(f"Fine-tuned model did not improve enough. Skipping deployment.")
```

---

## Product-Specific Customization

Support chatbots must know the customer's context: what product tier they are on, what features they have access to, what version they are running.

```python
async def enrich_query_with_product_context(
    query: str,
    customer_id: str,
    crm_client
) -> dict:
    """
    Fetch customer context from CRM to enable personalized retrieval.
    """
    customer = await crm_client.get_customer(customer_id)
    
    product_context = {
        "plan": customer["subscription_plan"],          # "starter", "pro", "enterprise"
        "product_version": customer["product_version"], # "3.2.1"
        "enabled_features": customer["enabled_features"],
        "region": customer["region"],                   # Affects feature availability
        "account_age_months": customer["account_age_months"]
    }
    
    # Build metadata filters that respect customer's plan
    plan_filter = {
        "must": [
            {"key": "document_status", "match": {"value": "active"}},
            {
                "should": [
                    {"key": "applicable_plans", "match": {"any": [product_context["plan"], "all"]}},
                    {"key": "applicable_plans", "is_empty": True}
                ]
            }
        ]
    }
    
    # Enrich the query with context for better retrieval
    enriched_query = f"[{product_context['plan']} plan, v{product_context['product_version']}] {query}"
    
    return {
        "enriched_query": enriched_query,
        "metadata_filter": plan_filter,
        "product_context": product_context
    }
```

---

## Operational Metrics Dashboard

```python
SUPPORT_RAG_METRICS = {
    # Quality
    "ai_resolution_rate": "% of tickets fully resolved by AI without escalation",
    "thumbs_up_rate": "% of AI answers rated positively by customers",
    "escalation_rate": "% of queries escalated to human agents",
    "false_escalation_rate": "% of escalations where AI could have answered correctly",
    
    # Freshness
    "release_note_indexing_lag_minutes": "Time from release note publish to searchable",
    "kb_article_staleness_rate": "% of KB articles not updated after related release",
    
    # Efficiency
    "cache_hit_rate": "% of queries served from cache",
    "avg_cost_per_query_usd": "LLM + embedding cost per query",
    "p95_latency_ms": "95th percentile end-to-end latency",
    
    # Business
    "tickets_deflected_daily": "Tickets resolved by AI (not routed to human)",
    "avg_time_to_resolution_ai": "Average seconds for AI resolution",
    "avg_time_to_resolution_human": "Average hours for human resolution"
}
```

---

## Results After 6 Months

| Metric | Before (Human-Only) | After (AI-First) |
|---|---|---|
| Tier 1 resolution time | 4-8 hours | 2 seconds |
| Tier 1 CSAT | 72% | 81% |
| AI resolution rate | N/A | 68% |
| Escalation rate | 100% | 32% |
| Cost per resolved ticket | $18 | $4.20 |
| Cache hit rate | N/A | 54% |
| p95 latency | N/A | 2.6s |

---

## Lessons Learned

**Lesson 1:** Caching is the highest-ROI optimization for support RAG. The top 200 questions account for 60% of ticket volume. Exact and semantic caching served more than half of production traffic within 6 months.

**Lesson 2:** Escalation keyword detection must be maintained as a list by the product and legal teams, not hardcoded by engineers. Billing disputes, data deletion requests, and SLA breach claims emerged as escalation triggers only after going live.

**Lesson 3:** The feedback loop from resolved tickets is slow to activate. It takes 3-4 months to accumulate enough fine-tuning data for a meaningful embedding model update. Start collecting training data on day one even if fine-tuning happens later.

**Lesson 4:** Release note indexing lag caused the most user-visible failures post-launch. The webhook-based system reduced lag from hours to minutes, but the critical missing piece was KB article invalidation — old articles that now gave wrong advice due to UI changes kept getting retrieved.

---

## Interview Questions This Case Study Prepares You For

**"How do you design a RAG system for high QPS?"**
Answer: Multi-layer caching (exact, semantic, embedding), GPU batch embedding server, async Qdrant client with gRPC, horizontal pod scaling, strict latency budget per component, streaming responses. Cache hit rate is the most impactful lever.

**"How do you decide when to escalate to a human?"**
Answer: Multi-signal confidence score: retrieval score (cross-encoder), IDK language detection, answer length check, LLM self-assessed confidence, and hard-coded escalation keywords for billing/legal/data topics. Escalation packages the full context (query, AI answer, retrieved sources) for human agent takeover.

**"How do you close the feedback loop in a RAG support system?"**
Answer: Resolved tickets feed three pipelines: (1) negative examples from wrong AI answers, (2) retrieval miss logging when IDK was wrong (triggers indexing of missed articles), (3) positive examples for embedding fine-tuning. Weekly fine-tuning cycle uses accumulated positives to improve the embedding model.

**"How do you keep a support knowledge base fresh?"**
Answer: Webhook-triggered immediate re-indexing for release notes (15-minute SLA). Content team notified when new releases may affect existing KB articles. Cache invalidation by topic on re-index. Staleness monitoring alerts when source documents change but index does not.