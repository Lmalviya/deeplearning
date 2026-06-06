# Lesson 7.5 — Tracing and Observability: Building a RAG Trace

---

## Why Observability Is Different for RAG

Traditional software observability asks: did the system work? CPU usage, memory, error rates, latency — these tell you whether the service is healthy.

RAG observability asks a harder question: did the system work correctly? A RAG pipeline can complete in 800ms with no errors, perfect uptime, and still give a completely wrong answer. Infrastructure metrics tell you nothing about answer quality.

This creates a gap: your monitoring system says everything is green while users are silently getting wrong answers. Without purpose-built RAG observability, you will not know until a user complains — or worse, until a wrong answer causes a consequential error.

RAG observability requires tracing every decision the system makes, not just whether it completed. Specifically:
- What query understanding decisions were made?
- What was retrieved, at what scores, from what sources?
- Which chunks made it into the final context?
- What did the LLM generate, and is it grounded in the context?

This lesson covers how to build this tracing infrastructure and what to do with the data.

---

## The RAG Trace Structure

A RAG trace is a structured record of every pipeline stage for a single query. It is the single most important debugging artifact in a production RAG system.

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class QueryUnderstandingTrace:
    original_query: str
    resolved_query: Optional[str]         # After conversational resolution
    rewritten_query: Optional[str]        # After rewriting
    expanded_queries: list[str]           # From query expansion
    sub_questions: list[str]              # From decomposition
    detected_entities: list[str]         # Extracted named entities
    applied_filters: dict                 # Metadata filters derived from query
    latency_ms: float

@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text_preview: str                     # First 200 chars for inspection
    dense_score: Optional[float]          # Cosine similarity score
    sparse_score: Optional[float]         # BM25 score
    rrf_score: Optional[float]           # After RRF fusion
    rerank_score: Optional[float]        # Cross-encoder score
    final_rank: int                       # Position in final context

@dataclass
class RetrievalTrace:
    query_used: str                       # Actual query sent to retrieval
    dense_top_k: list[RetrievedChunk]   # Top results from dense retrieval
    sparse_top_k: list[RetrievedChunk]  # Top results from sparse retrieval
    rrf_merged: list[RetrievedChunk]    # After RRF fusion
    reranked: list[RetrievedChunk]      # After cross-encoder re-ranking
    final_context_chunks: list[RetrievedChunk]  # What went to the LLM
    total_context_tokens: int
    latency_ms: float

@dataclass
class GenerationTrace:
    system_prompt_tokens: int
    context_tokens: int
    query_tokens: int
    total_input_tokens: int
    output_tokens: int
    raw_response: str
    model_used: str
    latency_ms: float
    finish_reason: str                   # "stop", "length", etc.

@dataclass
class QualityTrace:
    faithfulness_score: Optional[float]  # NLI or LLM-judged
    answer_relevancy: Optional[float]
    retrieved_relevant_count: int        # How many retrieved chunks were relevant
    idk_response: bool                   # Did system say IDK?
    citations_present: bool
    user_feedback: Optional[str]         # "thumbs_up", "thumbs_down", or None

@dataclass
class RAGTrace:
    # Identity
    trace_id: str
    session_id: str
    user_id: Optional[str]
    timestamp: datetime
    
    # Pipeline stages
    query_understanding: QueryUnderstandingTrace
    retrieval: RetrievalTrace
    generation: GenerationTrace
    quality: QualityTrace
    
    # Final output
    final_answer: str
    
    # Timing summary
    total_latency_ms: float
    stage_latencies: dict[str, float]
    
    # Metadata
    pipeline_version: str
    embedding_model: str
    llm_model: str
    experiment_assignments: dict         # A/B test assignments
```

---

## Instrumenting the Pipeline

Every pipeline stage must emit trace data. Here is how to instrument a complete pipeline:

```python
import time
import uuid

class TracedRAGPipeline:
    def __init__(
        self,
        retriever,
        llm_client,
        trace_store,
        embedding_model,
        reranker,
        pipeline_version: str = "1.0.0"
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.trace_store = trace_store
        self.embedder = embedding_model
        self.reranker = reranker
        self.version = pipeline_version
    
    async def answer(
        self,
        query: str,
        session_id: str,
        user_id: str = None,
        conversation_history: list = None
    ) -> dict:
        """
        Full pipeline with comprehensive tracing at every stage.
        """
        
        trace_id = str(uuid.uuid4())
        pipeline_start = time.perf_counter()
        stage_latencies = {}
        
        # ─────────────────────────────────────────────
        # Stage 1: Query Understanding
        # ─────────────────────────────────────────────
        stage_start = time.perf_counter()
        
        resolved_query = None
        rewritten_query = None
        expanded_queries = []
        sub_questions = []
        
        if conversation_history:
            resolved_query = await resolve_conversational_query(
                query, conversation_history, self.llm
            )
        
        primary_query = resolved_query or query
        
        if should_rewrite(primary_query):
            rewritten_query = await rewrite_query(primary_query, self.llm)
        
        expanded_queries = await expand_query(primary_query, self.llm, n=2)
        
        applied_filters = extract_metadata_filters(primary_query)
        
        qu_trace = QueryUnderstandingTrace(
            original_query=query,
            resolved_query=resolved_query,
            rewritten_query=rewritten_query,
            expanded_queries=expanded_queries,
            sub_questions=sub_questions,
            detected_entities=[],  # Would be populated by NER
            applied_filters=applied_filters,
            latency_ms=(time.perf_counter() - stage_start) * 1000
        )
        stage_latencies["query_understanding"] = qu_trace.latency_ms
        
        # ─────────────────────────────────────────────
        # Stage 2: Retrieval
        # ─────────────────────────────────────────────
        stage_start = time.perf_counter()
        
        retrieval_query = rewritten_query or primary_query
        
        # Dense retrieval
        query_embedding = await self.embedder.embed(retrieval_query)
        dense_results = await self.retriever.dense_search(
            query_embedding, k=50, filter=applied_filters
        )
        
        # Sparse retrieval
        sparse_results = await self.retriever.sparse_search(
            retrieval_query, k=50
        )
        
        # RRF fusion
        rrf_results = reciprocal_rank_fusion([dense_results, sparse_results])
        
        # Re-ranking
        reranked_results = self.reranker.rerank(retrieval_query, rrf_results[:50])
        
        # Final context selection
        final_chunks = reranked_results[:8]
        
        def to_retrieved_chunk(result: dict, rank: int) -> RetrievedChunk:
            return RetrievedChunk(
                chunk_id=result.get("chunk_id", ""),
                doc_id=result.get("metadata", {}).get("doc_id", ""),
                doc_title=result.get("metadata", {}).get("doc_title", ""),
                section=result.get("metadata", {}).get("heading_path", ""),
                text_preview=result.get("text", "")[:200],
                dense_score=result.get("dense_score"),
                sparse_score=result.get("sparse_score"),
                rrf_score=result.get("rrf_score"),
                rerank_score=result.get("rerank_score"),
                final_rank=rank
            )
        
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        context_text = "\n\n".join(c.get("text", "") for c in final_chunks)
        context_tokens = len(enc.encode(context_text))
        
        retrieval_trace = RetrievalTrace(
            query_used=retrieval_query,
            dense_top_k=[to_retrieved_chunk(r, i+1) for i, r in enumerate(dense_results[:10])],
            sparse_top_k=[to_retrieved_chunk(r, i+1) for i, r in enumerate(sparse_results[:10])],
            rrf_merged=[to_retrieved_chunk(r, i+1) for i, r in enumerate(rrf_results[:10])],
            reranked=[to_retrieved_chunk(r, i+1) for i, r in enumerate(reranked_results[:10])],
            final_context_chunks=[to_retrieved_chunk(r, i+1) for i, r in enumerate(final_chunks)],
            total_context_tokens=context_tokens,
            latency_ms=(time.perf_counter() - stage_start) * 1000
        )
        stage_latencies["retrieval"] = retrieval_trace.latency_ms
        
        # ─────────────────────────────────────────────
        # Stage 3: Generation
        # ─────────────────────────────────────────────
        stage_start = time.perf_counter()
        
        formatted_context = format_context_with_sources(final_chunks)
        system_prompt = build_system_prompt()
        user_message = f"Context:\n{formatted_context}\n\nQuestion: {query}"
        
        sys_tokens = len(enc.encode(system_prompt))
        query_tokens = len(enc.encode(user_message))
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=800,
            temperature=0.1
        )
        
        raw_answer = response.choices[0].message.content
        output_tokens = response.usage.completion_tokens
        
        gen_trace = GenerationTrace(
            system_prompt_tokens=sys_tokens,
            context_tokens=context_tokens,
            query_tokens=query_tokens,
            total_input_tokens=sys_tokens + context_tokens + query_tokens,
            output_tokens=output_tokens,
            raw_response=raw_answer,
            model_used="gpt-4o",
            latency_ms=(time.perf_counter() - stage_start) * 1000,
            finish_reason=response.choices[0].finish_reason
        )
        stage_latencies["generation"] = gen_trace.latency_ms
        
        # ─────────────────────────────────────────────
        # Stage 4: Quality Assessment (async, non-blocking)
        # ─────────────────────────────────────────────
        
        # Fast IDK detection
        idk_phrases = ["don't have information", "not in the provided", "cannot find"]
        is_idk = any(phrase in raw_answer.lower() for phrase in idk_phrases)
        
        quality_trace = QualityTrace(
            faithfulness_score=None,   # Filled in async
            answer_relevancy=None,     # Filled in async
            retrieved_relevant_count=0,  # Filled in async
            idk_response=is_idk,
            citations_present="[1]" in raw_answer or "[2]" in raw_answer,
            user_feedback=None         # Filled in later
        )
        
        # ─────────────────────────────────────────────
        # Assemble trace
        # ─────────────────────────────────────────────
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        
        trace = RAGTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            query_understanding=qu_trace,
            retrieval=retrieval_trace,
            generation=gen_trace,
            quality=quality_trace,
            final_answer=raw_answer,
            total_latency_ms=total_latency,
            stage_latencies=stage_latencies,
            pipeline_version=self.version,
            embedding_model=self.embedder.model_name,
            llm_model="gpt-4o",
            experiment_assignments={}
        )
        
        # Persist trace asynchronously (non-blocking)
        await self.trace_store.save(trace)
        
        # Schedule async quality assessment
        asyncio.create_task(
            self._compute_async_quality(trace_id, query, raw_answer, context_text)
        )
        
        return {
            "answer": raw_answer,
            "trace_id": trace_id,
            "sources": [
                {
                    "doc_title": c.doc_title,
                    "section": c.section,
                    "chunk_id": c.chunk_id
                }
                for c in retrieval_trace.final_context_chunks
            ]
        }
    
    async def _compute_async_quality(
        self,
        trace_id: str,
        query: str,
        answer: str,
        context: str
    ):
        """
        Compute quality metrics asynchronously after response is returned.
        Does not block the user from receiving their answer.
        """
        try:
            # NLI faithfulness check
            faithfulness = compute_nli_faithfulness(context, answer)
            
            # Update trace with quality metrics
            await self.trace_store.update_quality(trace_id, {
                "faithfulness_score": faithfulness
            })
        except Exception as e:
            log_error(f"Async quality computation failed for trace {trace_id}: {e}")
```

---

## Trace Storage and Querying

Traces must be stored in a way that enables both real-time alerting and historical analysis.

```python
class TraceStore:
    def __init__(self, db_client, retention_days: int = 30):
        self.db = db_client
        self.retention_days = retention_days
    
    async def save(self, trace: RAGTrace):
        """Persist trace to storage."""
        
        # Convert to flat document for storage
        doc = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "timestamp": trace.timestamp.isoformat(),
            "query": trace.query_understanding.original_query,
            "answer": trace.final_answer,
            "total_latency_ms": trace.total_latency_ms,
            "pipeline_version": trace.pipeline_version,
            
            # Query understanding
            "qu_resolved_query": trace.query_understanding.resolved_query,
            "qu_applied_filters": trace.query_understanding.applied_filters,
            "qu_latency_ms": trace.query_understanding.latency_ms,
            
            # Retrieval
            "ret_final_chunk_count": len(trace.retrieval.final_context_chunks),
            "ret_context_tokens": trace.retrieval.total_context_tokens,
            "ret_top_chunk_id": trace.retrieval.final_context_chunks[0].chunk_id if trace.retrieval.final_context_chunks else None,
            "ret_top_chunk_doc": trace.retrieval.final_context_chunks[0].doc_title if trace.retrieval.final_context_chunks else None,
            "ret_top_rerank_score": trace.retrieval.final_context_chunks[0].rerank_score if trace.retrieval.final_context_chunks else None,
            "ret_latency_ms": trace.retrieval.latency_ms,
            
            # Generation
            "gen_total_input_tokens": trace.generation.total_input_tokens,
            "gen_output_tokens": trace.generation.output_tokens,
            "gen_finish_reason": trace.generation.finish_reason,
            "gen_latency_ms": trace.generation.latency_ms,
            
            # Quality
            "is_idk_response": trace.quality.idk_response,
            "citations_present": trace.quality.citations_present,
            "faithfulness_score": trace.quality.faithfulness_score,
            "user_feedback": trace.quality.user_feedback,
            
            # Full trace for detailed debugging (stored as JSON)
            "_full_trace": trace  # Serialized to JSON
        }
        
        await self.db.insert("rag_traces", doc)
    
    async def query_traces(
        self,
        filters: dict = None,
        time_range: tuple = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Query traces with flexible filters for analysis.
        
        Example filters:
        - {"is_idk_response": True} — all IDK responses
        - {"user_feedback": "thumbs_down"} — all negative feedback
        - {"ret_top_rerank_score": {"$lt": 0.3}} — low confidence retrievals
        """
        return await self.db.query("rag_traces", filters, time_range, limit)
    
    async def get_trace(self, trace_id: str) -> dict:
        """Retrieve a specific trace for debugging."""
        return await self.db.get_by_id("rag_traces", trace_id)
    
    async def update_quality(self, trace_id: str, quality_updates: dict):
        """Update quality metrics after async computation."""
        await self.db.update("rag_traces", trace_id, quality_updates)
    
    async def attach_user_feedback(self, trace_id: str, feedback: str):
        """Attach user feedback to the trace."""
        await self.db.update("rag_traces", trace_id, {"user_feedback": feedback})
```

---

## Useful Trace Queries for Debugging

Once traces are stored, you can run powerful diagnostic queries:

```python
class TraceAnalyzer:
    def __init__(self, trace_store: TraceStore):
        self.store = trace_store
    
    async def find_low_confidence_retrievals(
        self,
        rerank_score_threshold: float = 0.3,
        hours: int = 24
    ) -> list[dict]:
        """Find queries where the top chunk had a low re-ranking score."""
        return await self.store.query_traces(
            filters={"ret_top_rerank_score": {"$lt": rerank_score_threshold}},
            time_range=("now-{}h".format(hours), "now"),
            limit=50
        )
    
    async def find_unfaithful_responses(
        self,
        faithfulness_threshold: float = 0.7
    ) -> list[dict]:
        """Find responses that may contain hallucinations."""
        return await self.store.query_traces(
            filters={"faithfulness_score": {"$lt": faithfulness_threshold}},
            limit=50
        )
    
    async def find_negative_feedback_traces(self) -> list[dict]:
        """Find all queries that received thumbs-down feedback."""
        return await self.store.query_traces(
            filters={"user_feedback": "thumbs_down"},
            limit=100
        )
    
    async def compute_stage_latency_breakdown(
        self,
        hours: int = 24
    ) -> dict:
        """Compute P50/P95/P99 latency for each pipeline stage."""
        traces = await self.store.query_traces(
            time_range=("now-{}h".format(hours), "now"),
            limit=10000
        )
        
        import numpy as np
        
        stages = ["query_understanding", "retrieval", "generation"]
        latency_breakdown = {}
        
        for stage in stages:
            field = f"{stage[:3]}_latency_ms"
            latencies = [t[field] for t in traces if t.get(field)]
            
            if latencies:
                latency_breakdown[stage] = {
                    "p50": float(np.percentile(latencies, 50)),
                    "p95": float(np.percentile(latencies, 95)),
                    "p99": float(np.percentile(latencies, 99)),
                    "mean": float(np.mean(latencies))
                }
        
        total_latencies = [t["total_latency_ms"] for t in traces if t.get("total_latency_ms")]
        latency_breakdown["total"] = {
            "p50": float(np.percentile(total_latencies, 50)),
            "p95": float(np.percentile(total_latencies, 95)),
            "p99": float(np.percentile(total_latencies, 99))
        }
        
        return latency_breakdown
    
    async def find_retrieval_failures(self) -> list[dict]:
        """
        Find queries where retrieval likely failed:
        - IDK responses with low-scoring top chunks
        - Low re-rank scores with negative feedback
        """
        idk_with_low_scores = await self.store.query_traces(
            filters={
                "is_idk_response": True,
                "ret_top_rerank_score": {"$lt": 0.4}
            },
            limit=50
        )
        
        negative_with_low_scores = await self.store.query_traces(
            filters={
                "user_feedback": "thumbs_down",
                "ret_top_rerank_score": {"$lt": 0.5}
            },
            limit=50
        )
        
        return {
            "idk_with_low_confidence": idk_with_low_scores,
            "negative_feedback_with_low_confidence": negative_with_low_scores,
            "total_likely_retrieval_failures": len(idk_with_low_scores) + len(negative_with_low_scores)
        }
```

---

## Integration with Standard Observability Tools

RAG traces should flow into your existing observability stack. Three common integration patterns:

### LangSmith / LangFuse (RAG-specific)

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="...",
    secret_key="...",
    host="https://cloud.langfuse.com"
)

def trace_with_langfuse(pipeline_func):
    """Decorator to trace RAG pipeline calls with LangFuse."""
    
    async def wrapper(query: str, session_id: str, **kwargs):
        trace = langfuse.trace(
            name="rag_pipeline",
            session_id=session_id,
            input={"query": query}
        )
        
        try:
            # Instrument each stage as a span
            with trace.span(name="query_understanding") as qu_span:
                qu_result = await run_query_understanding(query)
                qu_span.end(output=qu_result)
            
            with trace.span(name="retrieval") as ret_span:
                ret_result = await run_retrieval(qu_result["query"])
                ret_span.end(output={"n_chunks": len(ret_result)})
            
            with trace.span(name="generation") as gen_span:
                answer = await run_generation(query, ret_result)
                gen_span.end(output={"answer": answer})
            
            trace.update(output={"answer": answer})
            return {"answer": answer, "trace_url": trace.get_trace_url()}
        
        except Exception as e:
            trace.update(level="ERROR", status_message=str(e))
            raise
    
    return wrapper
```

### OpenTelemetry (Standard)

```python
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

tracer = otel_trace.get_tracer("rag_pipeline")

async def answer_with_otel_tracing(query: str) -> str:
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query.text", query[:200])
        
        with tracer.start_as_current_span("retrieval") as ret_span:
            chunks = await retrieve(query)
            ret_span.set_attribute("retrieval.chunk_count", len(chunks))
            ret_span.set_attribute("retrieval.top_score", 
                                  chunks[0].get("rerank_score", 0) if chunks else 0)
        
        with tracer.start_as_current_span("generation") as gen_span:
            answer = await generate(query, chunks)
            gen_span.set_attribute("generation.output_tokens", 
                                  len(answer.split()))
        
        span.set_attribute("is_idk_response", "don't have" in answer.lower())
        return answer
```

---

## Alerting on Trace Patterns

Set up alerts that fire on patterns in traces, not just on infrastructure metrics:

```python
class RAGAlertManager:
    def __init__(self, trace_store, alert_client):
        self.store = trace_store
        self.alerts = alert_client
    
    async def run_pattern_checks(self):
        """
        Check trace data for alarming patterns and fire alerts.
        Run every 5-15 minutes.
        """
        
        # 1. Sudden spike in IDK responses (may indicate index gap)
        recent_idk_rate = await self._compute_recent_metric(
            "is_idk_response", window_minutes=30
        )
        baseline_idk_rate = await self._compute_recent_metric(
            "is_idk_response", window_minutes=1440  # 24 hours
        )
        
        if recent_idk_rate > baseline_idk_rate * 1.5 and recent_idk_rate > 0.10:
            await self.alerts.fire(
                severity="medium",
                title="IDK rate spike detected",
                detail=f"IDK rate in last 30m: {recent_idk_rate:.1%}, baseline: {baseline_idk_rate:.1%}",
                action="Check for recent index issues or new query pattern"
            )
        
        # 2. Retrieval confidence dropping (top rerank scores going down)
        recent_avg_score = await self._compute_avg_metric(
            "ret_top_rerank_score", window_minutes=60
        )
        if recent_avg_score < 0.45:
            await self.alerts.fire(
                severity="high",
                title="Low retrieval confidence",
                detail=f"Average top rerank score: {recent_avg_score:.2f}",
                action="Check for query distribution shift or embedding drift"
            )
        
        # 3. Latency degradation
        p95_latency = await self._compute_percentile_metric(
            "total_latency_ms", percentile=95, window_minutes=30
        )
        if p95_latency > 4000:
            await self.alerts.fire(
                severity="high",
                title="P95 latency exceeds 4 seconds",
                detail=f"P95 latency: {p95_latency:.0f}ms",
                action="Check for LLM API slowness or retrieval index issues"
            )
        
        # 4. Consecutive negative feedback (from same session)
        negative_streaks = await self._find_negative_feedback_streaks(
            streak_length=3,
            window_minutes=60
        )
        if negative_streaks:
            await self.alerts.fire(
                severity="medium",
                title=f"{len(negative_streaks)} sessions with 3+ consecutive negative feedback",
                detail=str(negative_streaks[:3]),
                action="Investigate specific sessions for systematic failure patterns"
            )
```

---

## Building the Observability Stack: Tool Recommendations

| Layer | Purpose | Tools |
|---|---|---|
| **Trace storage** | Store and query RAG traces | PostgreSQL + JSONB, BigQuery, ClickHouse |
| **RAG-specific tracing** | Trace LLM calls with context | LangFuse, LangSmith, Phoenix (Arize) |
| **Distributed tracing** | Link to infrastructure traces | OpenTelemetry, Jaeger, Datadog APM |
| **Metrics and alerting** | Infrastructure + RAG quality metrics | Prometheus + Grafana, Datadog |
| **Log aggregation** | Query and search logs | Elasticsearch, Loki, CloudWatch Logs |
| **Error tracking** | Capture and alert on exceptions | Sentry, Rollbar |

For most teams, the minimal viable stack is: LangFuse for RAG-specific tracing + Grafana for dashboards + PagerDuty for alerting.

---

## Summary

- RAG observability requires tracing pipeline decisions, not just infrastructure health. A system with green infrastructure metrics can be delivering completely wrong answers.
- A RAG trace captures: query understanding decisions, retrieved chunks with per-stage scores, generation parameters and output, and quality signals.
- Instrument every pipeline stage with timing and output capture. Store traces asynchronously — do not block the user response for trace storage.
- Compute expensive quality metrics (faithfulness) asynchronously after the response is returned.
- Store traces in a queryable store and build analysis functions that surface patterns: low-confidence retrievals, unfaithful responses, latency bottlenecks, and negative feedback clusters.
- Integrate with standard observability tools (OpenTelemetry, LangFuse) to avoid building custom infrastructure.
- Alert on trace patterns, not just infrastructure metrics: IDK rate spikes, retrieval confidence drops, consecutive negative feedback, and latency outliers.

---

## What's Next

Lesson 7.6 covers the common failure patterns catalog — a reference of the most frequent RAG failure modes, their symptoms, root causes, and fixes — organized as a diagnostic playbook for your team.