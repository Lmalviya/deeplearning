# Lesson 9.5 — Debugging in Production at Scale: Distributed Tracing, Log Aggregation, and Alerting

---

## The Scale Debugging Problem

At 100 queries/day, debugging a failure means looking at logs and quickly finding the relevant entry. At 1,000,000 queries/day, the same approach produces 1GB+ of logs per day. A single failing query is a needle in a haystack. The failure may have happened hours ago, across 5 different services, with the relevant context split between three different log files on different pods.

Production debugging at scale requires purpose-built tooling: structured logging, distributed tracing, and queryable observability that lets you find the exact failure context without reading through raw log files.

---

## Structured Logging

The foundation of production debugging. Every log line must be machine-parseable JSON, not free-form text.

### Why Structured Logging

```
# Bad: Free-form text
logger.info(f"Retrieved {len(chunks)} chunks for query '{query}' in {latency}ms")

# Good: Structured JSON
logger.info("retrieval_complete", extra={
    "event": "retrieval_complete",
    "trace_id": trace_id,
    "query_hash": hashlib.md5(query.encode()).hexdigest()[:8],  # Don't log PII
    "chunk_count": len(chunks),
    "latency_ms": latency,
    "top_score": chunks[0].get("rerank_score", 0) if chunks else 0,
    "service": "rag-api",
    "pod_id": os.environ.get("POD_NAME"),
    "pipeline_version": PIPELINE_VERSION
})
```

With structured logging, you can query: "show me all retrieval operations from the last hour where chunk_count < 3 AND top_score < 0.4" — impossible with free-form text.

### Logging Configuration

```python
# src/logging_config.py
import logging
import json
import time
import os

class StructuredJSONFormatter(logging.Formatter):
    """Format all log records as JSON for Elasticsearch/CloudWatch ingestion."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.environ.get("SERVICE_NAME", "rag-api"),
            "pod_id": os.environ.get("POD_NAME", "local"),
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "pipeline_version": os.environ.get("PIPELINE_VERSION", "unknown")
        }
        
        # Add any extra fields from logger.info("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in ("message", "levelname", "name", "msg", "args",
                          "created", "filename", "funcName", "levelno", "lineno",
                          "module", "msecs", "pathname", "process", "processName",
                          "relativeCreated", "stack_info", "thread", "threadName",
                          "exc_info", "exc_text"):
                log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO"):
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJSONFormatter())
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
```

---

## Distributed Tracing

A trace links all the log events from a single user request across multiple services. Without tracing, you know a request failed but not which service caused it or why.

### Implementing Trace Context Propagation

Every request gets a `trace_id`. This ID is passed to every downstream service call and included in every log entry.

```python
# src/middleware/tracing.py
import uuid
from fastapi import Request
from contextvars import ContextVar

# Thread-local (actually coroutine-local) trace context
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")

def get_trace_id() -> str:
    return _trace_id.get()

def get_session_id() -> str:
    return _session_id.get()


async def tracing_middleware(request: Request, call_next):
    """
    Assign or propagate trace ID for every request.
    """
    # Use trace ID from upstream (if called by another service) or generate new
    trace_id = (
        request.headers.get("X-Trace-Id") or
        request.headers.get("X-Request-Id") or
        str(uuid.uuid4())
    )
    
    session_id = request.headers.get("X-Session-Id", "no-session")
    
    _trace_id.set(trace_id)
    _session_id.set(session_id)
    
    response = await call_next(request)
    
    # Return trace ID in response headers for client-side tracing
    response.headers["X-Trace-Id"] = trace_id
    
    return response


# Use in logging
def log_with_trace(logger, level: str, event: str, **kwargs):
    """Helper to always include trace context in log entries."""
    getattr(logger, level)(event, extra={
        "trace_id": get_trace_id(),
        "session_id": get_session_id(),
        **kwargs
    })
```

### Propagating Trace ID to Downstream Services

```python
# When calling the embedding server
async def embed_with_trace(query: str) -> list[float]:
    response = await httpx.post(
        f"{EMBEDDING_SERVER_URL}/embed",
        json={"text": query},
        headers={
            "X-Trace-Id": get_trace_id(),  # Forward trace context
            "X-Session-Id": get_session_id()
        }
    )
    return response.json()["embedding"]


# When calling Qdrant (via custom wrapper)
async def search_with_trace(query_vector: list[float], **kwargs) -> list:
    start = time.perf_counter()
    
    results = await qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        **kwargs
    )
    
    latency = (time.perf_counter() - start) * 1000
    
    log_with_trace(logger, "info", "qdrant_search_complete",
        latency_ms=latency,
        result_count=len(results),
        top_score=results[0].score if results else None
    )
    
    return results
```

---

## Log Aggregation

Individual pod logs need to flow to a central store where they can be searched and correlated.

### AWS CloudWatch Logs

The simplest option on AWS: logs from each container automatically go to CloudWatch Logs.

```python
# ECS task definition log configuration (from Lesson 8.4)
"logConfiguration": {
    "logDriver": "awslogs",
    "options": {
        "awslogs-group": "/rag/production",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
    }
}
```

**CloudWatch Logs Insights queries:**

```sql
-- Find all log entries for a specific trace
fields @timestamp, service, event, latency_ms, error
| filter trace_id = "abc123-def456"
| sort @timestamp asc

-- Find slow queries in the last hour
fields @timestamp, query_hash, latency_ms, top_score, chunk_count
| filter event = "retrieval_complete"
| filter latency_ms > 3000
| sort latency_ms desc
| limit 50

-- Find high error rate in last 15 minutes
stats count(*) as total, 
      sum(level = "ERROR") as errors
by bin(1m) as minute
| filter @timestamp > now() - 15m
| sort minute desc

-- Find queries where retrieval likely failed (low chunk count + low scores)
fields @timestamp, trace_id, query_hash, chunk_count, top_score
| filter event = "retrieval_complete"
| filter chunk_count < 3 or top_score < 0.3
| sort @timestamp desc
| limit 100
```

### Elasticsearch + Kibana (for more complex queries)

For teams with higher log volume or more complex querying needs:

```python
# Elasticsearch log shipping via Filebeat or Fluentd
# Configure Filebeat in DaemonSet on Kubernetes nodes
# It reads container logs and ships to Elasticsearch

# Then query in Kibana or via API:
from elasticsearch import Elasticsearch

es = Elasticsearch("https://elasticsearch:9200")

def find_failure_trace(trace_id: str) -> list[dict]:
    """Find all log events for a failing trace."""
    
    response = es.search(
        index="rag-logs-*",
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"trace_id": trace_id}},
                        {"range": {
                            "@timestamp": {
                                "gte": "now-24h"
                            }
                        }}
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "asc"}}],
            "size": 100
        }
    )
    
    return [hit["_source"] for hit in response["hits"]["hits"]]
```

---

## Alerting That Actually Works

Alerts should be actionable. An alert that fires 50 times/day and is always investigated to find no real issue is worse than no alert — it creates alert fatigue and gets ignored.

### Three Alert Tiers

```python
ALERT_TIERS = {
    "critical": {
        "channel": "pagerduty",  # Wakes someone up
        "conditions": [
            "error_rate > 5% for 5 minutes",
            "p99_latency > 10s for 5 minutes",
            "qdrant_health_check failing",
            "embedding_server_health_check failing",
            "dlq_depth > 100"
        ]
    },
    "warning": {
        "channel": "slack #rag-alerts",  # Needs attention but not urgent
        "conditions": [
            "p95_latency > 3s for 10 minutes",
            "error_rate > 1% for 10 minutes",
            "cache_hit_rate < 20% for 30 minutes",
            "idk_rate spike > 2x baseline for 30 minutes",
            "daily_cost > 150% of 7-day average"
        ]
    },
    "info": {
        "channel": "slack #rag-monitoring",  # FYI
        "conditions": [
            "new_user_record_qps",
            "index_rebuild_completed",
            "embedding_model_updated"
        ]
    }
}
```

### Implementing Smart Alerts

Alerts based on absolute thresholds (p95 > 3s) fire during expected traffic spikes even when the system is healthy. Use relative baselines to reduce false positives.

```python
class SmartAlerter:
    def __init__(self, metrics_store, notification_client):
        self.metrics = metrics_store
        self.notifier = notification_client
    
    async def check_latency_alert(self) -> None:
        """
        Alert on latency anomalies relative to historical baseline.
        Avoids false positives during known traffic spikes.
        """
        
        # Current metric (5-minute window)
        current_p95 = await self.metrics.get_p95_latency(minutes=5)
        
        # Historical baseline (same hour, last 7 days)
        baseline_p95 = await self.metrics.get_historical_p95_latency(
            hour_of_day=datetime.utcnow().hour,
            days_back=7
        )
        
        # Alert only if current is significantly above historical
        if baseline_p95 > 0:
            ratio = current_p95 / baseline_p95
            
            if ratio > 2.0 and current_p95 > 1000:  # 2× worse AND above 1s absolute
                await self.notifier.send_warning(
                    f"Latency anomaly: p95={current_p95:.0f}ms ({ratio:.1f}× historical baseline of {baseline_p95:.0f}ms)"
                )
    
    async def check_quality_degradation(self) -> None:
        """
        Alert when answer quality metrics drop (using sampled LLM evaluation).
        """
        
        # Run sampled evaluation every hour
        recent_faithfulness = await compute_sampled_faithfulness(
            sample_size=50,
            window_minutes=60
        )
        
        # Compare to 7-day baseline
        baseline_faithfulness = await self.metrics.get_baseline_faithfulness(days=7)
        
        if recent_faithfulness < baseline_faithfulness * 0.90:
            await self.notifier.send_critical(
                f"Quality degradation detected:\n"
                f"Faithfulness: {recent_faithfulness:.2f} (baseline: {baseline_faithfulness:.2f})\n"
                f"Action: Check for embedding model drift or corpus conflicts"
            )
```

---

## Production Debug Playbook

When a specific failure is reported, follow this playbook:

```python
async def production_debug_session(trace_id: str) -> dict:
    """
    Full production debug for a specific failing trace.
    """
    
    # Step 1: Retrieve full trace from log store
    trace_events = await log_store.find_trace(trace_id)
    
    if not trace_events:
        return {"error": "Trace not found — may have been purged or trace_id incorrect"}
    
    # Step 2: Reconstruct timeline
    timeline = sorted(trace_events, key=lambda e: e["timestamp"])
    
    # Step 3: Identify the first error
    first_error = next(
        (e for e in timeline if e.get("level") == "ERROR"),
        None
    )
    
    # Step 4: Identify stage where failure occurred
    stages_completed = [e.get("event") for e in timeline if e.get("event")]
    
    stage_sequence = [
        "query_received",
        "query_understanding_complete",
        "retrieval_complete",
        "reranking_complete",
        "context_assembly_complete",
        "generation_complete"
    ]
    
    last_completed = None
    for stage in stage_sequence:
        if stage in stages_completed:
            last_completed = stage
    
    # Step 5: Extract key metrics from trace
    retrieval_event = next(
        (e for e in timeline if e.get("event") == "retrieval_complete"),
        None
    )
    
    generation_event = next(
        (e for e in timeline if e.get("event") == "generation_complete"),
        None
    )
    
    return {
        "trace_id": trace_id,
        "timeline": timeline,
        "first_error": first_error,
        "last_successful_stage": last_completed,
        "failure_stage": stage_sequence[stage_sequence.index(last_completed) + 1] if last_completed else "query_understanding",
        "retrieval_metrics": {
            "chunk_count": retrieval_event.get("chunk_count") if retrieval_event else None,
            "top_score": retrieval_event.get("top_score") if retrieval_event else None,
            "latency_ms": retrieval_event.get("latency_ms") if retrieval_event else None
        },
        "generation_metrics": {
            "input_tokens": generation_event.get("input_tokens") if generation_event else None,
            "output_tokens": generation_event.get("output_tokens") if generation_event else None,
            "latency_ms": generation_event.get("latency_ms") if generation_event else None
        }
    }
```

---

## Log Retention and Cost

Logs are expensive to store at scale. Design retention policies from day one:

```python
LOG_RETENTION_POLICY = {
    # Raw application logs
    "rag_api_logs": {
        "hot_retention_days": 7,      # In CloudWatch/ES — fast query
        "cold_retention_days": 90,    # In S3 Glacier — cheap archival
        "delete_after_days": 365
    },
    
    # RAG-specific traces (more valuable, keep longer)
    "rag_traces": {
        "hot_retention_days": 30,
        "cold_retention_days": 365,
        "delete_after_days": 730       # 2 years for compliance-relevant traces
    },
    
    # Audit logs (immutable, keep forever for regulated industries)
    "audit_logs": {
        "hot_retention_days": 90,
        "cold_retention_days": "forever",
        "delete_after_days": None      # Never delete
    }
}
```

---

## Summary

- Structured JSON logging is the foundation. Every log line must include: trace_id, session_id, service name, pod ID, event type, and relevant numeric metrics.
- Trace context (trace_id) must be propagated to every downstream service call and included in every log entry. Without it, debugging cross-service failures is impossible.
- Three alert tiers: critical (PagerDuty, wake someone up), warning (Slack, needs attention), info (FYI). Alert on anomalies relative to historical baseline, not just absolute thresholds — reduces alert fatigue.
- The production debug playbook: retrieve full trace → sort by timestamp → find first error → identify last successful stage → extract key metrics at each stage.
- Log retention policy: hot storage for 7-30 days (fast queries), cold archival for 90-365 days (S3 Glacier), forever for audit logs in regulated industries.
- Tools: CloudWatch Logs Insights for AWS-native simplicity, Elasticsearch+Kibana for complex log analysis, Datadog/Grafana for dashboards and alerting.

---

## What's Next

Lesson 9.6 covers security and access control — multi-tenant RAG, document-level permissions, PII handling, and the security considerations specific to RAG systems.