# Lesson 9.1 — Scaling to Millions of Users: Architecture Patterns, Bottlenecks, and Load Testing

---

## What Changes at Scale

A RAG system handling 100 queries per day is architecturally simple. At 1 million queries per day (~12 QPS average, 100+ QPS at peak), every component that worked fine at small scale becomes a potential failure point.

The fundamental challenge is not speed — modern hardware can handle very high throughput. The challenge is **predictability under load**: maintaining quality, latency SLAs, and reliability when all components are simultaneously under stress.

This lesson maps the bottlenecks at each scale tier, the architectural patterns that address them, and how to validate that your system can actually handle the expected load.

---

## Scale Tiers and Their Bottlenecks

### Tier 1: < 1,000 queries/day (~1 QPS peak)

**Characteristics:** A single server handles everything. No special scaling needed.

**Typical setup:** One EC2 instance running the API, Qdrant, and Redis. Embedding model loaded in the API process.

**Bottlenecks at this tier:** None significant. Cost and simplicity are the primary concerns.

### Tier 2: 1,000–100,000 queries/day (~10 QPS peak)

**Characteristics:** Single-node vector database, stateless API with horizontal scaling, shared embedding service.

**First bottleneck: The embedding server.** At 10 QPS with 50ms per embedding, you need 0.5 GPU-seconds per second — one GPU handles this easily. But embedding is the first service that needs to be a separate, dedicated process to avoid resource contention with the API.

**Second bottleneck: Qdrant single-node throughput.** A single Qdrant node handles ~500-1000 search QPS comfortably with good configuration (HNSW, payload indexes). At 10 QPS, this is not a concern. At 100 QPS, start monitoring.

**Typical setup:**
```
[ALB] → [2-3 API pods] → [Embedding server (1 GPU)]
                       → [Qdrant (1 node, r6i.2xlarge)]
                       → [Redis (ElastiCache)]
                       → [OpenAI API]
```

### Tier 3: 100,000–1,000,000 queries/day (~100 QPS peak, 500 QPS burst)

**Characteristics:** This is where most RAG systems at maturity live. Multiple bottlenecks emerge simultaneously.

**Bottleneck 1: LLM API rate limits.** At 500 QPS with 2,000 tokens input and 300 tokens output, you need 500 × 2,300 = 1,150,000 tokens per second. OpenAI's default tier limits are far below this. You need enterprise rate limit agreements or self-hosted models.

**Bottleneck 2: Embedding server throughput.** At 500 QPS, 50ms per embedding = 25 GPU-seconds needed per second. One A10G GPU handles ~20 embeddings/second without batching, ~200/second with batching. You need 2-3 GPU nodes with the batch embedding server from Lesson 8.3.

**Bottleneck 3: Qdrant search throughput.** At 500 QPS with 50ms search time, you need ~25 concurrent searches. One Qdrant node handles this, but with no margin. Add read replicas (2-3 total) for headroom and fault tolerance.

**Bottleneck 4: Context window cost.** At 500 QPS with 2,000 token context, you are sending 1 billion tokens/day to the LLM. At $5/million input tokens, that is $5,000/day. Caching (Lesson 8.6) must be active by this point.

**Typical setup:**
```
[ALB] → [10-20 API pods (auto-scaled)]
           ├── [Embedding cluster (3× A10G, batched)]
           ├── [Qdrant cluster (3 nodes, read replicas)]
           ├── [Redis cluster (ElastiCache)]
           ├── [Re-ranker (2× A10G)]
           └── [LLM: OpenAI enterprise OR self-hosted vLLM]
```

### Tier 4: > 1,000,000 queries/day (1,000+ QPS peak)

**Characteristics:** Full distributed architecture. Every component must be independently scaled and fault-tolerant.

**Bottleneck 1: Monolithic vector index.** A single Qdrant collection (even with replicas) has practical limits on concurrent search. Shard the index by topic, region, or user segment.

**Bottleneck 2: Cold start latency.** At high QPS, API pod cold starts (container startup + model initialization) add latency spikes during scale-out events. Use pod pre-warming.

**Bottleneck 3: Network bandwidth.** Transferring large context windows (10,000+ tokens) between services adds up at high QPS. Context compression (Lesson 3.7) becomes a cost and performance necessity.

**Bottleneck 4: Database connections.** PostgreSQL (metadata registry) has connection limits. Use PgBouncer as a connection pooler.

---

## The Multi-Region Architecture

For global users, a single-region deployment creates unacceptable latency for distant users (300-500ms network overhead for users in Asia reaching US-East).

```
                         [Global Load Balancer / DNS]
                         (Route 53 Latency-Based Routing)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         [US-East-1]    [EU-West-1]    [AP-Southeast-1]
              │               │               │
         [RAG Stack]    [RAG Stack]    [RAG Stack]
         [Qdrant]       [Qdrant]       [Qdrant]
              │               │               │
              └───────────────┴───────────────┘
                              │
                    [Global Document Store]
                    (S3 + Cross-Region Replication)
                              │
                    [Index Sync Service]
                    (Replicate Qdrant index across regions)
```

**The index synchronization challenge:** When documents are indexed in US-East, they need to be available in EU-West and AP-Southeast within your freshness SLA. Options:

1. **Re-index in each region independently:** Each region runs its own indexing pipeline from the same S3 source. Simple but creates lag between regions.

2. **Primary region indexing + binary replication:** Index only in the primary region. Replicate the Qdrant binary snapshot to other regions on a schedule. Lag = snapshot schedule (e.g., hourly).

3. **Qdrant distributed cluster across regions:** Qdrant supports multi-datacenter deployment. Higher complexity but enables near-real-time consistency.

For most use cases, option 1 (independent regional indexing) is the pragmatic choice. Each region's indexing worker reads from S3 (which has cross-region replication enabled) and builds its own index.

---

## Load Testing: Validating Before Production

Never assume your system can handle the expected load. Load test everything before going live.

### Load Testing Framework

```python
# load_tests/locustfile.py
from locust import HttpUser, task, between
import json
import random

# Sample queries from your evaluation dataset
SAMPLE_QUERIES = [
    "What is the parental leave policy?",
    "How do I submit an expense report?",
    "What are the eligibility requirements for the 401k?",
    "What is the notice period for voluntary resignation?",
    # ... 200+ realistic queries
]

class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)  # Simulate human-like think time
    
    @task(10)  # Weight 10: most common query type
    def ask_common_question(self):
        query = random.choice(SAMPLE_QUERIES[:50])  # Most common 50
        self._ask(query)
    
    @task(3)   # Weight 3: less common
    def ask_uncommon_question(self):
        query = random.choice(SAMPLE_QUERIES[50:150])
        self._ask(query)
    
    @task(1)   # Weight 1: rare edge cases
    def ask_edge_case(self):
        query = random.choice(SAMPLE_QUERIES[150:])
        self._ask(query)
    
    def _ask(self, query: str):
        with self.client.post(
            "/query",
            json={
                "query": query,
                "user_context": {
                    "department": "engineering",
                    "region": "us"
                }
            },
            headers={"Authorization": "Bearer test-token"},
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
                return
            
            data = response.json()
            
            # Validate response structure (not just status code)
            if "answer" not in data:
                response.failure("Response missing 'answer' field")
                return
            
            if len(data["answer"]) < 10:
                response.failure("Answer too short — likely an error response")
                return
            
            response.success()
```

```bash
# Run load test: ramp from 0 to 500 users over 60 seconds, hold for 300 seconds
locust \
  -f load_tests/locustfile.py \
  --host https://rag-api.staging.yourdomain.com \
  --users 500 \
  --spawn-rate 10 \
  --run-time 360s \
  --headless \
  --csv results/load_test_500_users
```

### What to Measure During Load Tests

```python
# load_tests/metrics_collector.py
import boto3
import time

def collect_load_test_metrics(
    test_duration_seconds: int,
    cloudwatch_namespace: str = "RAGLoadTest"
) -> dict:
    """
    Collect performance metrics during load test from CloudWatch.
    """
    cw = boto3.client("cloudwatch")
    
    end_time = time.time()
    start_time = end_time - test_duration_seconds
    
    metrics_to_check = {
        "api_p95_latency_ms": {
            "name": "p95_latency",
            "stat": "p95",
            "threshold": 3000  # Must stay under 3 seconds
        },
        "api_error_rate": {
            "name": "error_rate",
            "stat": "Average",
            "threshold": 0.01  # Must stay under 1% error rate
        },
        "qdrant_search_latency_ms": {
            "name": "search_latency",
            "stat": "p95",
            "threshold": 100
        },
        "embedding_server_latency_ms": {
            "name": "embedding_latency",
            "stat": "p95",
            "threshold": 100
        },
        "llm_latency_ms": {
            "name": "llm_latency",
            "stat": "p95",
            "threshold": 3000
        },
        "cache_hit_rate": {
            "name": "cache_hit_rate",
            "stat": "Average",
            "threshold": 0.3  # At least 30% cache hit rate
        }
    }
    
    results = {}
    failures = []
    
    for metric_key, config in metrics_to_check.items():
        response = cw.get_metric_statistics(
            Namespace=cloudwatch_namespace,
            MetricName=config["name"],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=[config["stat"]]
        )
        
        if response["Datapoints"]:
            value = response["Datapoints"][0][config["stat"]]
            results[metric_key] = value
            
            # Check against threshold
            if "latency" in metric_key:
                if value > config["threshold"]:
                    failures.append(f"{metric_key}: {value:.0f}ms > threshold {config['threshold']}ms")
            else:
                if value < config["threshold"]:
                    failures.append(f"{metric_key}: {value:.3f} < threshold {config['threshold']}")
    
    return {
        "results": results,
        "failures": failures,
        "passed": len(failures) == 0
    }
```

### Load Testing Scenarios

Run these scenarios in order. Each one probes a different failure mode:

```python
LOAD_TEST_SCENARIOS = [
    {
        "name": "baseline",
        "description": "Normal traffic — 50% of expected peak",
        "users": 250,
        "spawn_rate": 10,
        "duration": "300s",
        "pass_criteria": {
            "p95_latency_ms": 2000,
            "error_rate": 0.001
        }
    },
    {
        "name": "peak_traffic",
        "description": "Expected peak — 100% of expected peak load",
        "users": 500,
        "spawn_rate": 20,
        "duration": "300s",
        "pass_criteria": {
            "p95_latency_ms": 3000,
            "error_rate": 0.01
        }
    },
    {
        "name": "burst",
        "description": "Sudden spike — 200% of normal",
        "users": 1000,
        "spawn_rate": 100,  # Fast ramp simulates sudden spike
        "duration": "120s",
        "pass_criteria": {
            "p95_latency_ms": 5000,  # Allow higher latency during burst
            "error_rate": 0.05       # Allow some errors during burst
        }
    },
    {
        "name": "sustained_peak",
        "description": "Monday morning — peak for extended duration",
        "users": 500,
        "spawn_rate": 10,
        "duration": "3600s",  # 1 hour
        "pass_criteria": {
            "p95_latency_ms": 3000,
            "error_rate": 0.01,
            "memory_leak_check": True  # Memory should not grow over time
        }
    }
]
```

---

## Identifying Bottlenecks Under Load

During load tests, monitor each component's resource utilization to find the binding constraint:

```python
# Bottleneck identification checklist during load test

COMPONENT_INDICATORS = {
    "API pods": {
        "metrics": ["CPU %", "Memory %", "Active connections"],
        "bottleneck_signals": ["CPU > 80%", "Memory > 85%", "Response queue growing"],
        "fix": "Scale out pods (HPA threshold tuning)"
    },
    "Embedding server": {
        "metrics": ["GPU utilization %", "Queue depth", "Batch processing time"],
        "bottleneck_signals": ["GPU util < 60% but latency high (batching not working)",
                               "Queue depth growing"],
        "fix": "Tune max_wait_ms and batch_size, or add GPU node"
    },
    "Qdrant": {
        "metrics": ["Search latency P95", "CPU %", "RAM used/available"],
        "bottleneck_signals": ["Search latency growing", "CPU > 70%", "RAM > 85%"],
        "fix": "Add read replica, tune ef parameter, add RAM"
    },
    "Redis": {
        "metrics": ["Memory used %", "Commands/sec", "Connections"],
        "bottleneck_signals": ["Memory > 80%", "Eviction rate > 0"],
        "fix": "Increase maxmemory, tune eviction policy, add Redis node"
    },
    "LLM API": {
        "metrics": ["Rate limit errors (429)", "Queue depth", "Latency"],
        "bottleneck_signals": ["429 errors appearing", "Latency growing without CPU issue"],
        "fix": "Request rate limit increase, implement request queuing with backoff"
    }
}
```

---

## Capacity Planning

Before you hit a bottleneck in production, calculate headroom:

```python
def compute_capacity_headroom(
    current_qps: float,
    max_qps_per_component: dict,
    target_headroom_pct: float = 0.40  # Want 40% headroom above current peak
) -> dict:
    """
    Compute whether each component has sufficient headroom.
    """
    results = {}
    
    for component, max_qps in max_qps_per_component.items():
        utilization = current_qps / max_qps
        headroom = 1 - utilization
        
        results[component] = {
            "current_utilization": utilization,
            "headroom": headroom,
            "headroom_adequate": headroom >= target_headroom_pct,
            "capacity_for_current_load_at_qps": max_qps * (1 - target_headroom_pct),
            "action_needed": headroom < target_headroom_pct
        }
    
    return results

# Example at 200 QPS current load
capacity = compute_capacity_headroom(
    current_qps=200,
    max_qps_per_component={
        "qdrant_single_node": 800,
        "embedding_server_1gpu": 200,  # Fully saturated!
        "api_pods_3": 600,
        "redis": 50000,
        "openai_api": 500  # Rate limit
    }
)
# Result: embedding_server and openai_api have insufficient headroom at 200 QPS
```

---

## Summary

- Four scale tiers with distinct bottlenecks: < 1K (single server), 1K-100K (separate embedding service), 100K-1M (replicas, rate limits, caching), > 1M (sharding, multi-region, distributed architecture).
- LLM API rate limits are the first bottleneck that surprises teams at scale. Address them with enterprise agreements or self-hosted models before hitting the wall.
- Multi-region deployment requires an index sync strategy. Independent regional indexing from a common S3 source is the pragmatic starting point.
- Load test all four scenarios: baseline (50% peak), peak (100% peak), burst (200% spike), sustained peak (1 hour at 100%). Each probes a different failure mode.
- Monitor per-component utilization during load tests to identify the binding constraint. Fix the constraint, not the symptoms.
- Maintain 40% headroom above expected peak for all components. Plan capacity 3 months ahead.

---

## What's Next

Lesson 9.2 covers rate limiting, backpressure, and graceful degradation — what to do when the system is overwhelmed.