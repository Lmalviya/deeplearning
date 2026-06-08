# Lesson 8.2 — Self-Hosted vs. Managed: Trade-offs, Cost Model, and Ops Burden

---

## The Core Trade-off

Managed services trade money for time. Self-hosted trades time for money. Neither is universally better — the right answer depends on your team size, regulatory requirements, traffic patterns, and engineering capacity.

This lesson makes the trade-offs precise so you can make an informed decision for each component of your RAG infrastructure.

---

## Components That Require This Decision

A RAG system has several independently managed components, and the self-hosted vs. managed decision applies to each:

1. **Vector database** (Qdrant Cloud vs. self-hosted Qdrant / Pinecone vs. self-hosted Milvus)
2. **LLM API** (OpenAI/Anthropic API vs. self-hosted vLLM/Ollama)
3. **Embedding model** (OpenAI Embeddings API vs. self-hosted sentence-transformers)
4. **Document storage** (S3/GCS vs. self-hosted MinIO)
5. **Queue/messaging** (SQS/Pub-Sub vs. self-hosted Kafka/RabbitMQ)
6. **Monitoring** (Datadog/Grafana Cloud vs. self-hosted Prometheus/Grafana)

Each makes the same trade-off with different magnitudes.

---

## The Cost Model: When Self-Hosting Wins

Managed services have a predictable per-unit cost: per token, per vector stored, per query. Self-hosted has a high fixed cost (infrastructure, engineering time) and low marginal cost.

The crossover point — where self-hosting becomes cheaper — depends on volume.

### LLM Cost Crossover Example

```
OpenAI GPT-4o pricing (approximate, 2024):
  Input: $5 / 1M tokens
  Output: $15 / 1M tokens
  
Typical RAG query:
  Context: 2,000 tokens input
  Answer: 300 tokens output
  Cost per query: (2,000 × $5 + 300 × $15) / 1,000,000 = $0.0145 per query

At 100,000 queries/day:
  Daily API cost: $1,450
  Monthly: $43,500

Self-hosted alternative (e.g., Llama 3 70B on 2× H100 GPU):
  H100 instance (AWS p4de.24xlarge): ~$32/hour
  Monthly: $32 × 730 = $23,360 (+ engineering overhead)
  
Crossover: ~50,000 queries/day
```

This crossover calculation is specific to each model tier and use case. The key insight: at low volume, managed is always cheaper (no fixed cost). At high volume, self-hosted wins — but only if you have the engineering capacity to run and maintain it.

### Embedding Cost Crossover

```
OpenAI text-embedding-3-large:
  $0.13 / 1M tokens
  Average chunk: 200 tokens → $0.000026 per embedding
  
At 1M embeddings/day (large corpus updates + queries):
  Daily cost: $26
  Monthly: $780
  
Self-hosted bge-large-en-v1.5 on A10G GPU:
  A10G instance: ~$2/hour
  Monthly: $1,460
  
For embeddings alone, OpenAI is cheaper unless you exceed ~3M embeddings/day.
Self-hosted makes sense when combined with other GPU workloads (re-ranking, generation).
```

### Vector Database Cost Crossover

```
Pinecone pricing (serverless, approximate):
  Storage: $0.096/GB/month
  Queries: $4/million queries
  
At 1M vectors (1024 dims, float32 = ~4GB):
  Storage: $0.38/month
  Queries (100K/day): $12/month
  Total: ~$12.38/month
  
Self-hosted Qdrant (2-node cluster on c5.2xlarge):
  EC2: $0.34/hour × 2 × 730 = $496/month
  
Crossover: Pinecone is cheaper for almost any RAG use case below 50M vectors at moderate QPS.
Self-hosted wins at very high QPS or with strict data residency requirements.
```

---

## The Ops Burden: What You Are Actually Signing Up For

The cost calculation above ignores the largest hidden cost of self-hosting: engineering time.

### What Self-Hosting Actually Requires

**Vector database (Qdrant self-hosted):**
- Initial setup: 2-4 days (Docker, networking, storage, monitoring)
- Ongoing maintenance: 4-8 hours/month (updates, capacity planning, backup verification)
- Incident response: 2-8 hours per incident (when and if it occurs)
- Scaling operations: 1-2 days for major capacity expansions

**Self-hosted LLM (vLLM on GPU instances):**
- Initial setup: 3-7 days (model download, serving configuration, load testing)
- Ongoing maintenance: 8-16 hours/month (GPU driver updates, model updates, performance tuning)
- Incident response: GPU failures, OOM crashes, inference degradation — more frequent than for managed services
- GPU procurement and capacity planning: ongoing operational concern

**Total engineering cost:**
At $150/hour blended engineering cost and 20 hours/month of operations work: $3,000/month in engineering time. This is the invisible cost most cost analyses omit. It makes managed services appear more expensive than they are, and self-hosted appear cheaper than it is.

---

## Regulatory Requirements That Override Cost

Some decisions are not about cost — they are about compliance.

**Data residency:** If your data must remain in a specific geography (EU for GDPR, specific countries for local laws), your options narrow immediately. Managed services with multi-region options (OpenAI in EU, Pinecone with EU deployment) can satisfy some requirements. For stricter requirements, self-hosted is mandatory.

**Data processing agreements:** Using any third-party API for processing PII requires a Data Processing Agreement (DPA). Most major vendors offer DPAs. But some regulated industries prohibit sending data to third parties regardless of DPAs.

**Air-gap requirements:** Defense, intelligence, and some financial institutions require systems to operate with no external network access. Self-hosted on-premises is the only option.

**Audit requirements:** Some regulated industries require the ability to audit every API call, including the request and response content. Third-party APIs may not provide this level of logging.

---

## The Hybrid Architecture

Most production RAG systems end up with a hybrid: managed services for low-volume or low-risk components, self-hosted for high-volume or regulated components.

**Common hybrid pattern:**

```
Component            → Decision
─────────────────────────────────────────────
LLM generation       → OpenAI API (managed)
  Until 50K q/day    → then evaluate self-hosted

Embedding model      → Self-hosted (high volume, GPU shared with re-ranker)

Re-ranking model     → Self-hosted (MiniLM on same GPU as embedder)

Vector database      → Qdrant Cloud (managed, EU region for data residency)
  Until 50M vectors  → then evaluate self-hosted cluster

Document storage     → S3 (managed, cheap, reliable)

Queue (indexing)     → SQS (managed, no ops overhead)

Monitoring           → Grafana Cloud (managed, free tier sufficient for most)
```

The general principle: use managed for components where the marginal cost is low, the operational risk is low, or the engineering time saved is high. Self-host where volume makes managed cost prohibitive, where you need data sovereignty, or where the component is a core competency worth owning.

---

## LLM Serving: The Special Case

The decision between managed LLM APIs and self-hosted LLM serving deserves special treatment because the quality/cost/latency trade-offs are stark.

### Managed LLM APIs (OpenAI, Anthropic, Cohere)

**Advantages:**
- No GPU infrastructure to manage.
- Access to the best models (GPT-4o, Claude 3.5 Sonnet) that cannot be self-hosted.
- Automatic scaling — no capacity planning.
- No model download, serving configuration, or inference optimization.

**Disadvantages:**
- Data leaves your infrastructure.
- Cost scales linearly with usage.
- Rate limits can constrain burst capacity.
- Network latency (30-100ms round trip) is non-removable.
- Model changes (provider updates the model) can break behavior.

### Self-Hosted LLM Serving (vLLM, Ollama, TGI)

**vLLM** is the production standard for self-hosted LLM serving. It implements PagedAttention for efficient KV cache management and continuous batching for high throughput.

```python
# Deploy vLLM for production serving
# Dockerfile excerpt

FROM vllm/vllm-openai:latest

# Serve Llama 3 70B with tensor parallelism across 4 GPUs
CMD ["--model", "meta-llama/Meta-Llama-3-70B-Instruct",
     "--tensor-parallel-size", "4",
     "--max-model-len", "32768",
     "--gpu-memory-utilization", "0.90"]
```

**vLLM key features:**
- OpenAI-compatible API — your existing code works with minimal changes.
- Continuous batching — doesn't wait for a full batch before starting generation.
- PagedAttention — 2-4× higher throughput than naive serving.
- Supports Llama, Mistral, Falcon, Phi, and most open-source models.

**Latency comparison:**
```
OpenAI GPT-4o:
  Time to first token: 300-800ms (network + model)
  Throughput: rate-limited

Self-hosted Llama 3 70B on 4× A100:
  Time to first token: 100-200ms (no network hop)
  Throughput: ~1,500 tokens/second per instance

Self-hosted Llama 3 8B on 1× A10G:
  Time to first token: 50-100ms
  Throughput: ~3,000 tokens/second
```

**Quality trade-off:** Open-source models (Llama 3 70B, Mistral Large) are competitive with GPT-4o on many RAG tasks but still below GPT-4o on complex reasoning and instruction following. For most support and knowledge base Q&A tasks, the gap is small. For complex multi-hop reasoning or nuanced generation, GPT-4o still leads.

### Decision: Managed LLM vs. Self-Hosted

| Use case | Recommendation |
|---|---|
| < 50K queries/day | OpenAI/Anthropic API |
| > 100K queries/day | Evaluate self-hosted (cost) |
| Data residency required | Self-hosted mandatory |
| Need GPT-4 class quality | OpenAI/Anthropic API |
| RAG Q&A on domain content | Llama 3 70B competitive |
| Latency critical (< 500ms) | Self-hosted (no network hop) |
| Air-gap deployment | Self-hosted mandatory |

---

## Making the Decision: A Practical Checklist

For each component, ask these questions in order:

**1. Do regulations require self-hosting?**
If yes → self-host, stop here.

**2. What is our current volume?**
If low → start managed, revisit in 6 months.

**3. What is our projected volume in 12 months?**
Run the crossover calculation above. If you will cross the cost threshold within 12 months, plan the migration now.

**4. What is our engineering team size and ops capacity?**
If < 5 engineers total → strongly prefer managed for all components.
If dedicated infrastructure team exists → self-hosting is viable.

**5. Is this a core competency?**
LLM serving is not a core competency for a legal tech company. It may be for an AI company.

---

## Summary

- Managed services trade money for time. Self-hosted trades time for money. Neither is universally correct.
- The cost crossover depends on volume: LLM API at ~50K queries/day, embeddings at ~3M/day. Below these thresholds, managed is almost always cheaper when engineering time is included.
- Hidden ops cost (engineering time) is real and often omitted from cost comparisons. Add it explicitly.
- Regulatory requirements (data residency, air-gap, audit) override cost considerations.
- Hybrid architectures are common and practical: managed for low-volume or low-risk components, self-hosted for high-volume or regulated components.
- For LLM serving: OpenAI/Anthropic for < 50K queries/day or when GPT-4 quality is required. vLLM + open-source for > 100K queries/day or data residency constraints.

---

## What's Next

Lesson 8.3 covers containerizing a RAG system with Docker — service decomposition, Docker Compose setup, and building container images for each pipeline component.