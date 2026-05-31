# RAG Mastery: From Prototype to Production

## Part 1 — Foundations & Mental Models
**Lesson 1.1** — What RAG actually is and why it exists \
**Lesson 1.2** — The full RAG landscape: naive → advanced → agentic → graph \
**Lesson 1.3** — Anatomy of a retrieval pipeline (your pipeline dissected in depth) \
**Lesson 1.4** — Anatomy of an indexing pipeline \
**Lesson 1.5** — Choosing the right RAG variant for your data type (invoices, financial docs, long docs, code, structured/tabular data, multimodal)

---

## Part 2 — Indexing Deep Dive
**Lesson 2.1** — Chunking strategies: fixed, recursive, semantic, document-aware, late chunking \
**Lesson 2.2** — Embedding models: choosing, fine-tuning, matryoshka embeddings, late interaction (ColBERT) \
**Lesson 2.3** — Metadata design and filtering strategies \
**Lesson 2.4** — Document pre-processing pipelines: OCR, layout parsing, table extraction, chart/diagram handling \
**Lesson 2.5** — Handling multimodal documents (scanned PDFs, image-heavy, mixed content) \
**Lesson 2.6** — Incremental indexing and data freshness strategies \
**Lesson 2.7** — Parent-child chunking and hierarchical indexing

---

## Part 3 — Retrieval Deep Dive
**Lesson 3.1** — Dense retrieval internals: HNSW, IVF, product quantization \
**Lesson 3.2** — Sparse retrieval: BM25, SPLADE, learned sparse models \
**Lesson 3.3** — Hybrid search design and RRF vs. score-based fusion \
**Lesson 3.4** — Query understanding: rewriting, expansion, decomposition \
**Lesson 3.5** — Hypothetical Document Embeddings (HyDE) and query-side augmentation \
**Lesson 3.6** — Re-ranking: cross-encoders, LLM-as-reranker, ColBERT, Cohere Rerank \
**Lesson 3.7** — Contextual compression and context window packing \
**Lesson 3.8** — Retrieval failure modes and how to diagnose them 

---

## Part 4 — Generation & Prompting
**Lesson 4.1** — Prompt design for RAG: grounding, citation, refusal \
**Lesson 4.2** — Handling conflicting context vs. parametric knowledge \
**Lesson 4.3** — Long-context generation: stuffing vs. iterative vs. map-reduce \
**Lesson 4.4** — Structured output generation from retrieved context (tables, JSON, reports) \
**Lesson 4.5** — Hallucination: causes, detection, mitigation 

---

## Part 5 — Advanced RAG Architectures
**Lesson 5.1** — Corrective RAG (CRAG): self-assessment and fallback retrieval \
**Lesson 5.2** — Self-RAG: selective retrieval and reflection tokens \
**Lesson 5.3** — Agentic RAG: tool-calling, multi-step reasoning, ReAct loops \
**Lesson 5.4** — Graph RAG: knowledge graph construction, entity linking, community summaries \
**Lesson 5.5** — Multi-hop and multi-document reasoning \
**Lesson 5.6** — Conversational RAG: memory, history compression, session context 

---

## Part 6 — Evaluation
**Lesson 6.1** — Evaluation philosophy: offline vs. online, component-level vs. end-to-end \
**Lesson 6.2** — Retrieval metrics in depth: Precision@K, Recall@K, MRR, MAP, NDCG, Hit Rate, Coverage \
**Lesson 6.3** — Generation metrics in depth: Exact Match, F1, BLEU, ROUGE, METEOR, BERTScore, Semantic Similarity \
**Lesson 6.4** — RAG-specific metrics: faithfulness, answer relevancy, context precision/recall, answer correctness (RAGAS framework and alternatives) \
**Lesson 6.5** — Building evaluation datasets: golden sets, LLM-as-judge, human annotation \
**Lesson 6.6** — Online evaluation: A/B testing, user feedback signals, implicit signals (thumbs, dwell time, reformulations) \
**Lesson 6.7** — Data drift and distribution shift: detection and response 

---

## Part 7 — Debugging & Accuracy Problems
**Lesson 7.1** — Systematic debugging framework: isolating retrieval vs. generation failures \
**Lesson 7.2** — The "accuracy drops at 5K docs" problem — root cause analysis and fixes \
**Lesson 7.3** — Data conflicts and knowledge inconsistency resolution \
**Lesson 7.4** — Retrieval accuracy degradation at scale: index quality, embedding drift, query distribution shift \
**Lesson 7.5** — Tracing and observability: building a RAG trace (query → chunks → prompt → response) \
**Lesson 7.6** — Common failure patterns catalog and diagnostic playbook 

---

## Part 8 — Deployment & Infrastructure
**Lesson 8.1** — Vector database landscape: Qdrant, Pinecone, Weaviate, pgvector, Milvus — when to use what \
**Lesson 8.2** — Self-hosted vs. managed: trade-offs, cost model, ops burden \
**Lesson 8.3** — Containerizing a RAG system with Docker: service decomposition, compose setup \
**Lesson 8.4** — AWS deployment: EC2 vs. ECS vs. Lambda, S3 for documents, architecture patterns \
**Lesson 8.5** — Kubernetes for RAG: pods, HPA, resource limits, rolling deployments \
**Lesson 8.6** — Scaling the retrieval layer: read replicas, sharding, caching (query cache, embedding cache) \
**Lesson 8.7** — Serving LLMs: self-hosted (vLLM, Ollama) vs. API, latency vs. cost trade-offs \
**Lesson 8.8** — CI/CD for RAG systems: index versioning, model versioning, regression testing 

---

## Part 9 — Scale & Production Reliability
**Lesson 9.1** — Scaling to millions of users: architecture patterns, bottlenecks, load testing \
**Lesson 9.2** — Rate limiting, backpressure, and graceful degradation \
**Lesson 9.3** — Async indexing pipelines: queues, workers, retry logic (Celery, SQS, Kafka) \
**Lesson 9.4** — Cost management at scale: token budgets, caching, batching, tiered retrieval \
**Lesson 9.5** — Debugging in production at scale: distributed tracing, log aggregation, alerting \
**Lesson 9.6** — Security and access control: multi-tenant RAG, document-level permissions, PII handling 

---

## Part 10 — System Design Case Studies
**Case Study 1** — Enterprise document Q&A (mixed PDF types, 100K+ documents, multi-tenant) \
**Case Study 2** — Financial report analysis system (tables, charts, earnings calls, regulatory filings) \
**Case Study 3** — Customer support RAG (high QPS, freshness requirements, feedback loop) \
**Case Study 4** — Codebase assistant (code chunking, cross-file context, tool use) \
**Case Study 5** — Legal/compliance document search (precision-critical, citation required, audit trail) \
**Case Study 6** — Multimodal RAG (invoices, receipts, scanned forms — OCR + layout + retrieval) 