# Document Classification System — Design Notes (Scale Target: 100M docs, 50 classes)

A system-design reference for a document classification platform that starts small (5
classes, modest traffic) and is explicitly designed to grow toward **~100M documents** and
**~50 classes**, with a stated traffic mix of **80% batch / 20% real-time**, including sudden
traffic spikes.

## How these notes work

- **ML/pipeline decisions** (extraction method, model paradigm, aggregation, taxonomy design,
  etc.) are covered **in depth**: problem → options → trade-offs → the option chosen for this
  system → why → what would have to change to switch later.
- **Standard infra building blocks** (API gateway, load balancer, queue, cache, database) are
  **not trade-off-debated** — they're used directly, with their role in this system stated
  plainly. The interesting design work is in *how* they're wired together, not in re-litigating
  Kafka vs. RabbitMQ from first principles every time.
- **Architecture diagrams (Mermaid)** are included wherever a chapter changes the shape of the
  system — small-scale MVP, post-queue, post-microservices, post-100M-scale, etc. — so the
  system's evolution is visible, not just described in prose.
- Chapters build in order: small system first, then explicit chapters on *what breaks* and
  *what changes* as traffic and class count grow. Nothing is designed for 100M/50-class scale
  on day one — the notes show the path there and the concrete triggers for each step.

## Chapter Index

### Chapter 1 — Requirements and Capacity Planning
- 1.1 Functional and Non-Functional Requirements (states the 100M-doc/50-class target, the
  80% batch / 20% real-time traffic split, and latency/accuracy SLOs up front — without
  designing for end-state yet)
- 1.2 Back-of-Envelope Capacity Estimation (docs/day, pages/doc, storage/day, GPU-seconds/doc
  at steady state)
- 1.3 Spike Traffic and Burst Capacity Planning (how to size for bursts on top of steady-state
  load — spike multipliers, buffering vs. autoscaling limits, what happens if you under-provision)

### Chapter 2 — The Small-Scale Architecture (MVP)
- 2.1 Single-Service Architecture — the deliberately simple starting shape, and why starting
  simple is correct even with a 100M-scale target in mind
- 2.2 Content Extraction Pipeline (PDF routing, OCR, layout detection, HTR — consolidated,
  deep trade-off discussion, one option chosen and wired into the architecture diagram)
- 2.3 Classification Pipeline (signal families, late fusion, joint fusion, domain adaptation —
  consolidated into a single deep decision chain ending in one chosen design)
- 2.4 Page-to-Document Aggregation (strategy trade-offs + cost/latency tiering, one chosen
  design, batch vs. real-time handling introduced here for the first time)
- 2.5 MVP Architecture Diagram and Breakpoints (the full small-scale system diagram, plus the
  concrete signals — traffic, latency SLOs, class count, cost — that tell you it's about to break)

### Chapter 3 — API Design
- 3.1 Submission Contract: Synchronous vs. Asynchronous (why sync breaks first; polling/webhook
  design for async results)
- 3.2 Versioning, Idempotency, and Backward Compatibility (what breaks when class 51 is added
  or the response schema changes)
- 3.3 Where the API Gateway and Load Balancer Sit (role only, no trade-off debate) — updated
  request-flow diagram

### Chapter 4 — Data and Storage Architecture
- 4.1 What Goes Where: Object Storage vs. Relational DB vs. Cache (decision criteria specific
  to this system's data shapes — raw files, extracted text, embeddings, predictions)
- 4.2 Postgres Schema Design (documents, pages, predictions, classes, audit/review tables —
  actual DDL)
- 4.3 Schema Evolution: 5 → 50+ Classes Without Downtime
- 4.4 Partitioning, Sharding, and Read Replicas (when single-Postgres stops being enough, and
  what replaces it)

### Chapter 5 — Asynchronous Processing: Queues and Producer-Consumer Design
- 5.1 Why Synchronous Request-Response Breaks First at Scale
- 5.2 Producer-Consumer Architecture for an 80% Batch / 20% Real-Time Mix (two lanes sharing
  workers vs. fully separate pools — trade-off discussion, one design chosen)
- 5.3 Worker Pool Design, Dynamic GPU Batching, and Inference Serving (throughput-oriented
  batch workers vs. latency-bounded real-time workers)
- 5.4 Where the Queue Sits (role only) — updated architecture diagram

### Chapter 6 — Caching Strategy
- 6.1 What's Actually Cacheable Here (duplicate-document detection, OCR results, embeddings,
  hot-class prototypes) — the interesting design question
- 6.2 Where the Cache Sits (role only) — updated architecture diagram

### Chapter 7 — Scaling Traffic: 1K → 100M Documents
- 7.1 Bottleneck Analysis at Each Order of Magnitude (where OCR, GPU inference, DB writes, and
  queues break first, in sequence, with numbers)
- 7.2 Horizontal Scaling and Autoscaling Policies for Steady Growth
- 7.3 Handling the Spike Case in Practice (connecting back to Ch 1.3 — queue buffering,
  autoscaling lag, shedding/backpressure strategy when a spike exceeds provisioned capacity)
- 7.4 Multi-Region, Data Locality, and Latency vs. Consistency Trade-offs

### Chapter 8 — From Monolith to Microservices
- 8.1 Why and When to Decompose (the concrete triggers — not "microservices because scale" as
  a reflex, but the specific pain points in this system that justify it)
- 8.2 Service Boundaries for This System (ingestion, extraction, classification, aggregation,
  review/feedback — what owns what, and why the boundaries are drawn there)
- 8.3 Inter-Service Communication and Data Ownership
- 8.4 Full Microservice Architecture Diagram at 100M/50-class Target Scale

### Chapter 9 — Scaling the Class Taxonomy: 5 → 50+ Classes
- 9.1 Why Fixed Softmax Heads Break First (recap, extended to 50-class reality)
- 9.2 Hierarchical Taxonomies and Coarse-to-Fine Classification (is "50 classes" really flat,
  or a tree — trade-offs of each)
- 9.3 Open-Set / Embedding Architecture at 50-Class Scale (vector DB, ANN search vs. brute-force
  KNN, when exact KNN stops being viable)
- 9.4 Human-in-the-Loop and Active Learning for Onboarding New Classes

### Chapter 10 — Reliability, Observability, and Feedback Loops
- 10.1 Retries, Dead-Letter Queues, and Idempotent Processing in the Async Pipeline
- 10.2 Monitoring, Drift Detection, and Retraining Triggers
- 10.3 Human Review Workflows and Feedback Capture (closing the loop into training data)

### Chapter 11 — Cost Optimization at Scale
- 11.1 Cost Drivers Breakdown (OCR calls, GPU inference, storage, VLM fallback calls, DB I/O)
- 11.2 Tiered/Cascaded Design for Cost Control at 100M-doc Volume (system-wide extension of the
  cheap-primary/expensive-fallback pattern)

### Chapter 12 — Security, Privacy, and Compliance *(deferred — not yet scoped in detail)*
- 12.1 PII Handling for Sensitive Document Classes (ID documents, contracts) — encryption,
  redaction, retention
- 12.2 Multi-Tenancy and Access Control

### Chapter 13 — Hands-On Toolchain Reference *(deferred — not yet scoped in detail)*
- 13.1 OCR in Practice — EasyOCR
- 13.2 Layout Detection in Practice — PaddleOCR PP-StructureV3
- 13.3 Handwriting Recognition in Practice — TrOCR
- 13.4 LayoutLMv3 as a Frozen Embedding Extractor
- 13.5 Data Sourcing Reference

---

## Running Requirements Carried Through Every Chapter

- **Target end-state:** ~100M documents processed, ~50 document classes supported.
- **Traffic mix:** 80% batch (no human waiting on the result) / 20% real-time (user waiting on
  a response) — this split is a first-class input to queue, worker, and API design, not an
  afterthought.
- **Spikes:** the system must survive sudden bursts above steady-state average, not just scale
  smoothly with gradual growth — sized explicitly in Chapter 1.3 and handled operationally in
  Chapter 7.3.
- **Growth path, not a single design:** every chapter that touches scale states the concrete
  signal that tells you the current design is breaking, and what the next design looks like —
  the notes are a map of transitions, not one fixed final architecture.

---

Tell me which chapter(s) or lesson(s) to generate next.