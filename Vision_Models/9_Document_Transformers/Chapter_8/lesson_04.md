# 8.4 Full Microservice Architecture Diagram at 100M/50-Class Target Scale

## Problem

Chapters 1–8.3 have each added one piece — capacity planning, the MVP pipeline, the API
contract, the data schema, the queue/worker split, caching, scaling policy, and finally service
decomposition. This lesson assembles all of it into one coherent picture of the system at its
stated target scale (Chapter 1.1: ~100M documents/month, ~50 classes, 80/20 batch/real-time
split), so the full shape — and how each earlier chapter's decision shows up in it — is visible
in one place.

## The Full Architecture

```mermaid
flowchart TD
    Client[Clients] --> LB[Load Balancer]
    LB --> GW[API Gateway<br/>auth, rate limit, routing — Ch 3.3]

    GW -->|"/v1/documents"| Ing[Ingestion Service<br/>Ch 3, Ch 5.1, Ch 8.2]
    GW -->|"/v1/batches"| Ing

    Ing -->|dedup check| Cache[(Cache — Ch 6)]
    Ing --> PgIng[(Postgres: documents, batches<br/>Ch 4.2, owned by Ingestion Svc)]
    Ing --> Storage[(Object Storage — raw files, Ch 4.1)]

    Ing -->|enqueue| RQ[Real-time Queue — Ch 5.2]
    Ing -->|enqueue| BQ[Batch Queue — Ch 5.2]

    RQ --> RWorkers["Real-time Worker Pool<br/>(standing floor + predictive + reactive scaling, Ch 7.2)"]
    BQ --> BWorkers["Batch Worker Pool<br/>(queue-depth autoscaling, Ch 7.2)"]

    RWorkers --> Orch[Orchestration Service<br/>hierarchical early-exit, Ch 2.4, Ch 8.2]
    BWorkers --> Orch

    Orch -->|gRPC, per page| Extr[Extraction Service<br/>Ch 2.2, Ch 8.2]
    Orch -->|gRPC, per page| Clsf[Classification Service<br/>Ch 2.3, Ch 8.2<br/>~80+ GPUs at target scale, Ch 1.2]

    Clsf -->|reference embeddings, cached| Cache
    Clsf -.->|reads reference set| Tax[Taxonomy Service<br/>Ch 4.3, Ch 9, Ch 8.2]
    Cache -.->|invalidate on class change| Tax

    Extr --> PgExtr[(Postgres: pages<br/>owned by Extraction Svc)]
    Clsf --> PgClsf[(Postgres: predictions, embeddings<br/>owned by Classification Svc)]
    Tax --> PgTax[(Postgres: classes<br/>owned by Taxonomy Svc)]

    Orch -->|"document completed" event| Events[Async event bus<br/>Ch 5, Ch 8.3]
    Events --> Notif[Notification Service<br/>webhook dispatch, Ch 3.1]
    Events --> Review[Review Service<br/>human review queue, Ch 10.3]

    Notif -->|webhook| Client
    Ing -->|"polling / sync response, Ch 3.1"| Client
```

## Reading the Diagram Against Earlier Chapters

- **Client → LB → Gateway → Ingestion**: the API contract and infrastructure roles from
  Chapter 3, unchanged in shape, now routing into a real service rather than the MVP monolith.
- **Two queues, two worker pools**: the lane separation decided in Chapter 5.2, still fully
  intact — decomposition into services (Chapter 8) happened *within* each lane's processing,
  not instead of the lane split.
- **Orchestration → Extraction/Classification via synchronous gRPC**: the tight early-exit loop
  from Chapter 2.4, now crossing real service boundaries but still communicating synchronously
  within one document's processing, per the reasoning in Lesson 8.3.
- **Classification Service's GPU footprint**: this is where the ~80+ GPU estimate from Chapter
  1.2 actually lives — isolated as its own service specifically so its scaling can be managed
  independently of every other component's much lighter resource profile.
- **Taxonomy Service, read via cache, written rarely**: directly reflects Chapter 4.3's
  data-as-rows design and Chapter 6.1's caching rationale — the class list and reference
  embeddings are hot-read, cold-write data, served from cache in the common case, invalidated
  on the rare class-change event.
- **Async event bus for completion and review events**: the decoupled-reaction pattern from
  Lesson 8.3, feeding Notification Service (webhook dispatch, Chapter 3.1) and Review Service
  (Chapter 10.3, expanded later) without those services blocking the main processing path.
- **Per-service Postgres ownership**: each service's data lives behind its own API, per Lesson
  8.2/8.3's data-ownership discipline — drawn here as logically separate stores, though they
  may or may not be physically separate database instances depending on operational scale
  (Chapter 4.4's partitioning/read-replica guidance still applies per-service).

## What's Deliberately Not Shown Here

Multi-region topology (Chapter 7.4) is omitted from this diagram for clarity — it would overlay
on top of this architecture (regional Ingestion Service instances, centralized processing) only
if and when the triggers discussed in Chapter 7.4 are actually observed. Security/PII handling
(Chapter 12) and detailed observability/monitoring wiring (Chapter 10) are also not depicted
here, as they're cross-cutting concerns layered across every service shown, not a distinct
architectural component of their own.

## Summary

The target-scale architecture is not a different system from the MVP built in Chapter 2 — it's
the same pipeline (route, extract, classify, aggregate) with every subsequent chapter's decision
applied: an API layer with a hybrid sync/async contract (Ch 3), a schema built for zero-downtime
taxonomy growth (Ch 4), lane-separated asynchronous processing (Ch 5), targeted caching (Ch 6),
lane-specific autoscaling (Ch 7), and finally, service decomposition along the boundaries that
scaling profile, deploy cadence, and data ownership actually justified (Ch 8) — with
Classification Service, carrying the ~80+ GPU footprint from Chapter 1.2, as the component the
entire decomposition effort was most motivated by isolating correctly.