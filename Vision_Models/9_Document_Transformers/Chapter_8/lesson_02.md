# 8.2 Service Boundaries for This System

## Problem

Given the decision to decompose (Lesson 8.1), the boundaries have to be drawn somewhere
specific — and drawn badly, they recreate a "distributed monolith" (services that are
technically separate deployments but still tightly coupled via shared databases or chatty,
tangled calls, getting all of microservices' operational cost with none of its benefit). The
boundaries should follow the same logic that justified decomposition in the first place:
scaling profile, deploy cadence, and data ownership.

## Solution / Concept: Six Services, Each With a Clear Owning Responsibility

| Service | Owns | Maps to (from earlier chapters) | Scaling profile |
|---|---|---|---|
| **Ingestion Service** | Receiving submissions, idempotency/dedup check, persisting raw files + `documents`/`batches` records, enqueueing jobs | Ch 3 (API design), Ch 5.1 (producer role) | Lightweight, CPU-bound, scales with request rate, not GPU-bound |
| **Extraction Service** | Per-page routing (text-layer check), OCR, HTR, direct text extraction | Ch 2.2 | Mixed CPU/GPU (OCR/HTR models), scales with page-inference rate |
| **Classification Service** | Embedding backbone inference, similarity comparison against reference embeddings | Ch 2.3 | Heavily GPU-bound — this is the ~80+ GPU estimate from Ch 1.2, the single largest resource consumer |
| **Orchestration Service** | Hierarchical early-exit control flow (Ch 2.4) — decides whether to stop or pull another page, coordinates calls to Extraction and Classification per lane's orchestration policy (sequential vs. parallel, Ch 5.2/5.3) | Ch 2.4, Ch 5.2, Ch 5.3 | Lightweight control-flow logic; scales with in-flight document count, not compute-heavy itself |
| **Taxonomy Service** | The `classes` table, reference embedding sets, the class add/deprecate lifecycle | Ch 4.3, Ch 9 | Very low write volume, read-heavy (served mostly from cache, Ch 6) |
| **Notification/Review Service** | Webhook dispatch for batch completion (Ch 3.1), human review queue and correction capture (Ch 10.3, introduced fully later) | Ch 3.1, Ch 10.3 | Low volume relative to the main pipeline |

## Why These Boundaries, Specifically

- **Classification is its own service** because it is, by a wide margin, the most
  resource-intensive and differently-scaled component (Ch 1.2's ~80+ GPU estimate) — bundling
  it with anything else would force that component's sizing decisions onto unrelated code.
- **Orchestration is separated from both Extraction and Classification**, rather than folded
  into either, because it's a genuinely distinct responsibility: it's the "conductor" that
  calls Extraction and Classification repeatedly per document (per the early-exit loop from
  Chapter 2.4) and makes the stop/continue decision — this logic doesn't belong inside either
  of the services it's coordinating, or those services would need to know about aggregation
  policy, violating their own single responsibility.
- **Taxonomy is separated from Classification** even though Classification is Taxonomy's main
  consumer, because Taxonomy's lifecycle (Chapter 4.3's add/deprecate flow, Chapter 9's
  hierarchical taxonomy work) is managed on a completely different cadence and by a different
  concern (class governance, not model inference) than the inference path itself.
- **Ingestion is separated from Orchestration** because Ingestion's job ends at "durably
  recorded and enqueued" (Chapter 5.1) — it has no need to know anything about how a document
  is subsequently processed, keeping the API-facing service simple and independently scalable
  against request volume alone.

## Data Ownership Principle

Each service owns specific tables from the Chapter 4.2 schema, and **other services must go
through the owning service's API to read or write that data — never query another service's
tables directly.** This is the standard microservices data-ownership discipline, stated here
because violating it is the most common way a decomposition ends up as a distributed monolith
in practice: if Classification Service reaches directly into Ingestion Service's `documents`
table, the two are still coupled at the schema level, just with extra network hops added on
top for no benefit.

| Service | Owns these tables/data (from Ch 4.2) |
|---|---|
| Ingestion Service | `documents`, `batches` |
| Extraction Service | `pages` (extraction-related columns) |
| Classification Service | `predictions`, embeddings |
| Taxonomy Service | `classes` |
| Notification/Review Service | Review/correction fields on `predictions` (via Classification Service's API, or a dedicated review table — an implementation detail resolved when Ch 10.3 is built out in full) |

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Six services, boundaries drawn along scaling/cadence/ownership lines | Each service can be scaled, deployed, and owned independently, directly addressing the triggers from Lesson 8.1 | Six services is genuinely more operational surface area than one monolith — more deploy pipelines, more service-to-service contracts to maintain and version |
| Strict per-service data ownership (no cross-service direct DB access) | Avoids the distributed-monolith anti-pattern; each service's internal schema can evolve without breaking others, as long as its API contract is stable | Requires every cross-service data need to go through an API call rather than a direct query — adds latency and requires the owning service's API to actually expose what consumers need (a real design discipline, not automatic) |

## When to Use / When Not To

- **This exact six-service split** is a reasonable landing point specifically for a system with
  this pipeline shape (heterogeneous extraction, GPU-heavy classification, an orchestration
  loop, a growing taxonomy, and a human-review requirement) — it is not a universal
  microservices template.
- **Fewer services** would be reasonable if, on inspection, some of the triggers from Lesson
  8.1 apply to only a subset of these boundaries — e.g., if Taxonomy's write volume and
  governance needs turn out too small to justify a fully separate service, it could remain a
  well-isolated module inside Classification Service until its own trigger appears.

## Summary

Six services — Ingestion, Extraction, Classification, Orchestration, Taxonomy, and
Notification/Review — are drawn along the same lines that justified decomposition in the first
place: differing scaling profiles (especially Classification's GPU-heavy footprint),
differing deploy cadence, and clear data ownership per the Chapter 4.2 schema. The discipline
that prevents this from becoming a distributed monolith is strict per-service data ownership —
every cross-service data need goes through an API, never a direct database query into another
service's tables.