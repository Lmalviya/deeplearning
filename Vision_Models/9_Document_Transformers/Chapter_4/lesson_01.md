# 4.1 What Goes Where: Object Storage, Relational DB, and Cache

## Problem

The pipeline produces several genuinely different shapes of data — large raw files, small
structured metadata, numeric embedding vectors, frequently-re-read hot values — and putting
all of it in one place is a mistake in both directions: storing large binary files in Postgres
bloats the database and slows every query touching that table; storing structured, relational,
frequently-queried metadata in object storage makes basic operations (find all pending
documents for tenant X, join predictions to their reviewer corrections) painfully slow or
impossible without building a second index anyway. The data placement decision needs to be made
deliberately, per data shape, not defaulted to "put everything in the database" or "put
everything in a bucket."

## Solution / Concept: Decision Criteria by Data Shape

| Data | Where it lives | Why |
|---|---|---|
| Raw uploaded document (original PDF/image, ~1MB average per Ch 1.2) | **Object storage** | Large binary blob, accessed rarely after initial processing (mainly for audit/reprocessing/human review), no need for relational queries against its bytes |
| Rendered page images (for pages that needed OCR/HTR) | **Object storage** | Same reasoning — large binary, infrequently re-accessed once extraction has run |
| Extracted text per page | **Relational DB (Postgres)**, as a text column, not object storage | Small (KB-scale), frequently needed alongside other structured metadata (which document, which page, which extraction method), and benefits from being queryable (e.g., full-text search for audit/debugging) |
| Page-level and document-level predictions (label, confidence, model version, taxonomy version) | **Relational DB** | Inherently structured and relational — needs joins (document → pages → predictions → class taxonomy), needs to support queries like "all low-confidence predictions this week," needs transactional guarantees when a human review corrects a record |
| Class taxonomy (class definitions, versions, deprecation status) | **Relational DB** | Small, structured, relationally referenced by every prediction — this is also the mechanism that makes adding class #51 a data operation, not a schema migration (Ch 4.3) |
| Embeddings (page/document-level vectors from the classification pipeline, Ch 2.3) | **Relational DB for the MVP scale (as a column, e.g., Postgres's `pgvector` extension); a dedicated vector index once class/reference-set count grows (Ch 9.3)** | At low class/reference-set counts, brute-force similarity search inside Postgres is fine and avoids operating a separate system; this changes as the reference set grows (flagged explicitly in Ch 9.3, not decided here) |
| Frequently-re-read, short-lived hot data (recent duplicate-submission hash lookups, active reference embeddings for hot classes, batch job status pings) | **Cache** | Read-heavy, latency-sensitive, doesn't need durability guarantees as strong as the primary database — role only, see Ch 6 |

## The Underlying Principle

The split isn't arbitrary — it follows from two questions asked per piece of data: **(1) is it
large, binary, and infrequently accessed after initial write** (→ object storage), and **(2) is
it small, structured, relationally connected to other data, and needs to support queries or
transactional updates** (→ relational DB)? Data that's small but extremely frequently re-read
with tight latency needs (→ cache) is a third, orthogonal axis layered on top of whichever
system is the source of truth — the cache never becomes the only place data lives.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Raw files in object storage, not in Postgres (e.g., as a `bytea` blob) | Keeps the database small, fast, and focused on structured/relational queries; object storage is built for this access pattern and is cheaper per GB | Requires an extra network hop (DB row → object storage URI → fetch) whenever raw bytes are genuinely needed (e.g., human review UI displaying the original document) |
| Extracted text and predictions in the relational DB, not object storage | Enables the queries the system actually needs — joins, filters, audit trails, transactional review corrections | Requires the schema (Ch 4.2) to be designed well; a poorly normalized schema here causes real pain later |
| Embeddings inside Postgres (pgvector) at MVP scale, rather than a dedicated vector DB from day one | One fewer system to operate while class/reference-set count is still small | Genuinely stops scaling cleanly once reference-set size or query volume grows — explicitly flagged as a Ch 9.3 concern, not deferred silently |

## When to Use / When Not To

- **This placement scheme is appropriate from the MVP (Ch 2) onward** — it's not a
  large-scale-only decision; getting it right early avoids a painful data migration later.
- **Revisit the embeddings placement specifically** once the breakpoint from Ch 2.5 (class count
  approaching double digits, reference-set lookups slowing down) is observed — this is the one
  placement decision in this table explicitly deferred rather than fully settled here.

## Summary

Data placement follows from the shape of the data, not convenience: large, infrequently-accessed
binary content goes to object storage; small, structured, relationally-connected, and
transactionally-updated data goes to the relational database; frequently-re-read hot data sits
in a cache layered on top of whichever system is authoritative. Embeddings are the one
deliberately provisional decision in this scheme — fine inside Postgres at MVP scale, explicitly
flagged for reconsideration as the class taxonomy and reference sets grow (Ch 9.3).