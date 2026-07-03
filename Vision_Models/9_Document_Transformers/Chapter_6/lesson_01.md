# 6.1 What's Actually Cacheable Here

## Problem

Caching isn't universally beneficial — it helps specifically for data that is **read
frequently relative to how often it changes**, and adds real cost (staleness risk, invalidation
complexity, one more system to keep consistent) when applied to data that doesn't fit that
pattern. Reflexively caching "everything hot-sounding" without checking read/write ratios and
size wastes engineering effort and introduces subtle correctness bugs (serving stale
predictions, stale class lists) without a corresponding performance win. This system has a
specific set of genuinely good caching candidates, and a specific set of things that look
tempting but aren't — both worth stating explicitly.

## Solution / Concept: Four Genuine Caching Candidates

### 1. Duplicate-submission hash lookups

Every submission triggers a check against `documents.content_hash` (Chapter 4.2's idempotency
mechanism, Chapter 3.2's dedup contract). At real-time volume (Chapter 1.2: ~7.7 docs/sec
average, up to ~62 docs/sec at spike peak, Chapter 1.3), this check runs on the hot path of
*every single submission*, including the overwhelming majority that are not duplicates.
Caching recently-seen hashes (a Redis `SET`/lookup, with a TTL matching the dedup window from
Chapter 3.2, e.g. 24 hours) avoids a Postgres round-trip for this extremely frequent check —
**read-to-write ratio is very high** (every submission reads; only a genuinely new document
writes), making this a strong caching candidate.

### 2. Reference embeddings / class prototypes

The classification pipeline (Chapter 2.3) compares every page's embedding against the
per-class reference set on **every single inference call** — this is the single most
frequently-read piece of data in the entire system, happening at the full page-inference rate
(~58/sec average from Chapter 1.2, higher at spike). It changes only when a class is added,
its reference set is updated, or deprecated (Chapter 4.3) — a rare, deliberate event by
comparison. This is close to the ideal caching pattern: **extremely high read frequency,
extremely low write frequency.** Keeping the current reference set in an in-memory cache
(refreshed on class-taxonomy change, not polled continuously) removes a database or vector-
index round-trip from the single hottest path in the system.

### 3. Taxonomy metadata (class list, names, taxonomy_version mapping)

Nearly every response needs to resolve a `class_id` to a name and attach the current
`taxonomy_version` (Chapter 3.2, Chapter 4.2). Like reference embeddings, this is read on
nearly every request and changes only on the rare class-addition/deprecation event (Chapter
4.3) — same high-read/low-write shape, same justification for caching.

### 4. Batch job status

Batch clients poll `GET /v1/batches/{batch_id}` (Chapter 3.1) — often repeatedly, sometimes at
a tight client-side polling interval, especially for large batches with many documents still
in flight. The underlying status (completed_count/total_count) only changes as individual
documents finish processing, which happens far less often than a busy client might poll.
Caching the computed status response for a short TTL (e.g., a few seconds) absorbs repeated
polling without adding meaningful staleness — a batch client polling every second doesn't need
a strictly real-time answer every single time.

## What's Deliberately *Not* Cached

| Data | Why it's not a good caching candidate |
|---|---|
| Raw document files | Large (≈1MB average, Chapter 1.2), read rarely after initial processing (mainly for audit/review), and already served from object storage — caching large, cold, infrequently-read blobs wastes cache memory for no real latency win |
| Individual page-level predictions, long after processing | Written once, read rarely after the document's result has been returned to the client — a write-once/read-rarely pattern is the opposite of what caching helps with |
| Extracted OCR text | Same reasoning as page predictions — read once (to produce a prediction), then mostly accessed only for audit or human review, an infrequent path already well-served directly from Postgres |

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Caching dedup hashes, reference embeddings, taxonomy metadata, and batch status | Removes database round-trips from the system's hottest, most frequently-hit paths — directly improves both latency (real-time SLO, Chapter 1.1) and database load (reduces pressure that would otherwise push toward sharding, Chapter 4.4, sooner than necessary) | Requires correct invalidation specifically on class-taxonomy changes (Chapter 4.3) — a stale cached reference set or class list served after a class addition would silently misclassify or mislabel, a real correctness risk if invalidation is handled sloppily |
| Not caching raw files, page text, and individual predictions | Avoids wasting cache capacity and avoids a staleness risk for data where it provides little to no benefit | None significant — this is a clear-cut "don't bother" case, not a real trade-off |

## When to Use / When Not To

- **Cache the four candidates above** as soon as their read volume becomes meaningful — for
  reference embeddings and taxonomy metadata, this is essentially from day one of moving past
  the single-instance MVP (Chapter 2), since every inference call touches them.
- **Do not cache raw files, extracted text, or individual page predictions** — there's no
  read pattern in this system that benefits from it, and doing so anyway is wasted engineering
  effort and unnecessary staleness surface area.

## Summary

Good caching candidates in this system share one shape: read very frequently, written rarely —
duplicate-hash lookups, class reference embeddings, taxonomy metadata, and batch status all fit
this pattern and directly relieve pressure on the database's hottest paths. Raw files,
extracted text, and individual predictions don't fit this pattern and are deliberately left
uncached — caching them would add staleness risk and engineering overhead without a
corresponding performance benefit.