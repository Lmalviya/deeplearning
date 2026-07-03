# 3.1 Submission Contract: Synchronous vs. Asynchronous

## Problem

The API has to serve two traffic lanes with very different needs (Ch 1.1): real-time
submissions where a caller is waiting within the same interaction, and batch submissions where
nobody is waiting at all. A single, naive contract — "submit a document, get the label back in
the same HTTP response" — works fine for small documents at low volume, but breaks in two
specific ways as the system grows: it ties up a connection and a worker for the full duration of
processing (bad for throughput under load), and it has no sane behavior for a batch caller
submitting thousands of documents at once (nobody wants a client holding open thousands of
blocking connections).

## Solution / Concept: A Hybrid Contract, Different Per Lane

**Batch lane — always asynchronous, never blocks:**

```
POST /v1/batches
  body: { documents: [...] }  or a reference to a bulk source (e.g., object storage prefix)
  → 202 Accepted
  → { batch_id, status: "queued", document_count }

GET /v1/batches/{batch_id}
  → { batch_id, status: "processing" | "completed" | "partial_failure",
      completed_count, total_count, results_url }
```

Results are retrieved either by **polling** `GET /v1/batches/{batch_id}` or via a
**webhook callback** (a `callback_url` provided at submission time, called once the batch
completes or reaches a defined completion threshold). Since the batch SLO is a completion
window, not a per-document latency target (Ch 1.1), there is no reason to keep a connection
open — the client submits and comes back later.

**Real-time lane — synchronous when fast, with an explicit async fallback:**

```
POST /v1/documents
  body: { document: <file or reference> }
  → 200 OK  { document_id, label, confidence, taxonomy_version }
     (returned within the real-time SLO, e.g. a few seconds — Ch 1.1)

  OR, if processing exceeds an internal timeout budget before completing:
  → 202 Accepted  { document_id, status: "processing" }
     (caller falls back to polling GET /v1/documents/{document_id}, same as batch)
```

**Why hybrid, not pure-sync or pure-async, for the real-time lane:** pure synchronous-only
would mean a slow document (e.g., a 5-page contract that doesn't early-exit quickly) either
violates the latency SLO or times out with no result at all. Pure asynchronous-only would force
every real-time caller — including the common case of a fast, confidently-classified single-page
document — to make a second polling round-trip for no reason, adding latency to the majority
case to protect against the minority case. The hybrid contract optimizes for the common case
(fast, single early-exit page) while gracefully degrading the uncommon case (slow document) to
the same async pattern batch already uses — one underlying mechanism, two entry behaviors.

## Trade-offs

| Approach | Gain | Cost |
|---|---|---|
| Pure synchronous for real-time | Simplest possible client experience — one request, one response | Ties up a connection/worker for the full processing duration; no graceful behavior for slow documents; doesn't scale past the point where concurrent open connections become the bottleneck |
| Pure asynchronous for everything (batch and real-time alike) | One uniform code path, simplest to implement on the backend | Forces every real-time caller to poll even for the common fast case, adding latency where none is needed — actively works against the stated real-time SLO (Ch 1.1) |
| Hybrid: sync-if-fast with async fallback for real-time, always-async for batch | Fast path stays fast for the common case; slow-path and batch both degrade to the same well-tested async mechanism, rather than needing separate handling | More implementation complexity than either pure approach — the server needs an internal timeout budget and a clean handoff from "still trying synchronously" to "converted to an async job with the same ID" |

## When to Use Which

- **Pure synchronous** is acceptable only very early (Ch 2's MVP, low volume, no meaningful
  spike risk yet) — it's what the MVP does implicitly, and this lesson's hybrid contract is the
  natural evolution once real load and the breakpoints from Ch 2.5 are observed.
- **Always-async batch** should be the contract from the start, even at MVP scale — there's no
  version of "batch" where blocking on a full batch's completion makes sense.
- **Hybrid real-time** should be introduced as soon as the real-time lane's p95/p99 latency
  starts showing meaningful variance (some documents resolve in one early-exit page, others need
  the full page budget) — exactly the signal flagged in Ch 2.5's breakpoint table.

## Summary

Batch submissions are always asynchronous, since nothing is waiting on them and forcing a
blocking contract on bulk traffic has no upside. Real-time submissions use a hybrid contract —
synchronous when the document resolves quickly (the common case, optimized for the stated
latency SLO), with a graceful fallback to the same asynchronous job/polling mechanism batch
already uses when a document takes longer than the internal timeout budget. This keeps the
system's async machinery singular (one mechanism, used by both lanes when needed) while still
giving real-time callers the fast, simple response they expect in the common case.