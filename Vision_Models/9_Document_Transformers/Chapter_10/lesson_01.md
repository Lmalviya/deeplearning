# 10.1 Retries, Dead-Letter Queues, and Idempotent Processing in the Async Pipeline

## Problem

The asynchronous pipeline built in Chapter 5 means processing can fail partway through — a
worker crashes mid-extraction, a downstream service (Classification, per Chapter 8.2) times
out, a transient network blip drops a call. Most message queues guarantee **at-least-once**
delivery, meaning a message can be redelivered after a failure — which raises two distinct
risks if not handled deliberately: a document could be **lost** (violating the durability
requirement from Chapter 1.1) if failures aren't retried at all, or a document could be
**double-processed** (corrupting state, producing duplicate predictions) if retries aren't
handled safely.

## Solution / Concept: Three Mechanisms Working Together

### Retry with backoff

Transient failures (a momentary service unavailability, a network blip) should be retried, but
not instantly and not unboundedly — instant retries can hammer an already-struggling downstream
service (making the underlying problem worse), and unbounded retries can leave a genuinely
broken message cycling forever, consuming worker capacity without ever succeeding. **Exponential
backoff with a bounded retry count** (e.g., retry with increasing delay, up to a defined
maximum number of attempts) is the standard mechanism: it gives transient issues time to
resolve while guaranteeing a message eventually stops being retried and moves to explicit
failure handling.

### Dead-letter queue (DLQ)

Once a message exceeds its retry limit, it moves to a **dead-letter queue** rather than being
silently dropped or retried indefinitely. This directly satisfies the durability requirement
from Chapter 1.1 even in the failure case — the document isn't lost, it's preserved for
investigation, alerting (Chapter 10.2), and manual or automated reprocessing once the
underlying issue (a corrupted file, a downstream bug) is understood and fixed.

### Idempotent processing

Because at-least-once delivery means a message *can* be redelivered even after it was actually
processed successfully (e.g., the worker crashed after completing work but before
acknowledging the message), processing logic must be safe to run twice on the same message
without producing duplicate or corrupted results. This is not a queue-level concern alone — it
requires the **application logic and schema** (Chapter 4.2) to cooperate:

- Writes to `pages` and `predictions` should use **upsert semantics** keyed on natural
  uniqueness (e.g., `(document_id, page_number)` for pages, per the unique index already defined
  in Chapter 4.2) rather than blind inserts that would fail or duplicate on redelivery.
- Before doing real work, a worker should **check the document's current status** — if a
  document is already marked `completed`, a redelivered message for it can be safely
  acknowledged and skipped rather than reprocessed, avoiding wasted compute and any risk of
  producing a second, possibly-different prediction for the same document.

This is the same idempotency discipline established at the API layer in Chapter 3.2
(content-hash deduplication), applied one layer deeper — at the level of individual queue
messages within the processing pipeline, not just at the initial submission.

### Poison message handling

A message that **crashes the worker process itself** (rather than failing gracefully with a
catchable error) is a special case — if left to the standard retry mechanism, it could crash
every worker that picks it up, cycling indefinitely and consuming capacity that should go to
other messages. Such a message should be detected (e.g., via a low retry-count threshold
specifically for hard crashes, distinct from the graceful-failure retry count) and routed to
the DLQ quickly, rather than allowed to repeatedly take down healthy workers.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Bounded retries with exponential backoff (vs. no retry, or unbounded retry) | Recovers automatically from transient failures without hammering downstream services or letting broken messages cycle forever | Requires tuning the retry count and backoff schedule — too many retries delays DLQ visibility (and therefore human awareness) of a genuinely broken document; too few sends transient blips to the DLQ needlessly, creating noisy manual work |
| DLQ for exhausted-retry messages | Guarantees durability even in the failure case; preserves context for investigation | Requires active DLQ monitoring (Chapter 10.2) — an unwatched DLQ silently accumulating failures is just as bad as no DLQ at all, since nothing gets fixed |
| Upsert-based, status-checked idempotent processing | Safe under at-least-once delivery semantics — the actual guarantee real queue infrastructure provides | Requires deliberate schema and application-logic design (Chapter 4.2's unique constraints doing real work here, not just theoretical) — retrofitting idempotency after non-idempotent logic is already live is a much harder fix |

## When to Use / When Not To

- **All three mechanisms should be in place from the point the async pipeline (Chapter 5) goes
  live** — this is not an optional hardening step to add later; at-least-once delivery is the
  default behavior of essentially all real queue systems, so redelivery-safety is a day-one
  requirement, not a scale-triggered one.
- **DLQ alerting thresholds and retry tuning** should be revisited as real failure patterns are
  observed (Chapter 10.2) — the right retry count and backoff schedule are empirical questions,
  not values to set once and never revisit.

## Summary

At-least-once delivery, the standard guarantee of real queue infrastructure, means failures and
redeliveries are a normal, expected part of the async pipeline's operation — not an edge case.
Bounded retries with exponential backoff handle transient failures gracefully, a dead-letter
queue preserves durability and visibility for failures that exceed the retry budget, and
idempotent processing logic — built on the upsert semantics and status checks the schema
already supports (Chapter 4.2) — ensures that redelivery, whether from a retry or a
post-completion crash, never produces duplicate or corrupted state.