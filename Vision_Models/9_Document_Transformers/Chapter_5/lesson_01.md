# 5.1 Why Synchronous Request-Response Breaks First at Scale

## Problem

The MVP (Chapter 2) processes documents synchronously or via a lightweight in-process
mechanism — a request comes in, the service runs extraction and classification, and returns a
result on the same connection. This works while traffic is low, but breaks down for two
independent reasons as the system approaches real load, and both were already flagged as
breakpoints in Chapter 2.5.

**Reason 1 — connection/resource holding under concurrency.** Each in-flight request holds a
connection, a thread or async task, and (during classification) GPU access for the full
duration of processing — which, per Chapter 1.2's estimate, is on the order of ~1 second per
page-inference, and a document may need multiple pages before early-exit resolves. At even a
modest fraction of the target ~38.6 docs/sec average, a purely synchronous service needs enough
concurrent capacity to hold that many in-flight requests simultaneously — and every traffic
spike (Chapter 1.3) multiplies this requirement instantaneously, with no buffer between "a
request arrived" and "a request must be actively processed right now."

**Reason 2 — batch submissions never fit a synchronous model at all.** This was already
established in Chapter 3.1: a batch of thousands of documents cannot reasonably be processed
within one blocking HTTP request. The API contract already commits to asynchronous handling for
batch traffic — but Chapter 3's API design assumed *something* exists on the backend to accept
a submission and process it later. That "something" is the producer-consumer architecture this
chapter builds.

## Solution / Concept: Decouple Acceptance From Processing

The fix is to separate **"a document was received and durably recorded"** from **"a document
was processed"** into two distinct steps, connected by a queue:

1. **Producer (API service):** receives a submission, persists the raw file and a database
   record (Chapter 4.2's `documents` row, `status = 'queued'`), pushes a lightweight message
   (a reference to the document, not the file itself) onto a queue, and immediately
   acknowledges the client — this satisfies the durability requirement from Chapter 1.1
   ("submitted documents must not be lost") without requiring processing to have completed yet.
2. **Consumer (worker pool):** independently pulls messages off the queue and performs
   extraction, classification, and aggregation (Chapter 2's pipeline) at its own pace, updating
   the `documents`/`predictions` rows as it completes each one.

This decoupling is what makes the hybrid sync/async API contract from Chapter 3.1 actually
implementable: the real-time lane's "synchronous when fast" path waits on the queue+worker to
finish within an internal timeout budget, falling back to the async polling contract exactly
when that budget is exceeded — the same underlying producer-consumer mechanism serves both
lanes, just with different waiting behavior on the client-facing side.

## Trade-offs

| Aspect | Gain | Cost |
|---|---|---|
| Decoupling acceptance from processing | API throughput is no longer bound by processing time — the API can accept requests as fast as it can write a queue message and a DB row, independent of how long extraction/classification takes | Adds one queue hop of latency to every request, even fast ones — a small but real cost for the common fast-path case |
| Independent scaling of ingestion vs. processing | Worker pool capacity (GPU-bound, Chapter 1.2's ~80+ GPU estimate) can scale independently of API instance count (which is comparatively cheap, CPU-bound, and easy to scale) | Requires operating and monitoring a queue as a real piece of infrastructure — visibility into queue depth, consumer lag, and dead-letter handling becomes a new operational responsibility (Chapter 10.1) |
| Queue as a buffer against spikes | Directly enables the spike-absorption behavior designed in Chapter 1.3 — a burst of submissions queues up rather than overwhelming processing capacity instantaneously | The queue can mask a genuine under-provisioning problem if depth isn't actively monitored — a growing backlog is easy to miss without dashboards/alerting (Chapter 10.2) |

## When to Use / When Not To

- **Introduce the queue-based producer-consumer architecture as soon as the async batch
  contract (Chapter 3.1) needs a real backend implementation** — this is not an optional later
  optimization; it's the mechanism the API contract already assumed exists.
- **For very low, pre-production traffic**, the MVP's synchronous/in-process handling (Chapter
  2.1) remains acceptable — introducing a queue before there's any real concurrency to manage
  is unnecessary complexity, consistent with the "start simple" principle from Chapter 2.1.
- **The concrete trigger** is the same breakpoint table from Chapter 2.5: real-time p95 latency
  approaching SLO under normal (non-spike) load, or batch submissions being accepted at all in
  production (since batch was always async-by-contract from Chapter 3.1).

## Summary

Synchronous request-response ties a request's resource usage (connection, thread, GPU access)
to the full duration of processing, which doesn't scale past modest concurrency and was never a
viable model for batch traffic in the first place. Decoupling acceptance (durably record and
acknowledge) from processing (extract, classify, aggregate) via a queue is what makes the
hybrid API contract from Chapter 3.1 real, enables independent scaling of ingestion and
processing capacity, and is the mechanism that lets a traffic spike (Chapter 1.3) become a
growing queue depth instead of an outage.