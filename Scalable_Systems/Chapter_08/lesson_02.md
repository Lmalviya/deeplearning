# Lesson 8.2 — Latency vs Throughput

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 8.1 — Consistency vs Availability](./lesson-8.1-consistency-vs-availability.md) | Next: [Lesson 8.3 — SQL vs NoSQL](./lesson-8.3-sql-vs-nosql.md)

---

## What this lesson covers

- What latency and throughput actually measure and why they conflict
- Batching as the primary throughput technique — and its latency cost
- Streaming as the primary latency technique — and its throughput cost
- How to decide which to optimize for in your system
- The Little's Law formula — the math connecting latency, throughput, and concurrency

---

## 1. Definitions

**Latency:** Time for a single operation to complete, from start to finish. Measured in milliseconds. "How long does one request take?"

**Throughput:** How many operations complete per unit of time. Measured in requests/second, messages/second, bytes/second. "How much work does the system do per second?"

They are related but different, and optimizing for one often hurts the other.

---

## 2. Why They Conflict

The fundamental conflict: **batching improves throughput but adds latency. Processing individually improves latency but reduces throughput.**

### The coffee shop analogy

**Low latency, low throughput (one at a time):**
```
Customer arrives → barista makes coffee immediately → serves → next customer
Latency: 3 minutes per customer
Throughput: 20 customers/hour
```

**High throughput, high latency (batching):**
```
Barista waits until 5 orders accumulate → makes all 5 at once efficiently
→ serves all 5 simultaneously
Latency: first customer waits up to 10 minutes (waiting for batch to fill)
         last customer in batch waits only 3 minutes
Throughput: 40 customers/hour (efficiency of batch preparation)
```

Batching increased throughput (more customers served per hour) at the cost of latency (each customer waits longer).

---

## 3. The Core Technique: Batching for Throughput

Batching means accumulating multiple operations and processing them together.

### Database batch inserts

```python
# Low throughput — one write at a time:
for event in events:
    db.execute("INSERT INTO logs (user_id, action) VALUES (%s, %s)",
               event['user_id'], event['action'])
# 1,000 events = 1,000 round trips to DB = 1,000 × 5ms = 5 seconds

# High throughput — batch insert:
db.execute_many(
    "INSERT INTO logs (user_id, action) VALUES (%s, %s)",
    [(e['user_id'], e['action']) for e in events]
)
# 1,000 events = 1 round trip to DB = ~10ms total
# 500× throughput improvement
```

**Latency cost:** The first event in the batch waits until the batch is full before being inserted. If batch fills every 100ms, the first event in a batch may wait up to 100ms before it is persisted.

### Kafka batch publishing

```python
# Kafka producer batches messages before sending (configurable)
producer = KafkaProducer(
    linger_ms=10,         # wait up to 10ms for batch to fill
    batch_size=16384,     # or until 16KB of messages accumulate
)

# Messages are not sent immediately — they wait in the batch
producer.send("events", value=event_bytes)

# After 10ms (or 16KB), entire batch sent in one network call
# Throughput: 10× higher than sending one at a time
# Latency: every message waits up to 10ms before delivery
```

`linger_ms=0` minimizes latency (send immediately). `linger_ms=100` maximizes throughput. This is an explicit latency/throughput dial.

### HTTP/2 multiplexing

HTTP/2 sends multiple requests over one TCP connection simultaneously:
```
HTTP/1.1: 10 requests = 10 TCP connections = 10 × connection overhead
HTTP/2:   10 requests = 1 TCP connection × 10 streams = lower overhead

Throughput: higher (less connection overhead)
Latency: individual request latency similar, but more requests fit per connection
```

---

## 4. The Core Technique: Streaming for Latency

Streaming processes data as it arrives, one item at a time, instead of accumulating and processing in bulk.

```python
# Batch processing (high latency for individual items):
def process_report(data: list) -> list:
    results = []
    for item in data:
        results.append(transform(item))
    return results  # First item waits for ALL items to be processed before returning

# Streaming (low latency for individual items):
def process_report_streaming(data_stream):
    for item in data_stream:
        yield transform(item)  # Each item processed and yielded immediately
        # Consumer gets first item in milliseconds, not after all items

# Streaming API response:
async def get_large_dataset(response):
    async for row in db.stream("SELECT * FROM large_table"):
        await response.write(json.dumps(row) + "\n")
        # Client receives rows as they are fetched, not after all are fetched
```

Server-Sent Events (SSE) and WebSockets apply this principle to real-time communication — events are pushed to clients as they occur, not batched.

---

## 5. Little's Law — The Math

Little's Law connects latency, throughput, and concurrency:

```
L = λ × W

Where:
  L = average number of requests in the system (concurrency)
  λ = average throughput (requests/second)
  W = average latency (seconds/request)
```

This is extraordinarily useful for capacity planning:

```
Your API:
  Throughput λ = 1,000 requests/second
  Latency W = 100ms = 0.1 seconds

  Concurrency L = 1,000 × 0.1 = 100 concurrent requests in the system

Your thread pool has 200 threads → you have headroom.
Your thread pool has 50 threads → you are at 2× capacity → requests are queuing.

If latency increases to 500ms (slow DB):
  L = 1,000 × 0.5 = 500 concurrent requests
  Thread pool: 50 threads → 10× overloaded → timeout cascade
```

This shows why fixing latency (slow queries) is often more important than adding capacity. A 5× latency increase requires 5× more concurrency capacity to handle the same throughput.

---

## 6. The Decision: Latency or Throughput?

| Optimize for latency when | Optimize for throughput when |
|--------------------------|------------------------------|
| User-facing API responses | Batch ETL / data pipelines |
| Real-time features (chat, gaming, trading) | Log ingestion, analytics |
| Any interactive request where the user waits | Background jobs |
| Payment processing (must be fast and reliable) | Bulk data exports |
| p99 latency is in your SLA | Total jobs per hour is in your SLA |

### The p50 vs p99 distinction matters for user experience

```
Optimizing for average (p50) latency:
  p50: 10ms (50% of requests are this fast)
  p99: 5,000ms (1% of requests take 5 seconds)

  "Average looks great! Why are users complaining?"
  → 1 in 100 users is having a terrible experience
  → At 10M DAU and 100 req/user/day: 10M × 100 × 0.01 = 10M bad experiences/day

Optimizing for p99 latency (what you should do for user-facing APIs):
  Design so that p99 is acceptable, not just p50.
  Set SLOs on p99: "99% of requests complete in < 500ms"
```

---

## Summary

- Latency = time per operation. Throughput = operations per unit time. They conflict.
- Batching increases throughput (fewer round trips, amortized overhead) at the cost of latency (items wait in batch)
- Streaming decreases latency (process immediately) at the cost of throughput (per-item overhead)
- `linger_ms` in Kafka is a literal latency/throughput dial — explicit tradeoff in config
- Little's Law: `Concurrency = Throughput × Latency` — fixing latency (slow queries) reduces required concurrency
- Optimize for latency on user-facing APIs; optimize for throughput on background pipelines
- Measure and set SLOs on p99, not p50 — averages hide the worst user experiences

---

> Next: [Lesson 8.3 — SQL vs NoSQL](./lesson-8.3-sql-vs-nosql.md)