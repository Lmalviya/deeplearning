# Lesson 8.3 — SQL vs NoSQL

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 8.2 — Latency vs Throughput](./lesson-8.2-latency-vs-throughput.md) | Next: [Lesson 8.4 — Sync vs Async](./lesson-8.4-sync-vs-async.md)

---

## What this lesson covers

- The decision framework — it is access pattern matching, not a religious debate
- When SQL genuinely wins and when NoSQL genuinely wins
- The hidden costs of NoSQL that teams discover too late
- Polyglot persistence — how to use both correctly
- Red flags that tell you the wrong database was chosen

---

## 1. The Actual Question to Ask

The wrong question: "Should we use SQL or NoSQL?"

The right questions:
1. What are my primary access patterns? (How will I query this data most often?)
2. Do I need ACID transactions across multiple entities?
3. Will my schema be stable or frequently changing?
4. What is my write throughput requirement?
5. Do I need to join different types of data together?

Answer these, and the right database becomes obvious.

---

## 2. SQL Wins — These Access Patterns Always Favor SQL

### When you need joins

```sql
-- "Show me all orders for users who signed up in 2024 with their product details"
SELECT u.name, o.id, p.name, oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE u.created_at >= '2024-01-01'

-- In a document DB: requires multiple queries + application-level join
-- In SQL: one query, optimized by the query planner
```

Any time you need to combine data from multiple entities in a flexible way — SQL wins. The query planner in PostgreSQL can optimize joins in ways that application-level joins cannot match.

### When you need ACID transactions across entities

```sql
-- Transfer money: must be atomic
BEGIN;
UPDATE accounts SET balance = balance - 500 WHERE id = 1;
UPDATE accounts SET balance = balance + 500 WHERE id = 2;
COMMIT;
-- Either both happen or neither happens
```

Document databases have added multi-document transactions, but SQL databases were designed for this. If your core operations span multiple records and must be atomic, SQL is the right tool.

### When you need ad-hoc queries

In SQL you can query any column in any way, filtered, joined, aggregated — even combinations you did not anticipate when designing the schema.

In NoSQL, you must decide your query patterns upfront. A query you did not design for may require a full collection scan or may be impossible without additional indexes.

### When your data is relational by nature

Users → Orders → Products → Reviews → Tags — these entities have natural relationships. SQL is designed for exactly this. Forcing relational data into a document model means either:
- Denormalization (duplicate data, complex updates)
- Application-level joins (slow, complex code)

---

## 3. NoSQL Wins — These Access Patterns Always Favor NoSQL

### When you have a single primary key access pattern with high throughput

```
"Give me all data for user_id = 42"
→ Simple key lookup, no joins
→ DynamoDB or Redis handles this with sub-millisecond latency at any scale
→ PostgreSQL is overkill and slower at pure key lookups
```

DynamoDB's promise: single-digit millisecond latency at any scale for key lookups. PostgreSQL at scale might be 5–20ms due to network, connection overhead, and query planning.

### When write throughput exceeds what a single SQL primary can handle

```
IoT sensors: 10,000 devices × 10 updates/second = 100,000 writes/second
→ PostgreSQL primary: maxes out at ~10,000–50,000 writes/second
→ Cassandra: designed for this — scales horizontally, 1M+ writes/second
```

### When your schema genuinely varies per record

```json
// Product catalog: different products have completely different attributes
{"id": "tv_01", "brand": "Samsung", "screen_size": 55, "resolution": "4K", "smart_tv": true}
{"id": "shirt_01", "brand": "Uniqlo", "size": "L", "color": "blue", "material": "cotton"}
{"id": "book_01", "title": "Clean Code", "author": "Martin", "isbn": "978-0132350884"}
```

Storing these in a SQL table requires either: a generic key-value column (loses type safety), nullable columns for every possible attribute (wide sparse table), or EAV anti-pattern. A document store handles this naturally.

### When you need time-series at scale

```
Metrics: 1,000 servers × 50 metrics × 1/second = 50,000 inserts/second
→ Cassandra or InfluxDB: designed for sequential time-series appends, fast range queries by time
→ PostgreSQL: possible but requires careful partitioning and indexing, struggles at this insert rate
```

---

## 4. The Hidden Costs of NoSQL — What Teams Discover Too Late

Teams often choose NoSQL for perceived simplicity or scalability, then discover unexpected costs:

### Cost 1 — You must know your queries upfront

```
MongoDB collection: orders
You designed it for: "give me all orders for user X"
Business asks for: "give me all orders over $500 that contain product Y across all users"

→ No index designed for this query
→ Collection scan on 10M documents
→ 30 second query
→ Cannot add an index efficiently because the collection is too large to build without impact

In SQL: just add an index on (total_amount) and (product_id) → done
```

### Cost 2 — Denormalization maintenance

```json
// You denormalized user name into every order for fast reads
{
  "order_id": "ord_123",
  "user_name": "Alice Chen",   ← copied from users collection
  "user_id": "user_42",
  "total": 299.98
}

// User changes their name to "Alice Smith"
// You must now update every order document containing this user's name
// How many orders does Alice have? 500? 5000?
// How do you do this atomically?
// What if the update fails halfway through?
```

This maintenance burden is proportional to your data volume and update frequency. In SQL with a JOIN, the user name is always fresh — no maintenance needed.

### Cost 3 — No transactions across documents

```python
# Place an order: deduct inventory and create order record
# In SQL: one transaction, atomic
with db.transaction():
    db.execute("UPDATE inventory SET quantity = quantity - 1 WHERE product_id = ?", pid)
    db.execute("INSERT INTO orders ...")

# In MongoDB (without multi-document transactions):
inventory.update_one({"_id": pid}, {"$inc": {"quantity": -1}})
# Server crashes here
orders.insert_one(order_data)
# Inventory decremented but order not created → inconsistent state
```

MongoDB added multi-document transactions in 4.0, but they are limited, slower, and less mature than PostgreSQL's transaction model.

---

## 5. The Decision Matrix

```
Start here: What are my most frequent queries?

Key lookup only (GET by ID):
  → DynamoDB or Redis (fastest at scale)

Key lookup + simple filters on one table:
  → DynamoDB (with GSI) or PostgreSQL (with index)

Multiple entity joins, complex filters, ad-hoc:
  → PostgreSQL

Time-series, sequential append, time-range reads:
  → Cassandra, InfluxDB, or PostgreSQL with partitioning

Full-text search and relevance ranking:
  → Elasticsearch (alongside your primary DB)

Flexible per-document schema, catalog-style:
  → MongoDB or PostgreSQL JSONB column

Graph traversal (friends of friends, permissions):
  → Neo4j (or PostgreSQL with recursive CTEs for simpler cases)

High write throughput (>10K writes/sec), simple access:
  → Cassandra or DynamoDB
```

---

## 6. Polyglot Persistence — The Real-World Answer

Most mature systems use multiple databases, each for what it does best:

```
E-commerce platform:
  PostgreSQL      → orders, users, payments (relational, ACID-critical)
  Redis           → sessions, cart, rate limiting (in-memory, speed-critical)
  Elasticsearch   → product search (full-text, relevance-critical)
  DynamoDB        → user events, click stream (key-value, volume-critical)
  S3              → product images, PDFs (blob, cost-critical)

Each database is chosen for its specific workload.
No single database is "best" for all workloads.
```

The operational cost of running multiple database types is real. You need engineers who understand each one. The payoff is that each workload runs on infrastructure purpose-built for it.

---

## Summary

- SQL wins when: joins are needed, ACID across multiple entities, ad-hoc queries, relational data model
- NoSQL wins when: single key access at high throughput, write throughput exceeds SQL primary limits, flexible per-record schema, time-series at scale
- Hidden NoSQL costs: must know queries upfront, denormalization maintenance complexity, limited cross-document transactions
- The right answer is usually polyglot: SQL for core relational data, Redis for speed, Elasticsearch for search, Cassandra for time-series
- Never choose SQL vs NoSQL based on "scalability" alone — SQL scales further than most people realize, and NoSQL just shifts complexity from DB to application

---

> Next: [Lesson 8.4 — Sync vs Async](./lesson-8.4-sync-vs-async.md)

---

# Lesson 8.4 — Sync vs Async

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 8.3 — SQL vs NoSQL](./lesson-8.3-sql-vs-nosql.md) | Next: [Lesson 8.5 — Horizontal vs Vertical Scaling](./lesson-8.5-horizontal-vs-vertical.md)

---

## What this lesson covers

- When you must be synchronous and when async is always better
- The failure isolation benefit of async — the most underappreciated advantage
- The UX tradeoffs of async operations
- Designing async-first systems while keeping sync where it matters
- The "dual write" problem — the hardest part of going async

---

## 1. When You MUST Be Synchronous

Some operations cannot be made async because the user or the system needs the result to proceed.

| Operation | Why it must be sync |
|-----------|-------------------|
| Payment charge | User must know if charge succeeded before seeing "order confirmed" |
| Auth / login | Cannot serve a dashboard before knowing who the user is |
| Data validation | Must know if input is valid before accepting the form |
| Reservation / booking | Must confirm availability before user commits |
| File upload acknowledgement | User needs to know upload succeeded before navigating away |
| Database reads | Read results are needed to construct the response |

### The test for sync-or-async

> "Does the user need this result to proceed to the next step?"

If yes → sync. If no → async.

---

## 2. When You MUST Be Asynchronous

Some operations must be async because doing them synchronously either breaks the user experience or the system.

| Operation | Why it must be async |
|-----------|---------------------|
| Sending email / SMS | Email providers are slow (100–2000ms). User should not wait. |
| Video transcoding | Takes minutes to hours. HTTP request cannot wait. |
| Search index updates | Search is eventually consistent. Blocking on it delays the primary write. |
| Fan-out notifications | Notifying 1M followers synchronously would take hours. |
| Webhook delivery to third parties | Third-party systems can be slow or down. |
| Generating large reports | Can take 30 seconds+. Give user a job ID to poll or email when done. |
| ML inference on non-critical paths | Recommendation recompute, fraud score on non-blocking paths. |

---

## 3. The Failure Isolation Benefit

The most important (and underappreciated) benefit of async is **failure isolation**. A synchronous dependency that fails brings down your entire operation. An asynchronous dependency that fails affects only its own queue.

```
Synchronous email sending:
  POST /signup
    → create user in DB ✅
    → send welcome email via Sendgrid
    → Sendgrid is having an outage ❌
    → entire POST /signup fails ❌
    → user sees "Something went wrong"
    → user retries, creates duplicate accounts

Asynchronous email sending:
  POST /signup
    → create user in DB ✅
    → enqueue "send_welcome_email" job ✅
    → return 201 Created to user ✅

  Background (separately):
    → Worker picks up job
    → Sendgrid is down ❌
    → Job fails → retry queue
    → Wait 5 minutes → retry
    → Sendgrid recovers ✅
    → Email delivered ✅

User experience: signup works instantly, email arrives 5 minutes late.
Much better than "signup failed" during an email service outage.
```

---

## 4. The Dual Write Problem

When you move from sync to async, you often need to write to two places: your database and your message queue. Both must succeed, or you have inconsistency.

```python
# The naive approach — NOT safe:
def create_user(data):
    user = db.create_user(data)           # write to DB ✅
    queue.publish("user.created", user)   # write to queue
    # What if queue.publish fails? DB has the user, queue does not.
    # Email never sent. User never knows. Silently broken.
```

### Solution 1 — Transactional outbox pattern

Write to the queue via your database transaction, not directly to the queue:

```python
def create_user(data):
    with db.transaction():
        user = db.create_user(data)
        # Write the "event to publish" to a local outbox table — same transaction
        db.execute("""
            INSERT INTO outbox (event_type, payload, created_at, published)
            VALUES ('user.created', %s, NOW(), false)
        """, json.dumps(user))
    # Transaction commits: user AND outbox entry created atomically

# Separate background process (outbox relay):
def relay_outbox():
    pending = db.query("SELECT * FROM outbox WHERE published = false ORDER BY created_at LIMIT 100")
    for event in pending:
        queue.publish(event['event_type'], event['payload'])
        db.execute("UPDATE outbox SET published = true WHERE id = ?", event['id'])
```

The outbox table acts as a buffer. The relay process publishes events from the DB to the queue. If the relay fails, the event is still in the outbox — will be retried. The database and queue are eventually consistent, but the outbox guarantees nothing is lost.

### Solution 2 — Kafka transactions (if using Kafka)

```python
producer = KafkaProducer(transactional_id="my-producer")
producer.init_transactions()

with producer.transaction():
    db.execute("INSERT INTO users ...")
    producer.send("user.created", value=user_data)
# Both commit or both rollback — atomic cross-system transaction
```

---

## 5. The UX Design of Async Operations

When an operation is async, the user experience must be designed accordingly.

### Pattern 1 — Optimistic UI

Show the result immediately in the UI as if it succeeded, then confirm in the background.

```
User clicks "like" on a post
→ UI immediately shows +1 like (optimistic)
→ API sends async request to record like
→ If it fails: revert the UI counter
→ If it succeeds: confirm (UI already looked right)

User experience: instant, responsive
Risk: if the network request fails, UI briefly shows wrong count
```

### Pattern 2 — Job ID + polling

For long-running operations, return a job ID immediately and let the user poll for completion.

```
POST /reports/generate
→ Enqueues report generation job
→ Returns: {"job_id": "job_abc123", "status": "queued"}

Client polls:
GET /reports/job/abc123
→ {"status": "processing", "progress": 45}

GET /reports/job/abc123 (30 seconds later)
→ {"status": "complete", "download_url": "https://..."}
```

### Pattern 3 — Push notification on completion

For very long operations, tell the user "we'll email/notify you when it's done."

```
POST /exports/users
→ Returns: 202 Accepted {"message": "Export started. We'll email you when ready."}

User can close the browser.
Background worker generates export.
On completion: sends email with download link.
```

---

## Summary

- Must be sync: payment charges, auth, validation, booking confirmations — user needs the result to proceed
- Must be async: email, transcoding, search updates, fan-out, third-party webhooks — too slow or failure-prone to block user
- The biggest benefit of async is failure isolation — email service outage should not break user signup
- The dual write problem: use the transactional outbox pattern to atomically write to DB and guarantee event delivery
- UX patterns for async: optimistic UI (instant feedback, retry on failure), job ID + polling (progress visibility), push notification (fire-and-forget with completion callback)

---

> Next: [Lesson 8.5 — Horizontal vs Vertical Scaling](./lesson-8.5-horizontal-vs-vertical.md)