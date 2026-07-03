# Lesson 3.9 — Database Anti-Patterns

> **Chapter 3 — The Data Layer**
> Previous: [Lesson 3.8 — Schema Design for Scale](./lesson-3.8-schema-design.md) | Next: [Chapter 4 — The Caching Layer](../chapter-4/lesson-4.1-how-caching-works.md)

---

## What this lesson covers

- The most damaging database anti-patterns and exactly why they cause problems
- How to identify each anti-pattern in your own codebase
- How to fix each one
- The anti-patterns that are invisible in development but catastrophic in production

---

## 1. SELECT * — The Invisible Tax

`SELECT *` returns every column in a table. It feels convenient and saves typing. At scale, it quietly wastes enormous resources.

### Why it is harmful

```sql
-- users table has 40 columns including:
-- profile_photo BYTEA (stores image blob -- 50KB each)
-- bio TEXT (potentially thousands of characters)
-- ...

-- You only need name and email for this API endpoint
SELECT * FROM users WHERE id = 42;
-- Returns: 50KB of photo + long bio + 38 other columns you don't need

-- vs
SELECT id, name, email FROM users WHERE id = 42;
-- Returns: ~100 bytes
```

**The costs:**
- **Network bandwidth:** Transferring 50KB instead of 100 bytes per row, multiplied by thousands of requests per second
- **Memory:** Database must load entire rows into buffer pool, evicting useful cached pages
- **Serialization:** Application deserializes all 40 columns into objects when it only uses 2
- **Index bypass:** Covering index optimizations (Lesson 3.2) are defeated because `SELECT *` requires going to the heap

**The fix:** Always specify the columns you need.

```sql
-- Always name your columns
SELECT id, name, email, created_at FROM users WHERE id = 42;
```

This is also better for long-term maintainability — when a column is added to or removed from the table, `SELECT *` silently changes behavior while named columns fail loudly if a column is removed.

---

## 2. No Pagination — The Unbounded Query

Fetching all rows from a large table in one query. This is invisible in development (100 rows) and catastrophic in production (10 million rows).

```python
# Development: 100 users — returns instantly
users = db.query("SELECT * FROM users ORDER BY created_at DESC")
# Returns 100 rows, ~1ms, no problem

# Production: 10 million users — disaster
users = db.query("SELECT * FROM users ORDER BY created_at DESC")
# Returns 10M rows
# - Query runs for minutes
# - Loads 10M rows into application memory
# - App server runs out of memory and crashes
# - Database is occupied serving this one query for everyone
```

### Fix — Always paginate

**Offset-based pagination (simple, with limitations):**

```sql
-- Page 1
SELECT id, name, email FROM users ORDER BY created_at DESC LIMIT 20 OFFSET 0;
-- Page 2
SELECT id, name, email FROM users ORDER BY created_at DESC LIMIT 20 OFFSET 20;
-- Page N
SELECT id, name, email FROM users ORDER BY created_at DESC LIMIT 20 OFFSET (N-1)*20;
```

**Problem with OFFSET:** For `OFFSET 10000`, the database must read and discard 10,000 rows before returning results. Deep pages become progressively slower.

**Cursor-based pagination (keyset pagination — recommended):**

```sql
-- First page: no cursor
SELECT id, name, email, created_at FROM users
ORDER BY created_at DESC, id DESC
LIMIT 20;
-- Returns rows, last row has created_at='2024-01-15 08:30:00', id=9999

-- Next page: use last row's values as cursor
SELECT id, name, email, created_at FROM users
WHERE (created_at, id) < ('2024-01-15 08:30:00', 9999)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

Cursor pagination is O(1) regardless of page depth. The index is used efficiently. This is how Twitter, Instagram, and most social feeds implement infinite scroll.

---

## 3. No Indexes on Foreign Keys

PostgreSQL does not automatically create indexes on foreign key columns. This means every JOIN operation that uses a foreign key becomes a sequential scan.

```sql
CREATE TABLE posts (
    id      BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),  -- FK, no index created
    title   TEXT,
    body    TEXT
);

-- This JOIN does a sequential scan on posts to find all rows where user_id = 42
SELECT p.title, u.name
FROM posts p
JOIN users u ON p.user_id = u.id
WHERE u.id = 42;

-- Execution plan:
-- Seq Scan on posts (cost=0.00..25000.00)  ← reads ALL posts
--   Filter: (user_id = 42)
```

With 10 million posts and 1 million users, this scan reads all 10 million posts for every user profile view.

**The fix:** Always index foreign key columns immediately after creating them.

```sql
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- Now the same query:
-- Index Scan on posts using idx_posts_user_id
--   Index Cond: (user_id = 42)
-- Reads only the 50 posts by this user, not all 10M
```

---

## 4. Using the Database as a Queue

Polling a database table for new jobs to process is a common anti-pattern.

```sql
-- Jobs table
CREATE TABLE jobs (
    id          BIGSERIAL PRIMARY KEY,
    payload     JSONB,
    status      VARCHAR(20) DEFAULT 'pending',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Worker polls for pending jobs
SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED;
UPDATE jobs SET status = 'processing' WHERE id = $1;
-- process job...
UPDATE jobs SET status = 'completed' WHERE id = $1;
```

This pattern seems reasonable and works at low scale. At high scale it causes serious problems:

| Problem | Description |
|---------|-------------|
| **Polling overhead** | Workers constantly query the DB even when there are no jobs — wasted load |
| **Lock contention** | Multiple workers compete for the same rows — `FOR UPDATE` creates lock waits |
| **Dead jobs** | If a worker crashes after claiming a job, the job stays in 'processing' forever unless you build a timeout cleanup |
| **Table bloat** | Completed jobs accumulate and slow down the `WHERE status = 'pending'` query |
| **No fan-out** | You cannot easily broadcast one job to multiple consumers |

**The fix:** Use a real message queue — Kafka, RabbitMQ, or SQS.

Real queues give you: push delivery (no polling), at-least-once delivery with retry, dead letter queues, consumer groups, and horizontal scaling of consumers. These are covered in Chapter 5.

---

## 5. Long-Running Transactions

A transaction that stays open for a long time causes cascading problems.

```python
# This transaction is open for the duration of a long API call
with db.transaction():
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # This external API call takes 2 seconds
    stripe_response = stripe.charge(user.card_token, amount=100)
    
    db.execute("INSERT INTO payments ...", stripe_response)
    # Transaction commits here — 2+ seconds after it opened
```

**What happens during the 2 seconds this transaction is open:**

| Problem | Description |
|---------|-------------|
| **Row locks held** | Any rows locked by this transaction block other transactions from updating them |
| **MVCC dead tuples** | Other transactions cannot vacuum dead tuples created before this transaction's snapshot |
| **Connection held** | A database connection is held for 2 seconds — with a pool of 20 connections, 20 long transactions = total pool exhaustion |
| **Replication delay** | Very long transactions can cause replicas to lag |

**The fix:** Keep transactions short. Never hold a transaction open during external I/O.

```python
# Wrong: transaction spans external API call
with db.transaction():
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    stripe_response = stripe.charge(...)  # 2 second external call INSIDE transaction
    db.execute("INSERT INTO payments ...", stripe_response)

# Right: external call happens BEFORE the transaction
user = db.query("SELECT * FROM users WHERE id = ?", user_id)  # read outside transaction
stripe_response = stripe.charge(...)  # external call outside transaction

with db.transaction():  # transaction is open for microseconds
    db.execute("INSERT INTO payments ...", stripe_response)
```

---

## 6. The God Table — Storing Everything in One Table

A "god table" is a single wide table with dozens or hundreds of columns that attempts to represent many different things. Often created to "avoid joins" or "keep things simple."

```sql
-- God table: tries to represent both individual users and business accounts
CREATE TABLE entities (
    id                  BIGSERIAL PRIMARY KEY,
    type                VARCHAR(20),  -- 'individual' or 'business'
    first_name          TEXT,   -- only for individuals
    last_name           TEXT,   -- only for individuals
    business_name       TEXT,   -- only for businesses
    tax_id              TEXT,   -- only for businesses
    email               TEXT,
    phone               TEXT,
    billing_address     TEXT,
    personal_bio        TEXT,   -- only for individuals
    num_employees       INT,    -- only for businesses
    industry            TEXT,   -- only for businesses
    -- ... 40 more columns, half null for any given row
);
```

**Problems:**
- Most columns are NULL for any given row — wasted storage and confusing schema
- Business logic for individuals and businesses is mixed — hard to reason about
- Indexing is complex — which columns matter for which entity types?
- Queries require `WHERE type = 'individual'` everywhere — easy to forget
- Adding a column for one type affects the schema for all types

**The fix:** Separate tables or table inheritance.

```sql
-- Separate tables (simplest)
CREATE TABLE users (
    id        BIGSERIAL PRIMARY KEY,
    email     TEXT UNIQUE NOT NULL,
    first_name TEXT NOT NULL,
    last_name  TEXT NOT NULL
);

CREATE TABLE businesses (
    id           BIGSERIAL PRIMARY KEY,
    email        TEXT UNIQUE NOT NULL,
    business_name TEXT NOT NULL,
    tax_id       TEXT,
    num_employees INT
);

-- Or: shared identity table + type-specific tables (table inheritance pattern)
CREATE TABLE accounts (
    id    BIGSERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    type  VARCHAR(20) NOT NULL
);
CREATE TABLE user_profiles (
    account_id BIGINT PRIMARY KEY REFERENCES accounts(id),
    first_name TEXT, last_name TEXT, bio TEXT
);
CREATE TABLE business_profiles (
    account_id    BIGINT PRIMARY KEY REFERENCES accounts(id),
    business_name TEXT, tax_id TEXT, num_employees INT
);
```

---

## 7. Storing JSON for Everything

PostgreSQL's JSONB column type is powerful and genuinely useful. It is also frequently misused as a way to avoid schema design.

```sql
-- Anti-pattern: entire row as JSON "for flexibility"
CREATE TABLE users (
    id   BIGSERIAL PRIMARY KEY,
    data JSONB  -- stores everything: name, email, preferences, addresses, history
);

-- You cannot:
-- Index efficiently on json fields without GIN indexes (which are large and slow to update)
-- Enforce data types (a JSON field can be a string, number, or null interchangeably)
-- Use foreign keys on JSON fields
-- Run efficient aggregates (SUM, AVG) on JSON numeric fields
```

**When JSONB is appropriate:**
- Storing truly variable, unstructured data that differs per row (user preferences, feature flags, metadata)
- Storing third-party API responses you need to preserve but do not query deeply
- Audit logs where the schema of the audited data can change over time

```sql
-- Right: use JSONB for the variable part, proper columns for the queried part
CREATE TABLE products (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT NOT NULL,       -- always queried — proper column
    price       DECIMAL(10,2),       -- always queried — proper column
    category_id BIGINT,              -- always joined on — proper column
    attributes  JSONB                -- variable: {"color": "red", "size": "L", "material": "cotton"}
                                     -- queried rarely — JSONB is fine
);

-- Create GIN index only if you frequently query specific JSON fields
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);
```

---

## 8. Not Using Database Constraints

Enforcing data integrity in application code rather than in the database is fragile. The database is the last line of defense — if invalid data gets in, it is very hard to fix.

```sql
-- Without constraints: broken data gets in
-- Application code: "I'll check if email is unique before inserting"
-- Race condition: two requests check simultaneously, both see "no duplicate", both insert → duplicate

-- With constraints: database rejects invalid data regardless of application logic
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,         -- database enforces uniqueness
    age         INT CHECK (age >= 0 AND age < 150),  -- database enforces valid range
    status      TEXT CHECK (status IN ('active', 'suspended', 'deleted')),
    created_at  TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Foreign key constraint: database rejects orphaned records
CREATE TABLE orders (
    id      BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT
    -- ON DELETE RESTRICT: cannot delete user if they have orders
);
```

Use `NOT NULL`, `UNIQUE`, `CHECK`, `FOREIGN KEY`, and `DEFAULT` constraints liberally. They are enforced atomically by the database and cannot be bypassed by buggy application code.

---

## 9. Ignoring the Slow Query Log

PostgreSQL has a built-in slow query log that records queries taking longer than a threshold. Most teams never configure it. Those that do rarely review it.

```sql
-- In postgresql.conf:
log_min_duration_statement = 100   -- log queries taking > 100ms
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

-- Or set per session for investigation:
SET log_min_duration_statement = 0;  -- log ALL queries in this session
```

Review the slow query log weekly. The top 5 slowest queries by total time (not just per-execution time — a query taking 10ms but running 100K times/day costs more than one taking 5 seconds and running once) are almost always fixable with an index or a query rewrite.

```sql
-- pg_stat_statements extension summarizes this for you:
SELECT query,
       calls,
       round(mean_exec_time::numeric, 2) AS avg_ms,
       round(total_exec_time::numeric, 2) AS total_ms,
       round(stddev_exec_time::numeric, 2) AS stddev_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

---

## Summary Table — Anti-Patterns at a Glance

| Anti-pattern | Root cause | Impact | Fix |
|-------------|-----------|--------|-----|
| SELECT * | Laziness / habit | Bandwidth waste, buffer pool pollution | Name your columns |
| No pagination | Works fine in dev | OOM crashes, full table scans in prod | Always LIMIT + cursor |
| No FK indexes | PostgreSQL doesn't auto-create them | Slow JOINs on every request | Index every FK column |
| DB as queue | Convenient at first | Lock contention, polling overhead, dead jobs | Use Kafka / SQS / RabbitMQ |
| Long transactions | External I/O inside transaction | Lock contention, pool exhaustion | Keep transactions short, do I/O outside |
| God table | "Avoid joins" thinking | Null-heavy rows, mixed logic, poor indexability | Separate tables per entity type |
| JSON for everything | "Flexible schema" thinking | Unindexable, no constraints, no aggregates | Use proper columns for queried fields |
| No constraints | "App code handles it" | Invalid data gets in, race conditions | Use DB constraints as last line of defense |
| No slow query log | Not configured | Slow queries go undetected for months | Configure log_min_duration_statement |

---

## ✅ Chapter 3 Complete

Chapter 3 has covered the full depth of the data layer:

- **3.1** How PostgreSQL stores data (pages, buffer pool, WAL, MVCC)
- **3.2** Indexing — the highest-leverage optimization (B-trees, composite, partial, covering indexes, N+1)
- **3.3** Read replicas — scaling reads, replication lag, failover
- **3.4** Connection pooling — PgBouncer, transaction mode, pool sizing
- **3.5** Sharding — when to use it, strategies, the problems it introduces
- **3.6** NoSQL — four families, when each fits, polyglot persistence
- **3.7** CAP theorem — the real choice between consistency and availability
- **3.8** Schema design — normalization, migration safety, data types, soft deletes
- **3.9** Anti-patterns — the nine most common database mistakes and how to fix them

The database is the hardest component to scale and the source of most production performance problems. The concepts in this chapter will serve you throughout your entire engineering career.

---

> Next Chapter: [Chapter 4 — The Caching Layer](../chapter-4/lesson-4.1-how-caching-works.md)

---

> Previous: [Lesson 3.8 — Schema Design for Scale](./lesson-3.8-schema-design.md)