# Lesson 3.2 — Indexing — The Single Biggest Performance Lever

> **Chapter 3 — The Data Layer**
> Previous: [Lesson 3.1 — How Databases Work Internally](./lesson-3.1-how-databases-work-internally.md) | Next: [Lesson 3.3 — Read Replicas](./lesson-3.3-read-replicas.md)

---

## What this lesson covers

- How indexes work and when they help vs hurt
- Types of indexes: single column, composite, partial, covering
- The N+1 query problem — the most common and most damaging query bug
- How to find missing indexes in your database
- When NOT to add an index

---

## 1. The Index as a Sorted Lookup Structure

An index is a separate data structure maintained alongside your table. It stores a sorted copy of one or more columns, with pointers back to the original rows.

Think of it like a book's index at the back. Instead of reading every page to find "caching", you look it up in the index, get page numbers, and go directly there.

```
Table: orders (10 million rows)
Columns: id, user_id, status, created_at, total_amount

Query: SELECT * FROM orders WHERE user_id = 12345;

Without index on user_id:
  → Read all 10M rows, check each one
  → ~800MB of data read
  → Time: 2–30 seconds

With index on user_id:
  → B-tree lookup for user_id=12345 → get page locations
  → Read only those pages (maybe 50 orders for this user)
  → Time: ~5ms
```

This is not a small difference. It is a 400–6000× improvement for this query.

---

## 2. Types of Indexes

### 2.1 Single-Column Index

```sql
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

Speeds up queries that filter or sort by `user_id` alone.

```sql
-- Uses the index ✅
SELECT * FROM orders WHERE user_id = 42;
SELECT * FROM orders WHERE user_id = 42 ORDER BY user_id;

-- Does NOT use the index ❌
SELECT * FROM orders WHERE status = 'pending';  -- different column
```

---

### 2.2 Composite Index (Multi-Column Index)

An index on two or more columns together.

```sql
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
```

**The left-prefix rule:** A composite index on `(user_id, status)` can be used by queries that filter on:
- `user_id` alone ✅
- `user_id` AND `status` ✅
- `status` alone ❌ (cannot use this index without the leading column)

```sql
-- Uses the index ✅
SELECT * FROM orders WHERE user_id = 42;
SELECT * FROM orders WHERE user_id = 42 AND status = 'pending';

-- Does NOT use the index ❌
SELECT * FROM orders WHERE status = 'pending';
```

**Order matters.** Put the most selective column first (the one that filters out the most rows). Put equality conditions before range conditions.

```sql
-- Good: equality first, then range ✅
CREATE INDEX ON orders(user_id, created_at);
SELECT * FROM orders WHERE user_id = 42 AND created_at > '2024-01-01';

-- Bad: range first breaks the composite benefit ❌
CREATE INDEX ON orders(created_at, user_id);
SELECT * FROM orders WHERE created_at > '2024-01-01' AND user_id = 42;
-- Index only helps with created_at, then scans all matching rows for user_id
```

---

### 2.3 Partial Index

An index on a subset of rows — only rows where a condition is true.

```sql
-- Only index pending orders (not completed ones)
CREATE INDEX idx_orders_pending ON orders(user_id)
WHERE status = 'pending';
```

**When to use it:** If you almost always query a specific subset (e.g. active users, unprocessed jobs, pending orders), a partial index is smaller and faster than a full index. A jobs table might have 100M completed rows and only 10K pending rows — a partial index on pending rows is tiny and blazing fast.

```sql
-- Very common pattern: index on soft-deleted tables
CREATE INDEX idx_users_active ON users(email)
WHERE deleted_at IS NULL;
```

---

### 2.4 Covering Index (Index-Only Scan)

A covering index includes all the columns a query needs, so PostgreSQL never needs to read the heap (the actual table). The entire answer comes from the index itself.

```sql
-- Query needs: user_id (filter) + total_amount (select)
SELECT total_amount FROM orders WHERE user_id = 42;

-- Regular index on user_id:
--   Step 1: B-tree lookup for user_id=42 → get heap page locations
--   Step 2: Read heap pages to get total_amount
--   (two data structure reads)

-- Covering index includes total_amount:
CREATE INDEX idx_orders_user_total ON orders(user_id) INCLUDE (total_amount);
--   Step 1: B-tree lookup → total_amount is right there in the leaf node
--   (one data structure read — no heap access)
```

Use `INCLUDE` for columns frequently selected alongside the indexed column. This is a significant optimization for read-heavy queries.

---

### 2.5 Unique Index

Enforces uniqueness and provides fast lookups. Created automatically by `PRIMARY KEY` and `UNIQUE` constraints.

```sql
CREATE UNIQUE INDEX idx_users_email ON users(email);
-- or equivalently:
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);
```

---

## 3. The N+1 Query Problem

This is the most common and most damaging query bug in application code. It causes exponential database load and is often invisible in development (small datasets) but catastrophic in production.

### What N+1 looks like

```python
# You want to display a list of 100 posts with their authors

# Step 1: Get all posts — 1 query
posts = db.query("SELECT * FROM posts LIMIT 100")

# Step 2: For each post, get the author — 100 queries!
for post in posts:
    post.author = db.query(f"SELECT * FROM users WHERE id = {post.user_id}")
    # This runs once per post = 100 separate DB round trips
```

Total queries: 1 (posts) + 100 (authors) = **101 queries** to display 100 posts.

In development with 10 posts and a fast local DB: 11 queries, ~50ms. Invisible.
In production with 100 posts and network latency to the DB: 101 queries × 5ms = **505ms** just for DB round trips. Users see a slow page.

Scale this to 1,000 concurrent users each loading 100 posts: **101,000 queries per second** hitting your database. This alone can bring down a production database.

### How to fix N+1 — use a JOIN

```sql
-- Single query that gets posts AND authors together
SELECT posts.*, users.name, users.avatar_url
FROM posts
JOIN users ON posts.user_id = users.id
LIMIT 100;
```

One query. One round trip. The database does the join efficiently, especially if `users.id` is indexed (it always is — it's the primary key).

### How to detect N+1 in your app

Most ORMs have N+1 detection tools:

```
Django: django-debug-toolbar shows query count per request
Rails:  bullet gem warns about N+1 queries in development
Node:   logging middleware that counts queries per request
```

A simple rule: **if query count per request scales with the number of rows returned, you have an N+1 problem.**

---

## 4. Finding Missing Indexes in Production

### Method 1 — pg_stat_user_tables (sequential scans)

```sql
SELECT relname AS table,
       seq_scan,
       seq_tup_read,
       idx_scan,
       round(seq_scan::numeric / nullif(seq_scan + idx_scan, 0) * 100, 2) AS seq_scan_pct
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 20;
```

Tables with high `seq_tup_read` are doing sequential scans and reading many rows. If `seq_scan_pct` is high on a large table, it needs an index.

### Method 2 — pg_stat_statements (slow queries)

```sql
-- Enable the extension first: CREATE EXTENSION pg_stat_statements;

SELECT query,
       calls,
       round(total_exec_time::numeric / calls, 2) AS avg_ms,
       round(total_exec_time::numeric, 2) AS total_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

This shows which queries consume the most total time. A query that takes 500ms but runs 10,000 times per day is more important to fix than one that takes 5 seconds but runs once a day.

### Method 3 — EXPLAIN ANALYZE the slow queries you found

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE user_id = 42 AND status = 'pending';
```

Look for:
- `Seq Scan` on a large table → needs an index
- High `Buffers: shared hit / read` ratio → working set may not fit in buffer pool
- `rows=10000` estimated but `rows=1` actual → stale statistics, run `ANALYZE`

---

## 5. Index Selectivity — Why Some Indexes Don't Help

**Selectivity** is the fraction of rows an index eliminates. A highly selective index filters out most rows. A low-selectivity index barely helps.

```
Table: users (1,000,000 rows)

Column: country (200 distinct values, ~5,000 rows per country)
  Selectivity: 1/200 = 0.5%
  Index on country: USEFUL ✅ — filters 99.5% of rows

Column: is_active (2 distinct values: true/false, 950,000 true)
  Selectivity: 1/2 = 50%
  Index on is_active: NOT USEFUL ❌
  → For WHERE is_active = true: returns 950,000 rows (95% of table)
  → Faster to just scan the table sequentially
```

**The rule:** PostgreSQL will ignore your index and do a sequential scan if the index is not selective enough. This is correct behavior — a sequential scan is sometimes faster than using a low-selectivity index (because index reads involve random I/O, while sequential scans can use prefetching).

For low-selectivity columns, consider a **partial index** instead:

```sql
-- Index only inactive users (rare, ~50,000)
CREATE INDEX idx_users_inactive ON users(email)
WHERE is_active = false;

-- Now this query uses the index efficiently
SELECT * FROM users WHERE is_active = false AND email LIKE 'test%';
```

---

## 6. When NOT to Add an Index

Indexes are not free. Every index you add:

| Cost | Description |
|------|-------------|
| **Write overhead** | Every INSERT, UPDATE, DELETE must also update all indexes on that table. A table with 10 indexes has 10× the write overhead. |
| **Storage** | An index can be as large as the table itself |
| **VACUUM overhead** | Dead tuples in indexes must also be cleaned |
| **Planning time** | More indexes = more plans the optimizer must consider |

### Do NOT add an index when:

- The table is small (< 10,000 rows) — sequential scans are fine and fast
- The column has very low selectivity (boolean, status with 2-3 values, is_deleted)
- The table is write-heavy and rarely read (logs, events table that is only bulk-exported)
- The query already uses another index that is selective enough
- You are adding it "just in case" — index only columns used in actual slow queries

### The write overhead in practice

```
Table: events (write-heavy, 50,000 inserts/second)
With 1 index:  50,000 index writes/sec — manageable
With 5 indexes: 250,000 index writes/sec — significant overhead
With 10 indexes: 500,000 index writes/sec — may become the bottleneck
```

On high-write tables, audit indexes regularly and drop ones that are not being used:

```sql
-- Find unused indexes
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexname NOT LIKE 'pg_%'
ORDER BY schemaname, tablename;
```

An index with `idx_scan = 0` has never been used since the last statistics reset. It is safe to drop.

---

## 7. Indexes on Foreign Keys — The Forgotten Optimization

Foreign key columns are almost always involved in JOIN operations. PostgreSQL does NOT automatically create indexes on foreign key columns (only on the referenced primary key). You must add them manually.

```sql
-- posts.user_id references users.id
-- PostgreSQL auto-indexes users.id (primary key)
-- PostgreSQL does NOT auto-index posts.user_id

-- This JOIN scans all posts to find those matching user_id:
SELECT * FROM posts JOIN users ON posts.user_id = users.id WHERE users.id = 42;

-- Fix:
CREATE INDEX idx_posts_user_id ON posts(user_id);
```

**Rule: always index foreign key columns.** This is one of the most commonly missed optimizations.

---

## Summary

- An index is a B-tree structure that allows O(log n) lookups instead of O(n) full table scans
- Composite index column order matters: left-to-right, equality before range
- Partial indexes are smaller and faster when you consistently query a subset of rows
- Covering indexes eliminate heap access entirely for certain queries
- N+1 queries are the most common performance bug — fix them with JOINs or batch loading
- Low-selectivity columns (boolean, status with few values) do not benefit from regular indexes
- Every index adds write overhead — only add indexes that serve actual slow queries
- Always index foreign key columns — PostgreSQL does not do this automatically

---

## ⚠️ Common Mistakes

- Adding an index after noticing a slow query without checking if selectivity is high enough to matter
- Forgetting to index foreign key columns — makes every JOIN slow
- Having 15+ indexes on a write-heavy table — write performance degrades severely
- Not running `ANALYZE` after bulk data loads — stale statistics cause the planner to pick wrong indexes
- Using `SELECT *` in queries that could use covering indexes — forces heap reads even when an index-only scan would work

---

> Next: [Lesson 3.3 — Read Replicas](./lesson-3.3-read-replicas.md)