# Lesson 7.2 — 10K–100K DAU: The First Real Scale

> **Chapter 7 — Scale Tiers**
> Previous: [Lesson 7.1 — 1K–10K DAU](./lesson-7.1-1k-to-10k.md) | Next: [Lesson 7.3 — 100K–1M DAU](./lesson-7.3-100k-to-1m.md)

---

## What this lesson covers

- Why the database becomes the first bottleneck at this tier
- The exact sequence of fixes to apply and in what order
- Making the app stateless if it is not already
- Adding read replicas and caching
- The monitoring you must have before leaving this tier

---

## 1. What Changes at This Tier

At 10K DAU you were fine with a single database and no cache. At 100K DAU you are not.

Let us run the math from Lesson 0.1:

```
100K DAU × 50 requests/user/day = 5M requests/day
5M / 86,400 seconds = ~58 RPS average
Peak RPS = 58 × 3 = ~175 RPS

175 requests/second, each potentially hitting the database
A well-tuned PostgreSQL can handle 1K–5K simple queries/sec
BUT: if queries are unindexed, joins are slow, or N+1 exists → problems appear here
```

The bottleneck is not raw throughput yet — it is **query efficiency and connection count**. Fix these first before adding hardware.

---

## 2. The Fix Sequence — In Order

Do not jump straight to read replicas or caching. Work through this sequence:

```
Step 1: Fix slow queries (indexes, N+1)        ← free, highest leverage
Step 2: Add connection pooling (PgBouncer)      ← free, prevents connection exhaustion
Step 3: Add application caching (Redis)         ← moderate cost, huge DB load reduction
Step 4: Add a read replica                      ← moderate cost, scales reads
Step 5: Vertical scale the DB instance          ← higher cost, buys time
Step 6: Add more app servers                    ← if app is the bottleneck (rare at this tier)
```

Most teams jump to step 4 or 5 when step 1 alone would fix 80% of the problem.

---

## 3. Step 1 — Fix Slow Queries First

Before adding any infrastructure, spend one day on query analysis. The payoff is enormous.

```sql
-- Enable pg_stat_statements if not enabled
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find the top 10 most time-consuming queries
SELECT
    left(query, 100) AS query_snippet,
    calls,
    round(mean_exec_time::numeric, 2) AS avg_ms,
    round(total_exec_time::numeric / 1000, 2) AS total_sec
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

For each slow query, run `EXPLAIN ANALYZE` and look for:
- `Seq Scan` on tables with > 10K rows → add an index
- `Nested Loop` on large tables → check for missing join indexes
- High row estimates vs actual → run `ANALYZE tablename`

A single `CREATE INDEX CONCURRENTLY` on the right column can turn a 500ms query into a 2ms query. This costs nothing but time.

**Also check for N+1 queries (Lesson 3.2):** If query count per request scales with the number of results, you have N+1. Fix with JOINs or batch loading.

---

## 4. Step 2 — Add Connection Pooling

At 100K DAU with 3–5 app servers and 20 threads each, you are approaching PostgreSQL's connection limit without pooling.

```
3 app servers × 20 threads = 60 direct connections

At 5 app servers: 100 connections
At 10 app servers: 200 connections → PostgreSQL begins degrading
```

Add PgBouncer in transaction mode (Lesson 3.4):

```ini
# pgbouncer.ini — run as sidecar on each app server
[databases]
myapp = host=db.internal port=5432 dbname=myapp

[pgbouncer]
pool_mode = transaction
default_pool_size = 20    # 20 real DB connections per app server
max_client_conn = 500
listen_port = 6432
```

With this configuration, 10 app servers × 20 pool connections = 200 real DB connections regardless of how many threads are active.

---

## 5. Step 3 — Add Redis Caching (Cache-Aside)

At this tier the highest-impact cache targets are:

**User sessions (required for stateless design):**
```python
redis.setex(f"session:{token}", 1800, json.dumps(session_data))
```

**Frequently read, rarely changed data:**
```python
# User profiles: read on every page, changed rarely
@cache(ttl=300, key="user:{user_id}")
def get_user_profile(user_id):
    return db.query("SELECT * FROM users WHERE id = ?", user_id)

# Product catalog: same product read by thousands of users
@cache(ttl=3600, key="product:{product_id}")
def get_product(product_id):
    return db.query("SELECT * FROM products WHERE id = ?", product_id)

# Aggregates: expensive to compute, same for everyone
@cache(ttl=60, key="homepage:trending")
def get_trending_posts():
    return db.query("""
        SELECT id, title, view_count FROM posts
        ORDER BY view_count DESC LIMIT 20
    """)
```

**Target hit ratio > 85% for read-heavy workloads.** Use `redis-cli INFO stats` to monitor.

What happens to DB load after adding caching:

```
Before cache:
  175 RPS → 175 DB queries/second

After cache (85% hit ratio):
  175 RPS × 0.15 miss rate = 26 DB queries/second
  85% reduction in DB load
```

---

## 6. Step 4 — Add a Read Replica

By this point, if DB CPU is still high after indexing and caching, it is time for a read replica.

```
When to add a read replica:
  DB CPU > 70% sustained after adding cache and fixing indexes
  Read/write ratio is skewed (> 5:1)
  Specific read-heavy queries cannot be cached (user-specific, frequently changing)
```

Add a read replica via your managed DB provider (one click in RDS, Supabase, etc.).

Route reads explicitly:

```python
# In your ORM/DB config
DB_WRITE = "postgres://user:pass@primary.db:5432/myapp"
DB_READ  = "postgres://user:pass@replica.db:5432/myapp"

# In your data access layer
def get_user(user_id):
    # Read replica for non-critical reads
    return read_db.query("SELECT * FROM users WHERE id = ?", user_id)

def update_user(user_id, data):
    # Primary for writes
    write_db.execute("UPDATE users SET ... WHERE id = ?", user_id)
    # Invalidate cache
    redis.delete(f"user:{user_id}")
```

**Watch for replication lag (Lesson 3.3).** After writes, route reads to primary for a short window:

```python
def update_user(user_id, data):
    write_db.execute("UPDATE users ...")
    redis.delete(f"user:{user_id}")
    # Set a flag: "read from primary for next 2 seconds for this user"
    redis.setex(f"read_from_primary:{user_id}", 2, "1")

def get_user(user_id):
    use_primary = redis.exists(f"read_from_primary:{user_id}")
    db = write_db if use_primary else read_db
    # Try cache first regardless
    cached = redis.get(f"user:{user_id}")
    if cached: return json.loads(cached)
    return db.query("SELECT * FROM users WHERE id = ?", user_id)
```

---

## 7. The Architecture at This Tier

```mermaid
graph TD
    Users["Users\n10K–100K DAU"]
    CDN["Cloudflare CDN\n(static assets, DDoS)"]
    LB["Load Balancer\n(managed, HA pair)"]
    App1["App Server 1"]
    App2["App Server 2"]
    App3["App Server 3"]
    PgB["PgBouncer\n(connection pooler)"]
    Redis["Redis\n(sessions + cache)"]
    DBPrimary["PostgreSQL Primary\n(writes)"]
    DBReplica["PostgreSQL Replica\n(reads)"]
    S3["S3 + CDN\n(static files)"]

    Users --> CDN
    CDN --> LB
    LB --> App1 & App2 & App3
    App1 & App2 & App3 --> Redis
    App1 & App2 & App3 --> PgB
    PgB --> DBPrimary
    PgB --> DBReplica
    DBPrimary -- replication --> DBReplica
    App1 & App2 & App3 --> S3
```

---

## 8. Monitoring You Must Have Before Leaving This Tier

At 100K DAU, you are no longer flying solo. You need proactive monitoring, not reactive user complaints.

### The four metrics that matter most at this tier

**1. Database query performance**
```sql
-- Alert when any query exceeds 200ms average in pg_stat_statements
-- Check weekly, fix any query > 100ms avg that runs > 100 times/day
```

**2. Cache hit ratio**
```bash
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"
# Alert if hit ratio drops below 80%
```

**3. API p99 latency by endpoint**
```
Alert thresholds:
  p50 > 100ms → investigate
  p99 > 500ms → page on-call
```

**4. Error rate**
```
Alert: error rate > 1% for any endpoint
```

**Tools at this tier:**
- Application metrics: Datadog, New Relic, or Grafana + Prometheus (free)
- Error tracking: Sentry (has free tier)
- Log aggregation: CloudWatch Logs, Papertrail, or Logtail
- Uptime: Better Uptime or PagerDuty for on-call routing

---

## 9. What Not to Build Yet

| Thing | Why not yet |
|-------|-------------|
| Microservices | Still adds operational overhead without clear benefit |
| Kafka | A simple Redis queue or SQS handles background jobs fine |
| Elasticsearch | Add only when text search is measurably slow |
| Multi-region | Not justified until you have users globally AND downtime SLAs |
| Sharding | You have not maxed out a single primary — not close |
| Custom rate limiter service | Redis-based rate limiting is sufficient |

---

## Summary

- At 10K–100K DAU, the database is the first bottleneck — query efficiency before hardware
- Fix sequence: indexes → connection pooling → caching → read replica → vertical scale
- A single well-placed index can eliminate 99% of a slow query — do this before adding servers
- PgBouncer in transaction mode keeps connection count under control as server count grows
- Redis caching at 85%+ hit ratio reduces DB load by ~85% — the highest leverage change at this tier
- Read replica splits read and write load — watch for replication lag causing stale reads
- Add monitoring now: p99 latency per endpoint, DB query time, cache hit ratio, error rate

---

> Next: [Lesson 7.3 — 100K–1M DAU: Distributed Systems Begin](./lesson-7.3-100k-to-1m.md)