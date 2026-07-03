# Lesson 3.4 — Connection Pooling Deep Dive

> **Chapter 3 — The Data Layer**
> Previous: [Lesson 3.3 — Read Replicas](./lesson-3.3-read-replicas.md) | Next: [Lesson 3.5 — Database Sharding](./lesson-3.5-sharding.md)

---

## What this lesson covers

- Why database connections are expensive and limited
- What a connection pool is and how it works
- PgBouncer — the standard PostgreSQL connection pooler
- Session mode vs transaction mode vs statement mode
- The pool sizing formula
- What happens when the pool is exhausted

---

## 1. The Problem — Connections Are Expensive

Every connection to PostgreSQL is not just a socket — it is a full OS process on the database server, consuming:

- ~5–10MB of RAM per connection
- A file descriptor
- Background memory for query execution buffers

PostgreSQL can handle around **100–200 simultaneous connections** before memory pressure and context switching overhead degrades performance significantly. Pushing beyond 500 connections on a typical server makes things worse, not better.

Now consider a modern deployment:

```
50 app servers
× 20 threads per server (thread-per-request model)
= 1,000 simultaneous DB connections needed

PostgreSQL's safe limit: ~100–200 connections
Gap: 5–10× more connections than the DB can handle
```

Without connection pooling, you have two bad options:
- Let 1,000 connections hit the database → memory exhaustion, severe slowdown
- Limit each server to 2 connections → threads block waiting for a connection, requests time out

Connection pooling solves this by sitting between your app servers and the database, maintaining a small pool of real DB connections and multiplexing thousands of app connections onto them.

---

## 2. How a Connection Pool Works

```mermaid
graph LR
    A1["App Server 1\n(20 threads)"]
    A2["App Server 2\n(20 threads)"]
    A3["App Server 3\n(20 threads)"]
    Pool["PgBouncer\nConnection Pool\n(maintains 20 real DB connections)"]
    DB["PostgreSQL\n(20 connections max used)"]

    A1 -->|"up to 20 app connections"| Pool
    A2 -->|"up to 20 app connections"| Pool
    A3 -->|"up to 20 app connections"| Pool
    Pool -->|"20 real DB connections"| DB
```

**What the pool does:**
1. Maintains a fixed number of real connections to the database (the pool)
2. When an app thread needs a DB connection, it borrows one from the pool
3. When the thread is done with the query, it returns the connection to the pool
4. The connection is not closed — it stays open, ready for the next borrower
5. If all pool connections are busy, the app thread waits in a queue

**The benefit:** 1,000 app threads can share 20 real DB connections if queries are fast (which they should be — typically 1–50ms). At any given moment, only a small fraction of threads are actively querying the DB.

---

## 3. PgBouncer — The Standard PostgreSQL Pooler

PgBouncer is a lightweight, single-process connection pooler for PostgreSQL. It is the most widely used option and is simple to configure.

### Deployment topology

```
App Servers
    ↓  (connect to PgBouncer, not PostgreSQL directly)
PgBouncer (runs on each app server, or as a dedicated service)
    ↓  (maintains pool of real connections)
PostgreSQL Primary
```

PgBouncer can run as a sidecar on each app server (low latency, simpler) or as a centralized service (easier to manage pool globally).

### Basic configuration

```ini
# pgbouncer.ini

[databases]
myapp = host=postgres.internal port=5432 dbname=myapp

[pgbouncer]
listen_port = 6432
listen_addr = *

# Pool settings
pool_mode = transaction          # see section 4
max_client_conn = 1000           # max app connections to PgBouncer
default_pool_size = 25           # real DB connections per database per user
min_pool_size = 5                # keep at least 5 connections warm
reserve_pool_size = 5            # extra connections for emergencies
reserve_pool_timeout = 5         # wait 5s before using reserve pool

# Connection limits
server_idle_timeout = 600        # close idle server connections after 10min
client_idle_timeout = 0          # never close idle client connections
```

Your app connects to `localhost:6432` (PgBouncer) instead of `postgres.internal:5432` (PostgreSQL directly).

---

## 4. Pool Modes — The Most Important Configuration Decision

PgBouncer has three pooling modes. They differ in when a server connection is returned to the pool.

### Session Mode

```
App connects → gets a dedicated server connection → holds it for the entire session → disconnects → connection returned to pool
```

The server connection is held for as long as the app connection is open, even when no query is running.

**When to use:** When your app uses session-level features — temporary tables, advisory locks, `SET` commands, prepared statements that persist between queries. These features require a stable connection throughout the session.

**Problem:** Connection utilization is low. If an app thread holds a connection for 30 seconds but only queries for 50ms of that time, the connection is idle 99.8% of the time. Pool capacity is wasted.

---

### Transaction Mode (recommended for most apps)

```
App connects → starts a transaction → gets a server connection → transaction ends → connection returned to pool immediately
```

The server connection is held only for the duration of a single transaction (or single query if no explicit transaction).

**Benefit:** A connection is idle for microseconds between queries, not seconds. A pool of 25 connections can serve hundreds of concurrent app threads. This is the mode that gives connection pooling its dramatic capacity multiplication.

**Example:**
```
App thread lifecycle (200ms total):
  0ms:   Request starts
  10ms:  BEGIN TRANSACTION
  10ms:  → borrow connection from pool
  12ms:  SELECT * FROM users WHERE id = 42
  13ms:  UPDATE orders SET status = 'shipped' WHERE id = 99
  15ms:  COMMIT
  15ms:  → return connection to pool ← held for only 5ms
  200ms: Request ends (185ms of processing with no DB connection held)
```

**Limitation:** Session-level features do not work in transaction mode. Specifically:
- `SET` commands (they reset between transactions)
- Temporary tables (dropped when connection returns to pool)
- Advisory locks (`pg_advisory_lock` — they are session-scoped)
- Prepared statements (with protocol-level prepared statements, not SQL-level `PREPARE`)

Most applications do not use these features and can use transaction mode safely.

---

### Statement Mode

```
App gets a connection → runs a single statement → connection returned to pool
```

One statement per connection borrow. Multi-statement transactions are not supported.

**When to use:** Almost never for application code. Useful for simple read-only workloads where each request is a single SELECT.

---

### Mode Summary

| Mode | Connection held for | Session features | Best for |
|------|-------------------|-----------------|----------|
| Session | Entire app connection lifetime | ✅ All features work | Apps using temp tables, advisory locks |
| Transaction | Single transaction duration | ❌ Most session features break | Most web applications |
| Statement | Single statement | ❌ No transactions | Simple read-only workloads |

---

## 5. Pool Sizing — The Formula

Getting pool size right matters. Too small and app threads queue waiting for connections. Too large and you push PostgreSQL past its connection limit.

### The formula

```
pool_size = (core_count × 2) + number_of_spindle_disks

This is the PostgreSQL project's own recommendation.

For a modern server with 8 cores and SSDs (0 spindles):
  pool_size = (8 × 2) + 0 = 16

For a server with 4 cores and 2 HDDs:
  pool_size = (4 × 2) + 2 = 10
```

This seems low. It is intentional. PostgreSQL performs best with a modest number of connections because CPU context switching and memory pressure from many connections hurts more than parallelism helps.

### Accounting for multiple app servers and pools

```
Scenario:
  5 app servers
  Each runs a PgBouncer with pool_size = 20
  Total connections to PostgreSQL = 5 × 20 = 100

PostgreSQL max_connections = 120 (leave 20 for admin access)

This works ✅
```

If you use a centralized PgBouncer (one instance shared by all app servers):

```
1 PgBouncer
  pool_size = 20 (for the single pool to PostgreSQL)
  max_client_conn = 2000 (up to 2000 app threads can connect to PgBouncer)

Total connections to PostgreSQL = 20
```

This is even more efficient but the single PgBouncer is now a potential SPOF — run two with a load balancer in front.

---

## 6. What Happens When the Pool Is Exhausted

When all connections in the pool are busy and an app thread needs one, it enters a **wait queue**. PgBouncer holds the app thread waiting until a connection is freed.

```
pool_size = 20
Active queries: 20 (all connections busy)
New request arrives: waits in queue

PgBouncer behavior:
  - queue_timeout = 30 seconds (configurable)
  - If a connection frees up within 30s: query runs
  - If 30s passes and still no free connection: error returned to app
    "ERROR: no more connections allowed (max_client_conn)"
```

### What pool exhaustion looks like in monitoring

```
Symptoms:
  - API latency spikes suddenly (threads waiting for connections)
  - Error rate rises with "connection timeout" errors
  - Database CPU is LOW (DB is not the problem — threads aren't even reaching it)
  - PgBouncer "cl_waiting" metric is high (clients waiting for a pool connection)
```

### Common causes of pool exhaustion

| Cause | Description | Fix |
|-------|-------------|-----|
| Slow queries | Queries taking 10s each tie up connections | Fix the slow query (index, optimize) |
| Long transactions | Transaction held open while doing non-DB work | Never hold a transaction open during API calls, file I/O, etc. |
| Traffic spike | More concurrent requests than pool can handle | Increase pool size, or add read replicas to spread load |
| Connection leak | App code opens connections and never closes them | Audit code for missing `close()` or missing `with` blocks |
| N+1 queries | Each request makes 100 queries instead of 1 | Fix the N+1 (Lesson 3.2) |

---

## 7. Application-Level Connection Pools

In addition to PgBouncer (which is external to your app), most language runtimes have built-in connection pools:

```python
# Python (SQLAlchemy)
engine = create_engine(
    "postgresql://user:pass@localhost/myapp",
    pool_size=10,          # number of persistent connections
    max_overflow=20,       # extra connections above pool_size (temporary)
    pool_timeout=30,       # wait up to 30s for a connection
    pool_recycle=1800,     # recycle connections after 30min (prevents stale connections)
)
```

```javascript
// Node.js (pg library)
const pool = new Pool({
    max: 10,                    // pool size
    idleTimeoutMillis: 30000,   // close idle connections after 30s
    connectionTimeoutMillis: 2000, // fail fast if no connection available
});
```

**Using both PgBouncer and app-level pools:** This is common and correct. The app pool provides fast connection reuse within a single process. PgBouncer multiplexes many app pools onto a small number of real DB connections.

```
App process (pool of 10)
    ↓ (up to 10 connections)
PgBouncer (pool of 25 real connections)
    ↓ (up to 25 connections)
PostgreSQL
```

---

## Summary

- Each PostgreSQL connection is an OS process consuming 5–10MB of RAM; too many connections degrade performance
- A connection pool maintains a small set of real DB connections and shares them across many app threads
- PgBouncer is the standard PostgreSQL pooler — lightweight, battle-tested
- Transaction mode is right for most web apps: connection held only during the transaction
- Session mode is needed for temp tables, advisory locks, and other session-scoped features
- Pool size formula: `(core_count × 2) + spindle_disk_count` — typically 10–25 connections
- Pool exhaustion causes API latency spikes — monitor `cl_waiting` in PgBouncer and slow queries that hold connections

---

## ⚠️ Common Mistakes

- Connecting app servers directly to PostgreSQL without a pooler — connection count grows with server count and blows past PostgreSQL's limit
- Setting pool_size = 200 thinking more is better — PostgreSQL degrades under high connection counts; keep it small
- Using session mode with PgBouncer when transaction mode would work — wastes pool capacity
- Holding transactions open during external API calls or long computations — one slow API call blocks a DB connection for seconds
- Not setting `pool_recycle` — connections can go stale (TCP timeout, DB restart) without the app knowing, causing intermittent errors

---

> Next: [Lesson 3.5 — Database Sharding](./lesson-3.5-sharding.md)