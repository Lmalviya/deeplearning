# System Design at Scale — Complete Tutorial Index

> A practical, real-world guide to identifying bottlenecks, understanding tradeoffs,
> and making architecture decisions at every scale tier.
>
> **How to use this:** Read chapters in order if you are a beginner.
> Jump to a specific chapter if you already know the basics and want to go deep.
> Every lesson has theory + real-world examples + tradeoff decisions.

---

## Curriculum Map

```
Chapter 0 → Foundations (mindset before architecture)
Chapter 1 → Request journey (what happens before your code runs)
Chapter 2 → Compute layer (your app servers)
Chapter 3 → Data layer (databases — the most common bottleneck)
Chapter 4 → Caching (the performance multiplier)
Chapter 5 → Async & queues (decoupling for scale)
Chapter 6 → Delivery layer (CDN, blob storage, search)
Chapter 7 → Scale tiers (what breaks at 10K / 100K / 1M DAU)
Chapter 8 → Core tradeoffs (the decisions every architect must make)
Chapter 9 → Putting it together (real system walkthroughs)
```

---

## Chapter 0 — Foundations

> Before writing any architecture, you need the right mental models.
> This chapter is short but critically important.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 0.1 | How to think about scale | What DAU, RPS, throughput, and latency actually mean. How to estimate numbers. Back-of-envelope math. |
| 0.2 | The anatomy of a system | Every system has the same skeleton — client, network, compute, storage. Understanding this skeleton is how you spot bottlenecks systematically. |
| 0.3 | Single point of failure (SPOF) | What a SPOF is, why it is always the first thing to fix, and the pattern for eliminating it at every layer. |
| 0.4 | Stateless vs stateful design | The single most important design decision for horizontal scaling. Why stateless apps scale infinitely and stateful ones don't. |
| 0.5 | How to read a bottleneck | CPU-bound vs I/O-bound vs memory-bound vs network-bound. How to tell which one you have and what to do about each. |

---

## Chapter 1 — The Request Journey (Networking & Routing Layer)

> A request travels through 4–6 components before it reaches your code.
> Each one can be a bottleneck and each one has a job.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 1.1 | DNS — the ignored first step | How DNS resolution works, TTL tradeoffs, GeoDNS for global routing, DNS as a SPOF. |
| 1.2 | CDN — content delivery networks | How CDNs work, edge caching, cache hit ratio, cache invalidation, when CDN helps and when it doesn't. |
| 1.3 | Load balancer — distributing traffic | L4 vs L7 load balancing, algorithms (round robin, least connections, weighted), sticky sessions and why they are dangerous, health checks. |
| 1.4 | Reverse proxy — your front door | What a reverse proxy does that a load balancer doesn't, SSL termination, request buffering, Nginx vs HAProxy. |
| 1.5 | API gateway — the microservices entry point | Auth, rate limiting, routing, protocol translation. When to add an API gateway and when it becomes a bottleneck itself. |
| 1.6 | Rate limiting — protecting your system | Fixed window vs sliding window vs token bucket vs leaky bucket. Distributed rate limiting with Redis. Handling retry storms. |

---

## Chapter 2 — The Compute Layer (App Servers & Processing)

> Your application code runs here. This chapter covers how to scale it, what
> makes it slow, and how to never let your servers be the bottleneck.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 2.1 | Web server vs app server | The difference, why it matters, how Nginx + Gunicorn or Nginx + Node works, and why you should never serve static files from your app server. |
| 2.2 | Horizontal vs vertical scaling | The mechanics of both, cost comparison, when to scale up vs scale out, and why horizontal scaling requires stateless design. |
| 2.3 | Concurrency models | Thread-per-request (Java/Python) vs event loop (Node.js) vs coroutines (Go/async Python). Why this matters for I/O-heavy vs CPU-heavy workloads. |
| 2.4 | Connection pooling | Why opening a new DB connection per request kills performance, how connection pools work, pool sizing math, PgBouncer. |
| 2.5 | Background jobs & workers | Moving work off the request path, job queues, worker processes, scheduled jobs (cron), and how to avoid blocking your API on heavy computation. |
| 2.6 | Auto-scaling | Reactive vs predictive scaling, scale-in/scale-out triggers, warm-up time problem, stateful servers and why they break auto-scaling. |

---

## Chapter 3 — The Data Layer (Databases — The Most Common Bottleneck)

> The database is where 80% of performance problems live. This is the
> longest chapter because it deserves the most attention.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 3.1 | How relational databases work internally | B-tree indexes, buffer pool, WAL (write-ahead log), MVCC. Understanding internals helps you understand why queries are slow. |
| 3.2 | Indexing — the single biggest performance lever | How indexes work, when they help, when they hurt, composite indexes, covering indexes, the N+1 query problem. |
| 3.3 | Read replicas — scaling reads | How replication works (sync vs async), replication lag and when it causes bugs, routing reads to replicas, replica failure handling. |
| 3.4 | Connection pooling deep dive | PgBouncer vs ProxySQL, transaction mode vs session mode, pool sizing formula, what happens when the pool is exhausted. |
| 3.5 | Database sharding | What sharding is, sharding strategies (hash, range, directory), cross-shard queries (why they are painful), resharding, when to shard and when not to. |
| 3.6 | NoSQL — types and tradeoffs | Document (MongoDB), key-value (DynamoDB/Redis), wide-column (Cassandra), graph (Neo4j). When to use each and when to use both SQL and NoSQL together. |
| 3.7 | CAP theorem in practice | Consistency vs Availability vs Partition tolerance — not just theory but real examples of which DBs choose which and what it means for your data. |
| 3.8 | Schema design for scale | Normalization vs denormalization, designing for query patterns, schema migration at scale without downtime. |
| 3.9 | Database anti-patterns | SELECT *, no pagination, no indexes on foreign keys, long-running transactions, using the DB as a queue — why these kill performance at scale. |

---

## Chapter 4 — The Caching Layer (The Performance Multiplier)

> A well-designed cache can reduce DB load by 90%. A poorly designed cache
> causes bugs that are nearly impossible to debug. This chapter covers both sides.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 4.1 | How caching works — mental model | Cache hit vs miss, hit ratio math, where to place a cache (client, CDN, app, DB), Redis vs Memcached. |
| 4.2 | Cache writing strategies | Cache-aside (lazy loading), write-through, write-back (write-behind), read-through. When to use each and the tradeoffs of each. |
| 4.3 | Cache invalidation — the hard problem | TTL-based expiration, event-driven invalidation, cache tags, why invalidation is hard in distributed systems. |
| 4.4 | Cache stampede & thundering herd | What happens when a popular cache key expires and 10,000 requests hit the DB simultaneously. Mutex locks, probabilistic early expiration, background refresh. |
| 4.5 | Eviction policies | LRU, LFU, FIFO, Random — what each one does, which workload each fits, and how to pick the right one. |
| 4.6 | Redis deep dive | Data structures (string, hash, list, set, sorted set, stream), Redis Cluster, Redis Sentinel, Pub/Sub, Lua scripts, Redis as a rate limiter. |
| 4.7 | Distributed caching problems | Cache consistency across multiple app servers, cache warm-up after deploy, regional cache coherence in multi-region setups. |

---

## Chapter 5 — Async Processing & Message Queues

> Synchronous = user waits. Async = user gets an answer immediately, work happens later.
> This chapter covers when and how to go async, and all the ways it can go wrong.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 5.1 | Why async — the mental model | The cost of synchronous blocking, which operations should always be async, how async changes the user experience and system design. |
| 5.2 | Message queue fundamentals | Producer, consumer, broker, queue vs topic, point-to-point vs pub/sub, acknowledgement and why it matters. |
| 5.3 | Kafka deep dive | Partitions, offsets, consumer groups, retention, why Kafka is not a queue (it's a log), when to use Kafka vs simpler queues. |
| 5.4 | RabbitMQ & SQS | When to use RabbitMQ vs Kafka vs SQS, exchanges and routing in RabbitMQ, SQS visibility timeout, dead letter queues. |
| 5.5 | Idempotency — the hardest queue problem | At-least-once vs at-most-once vs exactly-once delivery. Why you must design consumers to be idempotent, and how to do it with an idempotency key. |
| 5.6 | Consumer lag & backpressure | What consumer lag is, how to detect it, horizontal scaling of consumers, backpressure mechanisms to protect downstream services. |
| 5.7 | Event-driven architecture | Events vs commands vs queries, event sourcing, CQRS (command query responsibility segregation), when event-driven design is worth the complexity. |

---

## Chapter 6 — The Delivery Layer (Storage, Search & Edge)

> Files, search, and edge computing — three components that are often added
> too late and cause painful migrations when they are.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 6.1 | Object / blob storage | How S3-style storage works, presigned URLs for direct client uploads, multipart upload, lifecycle policies, cost optimization. |
| 6.2 | CDN deep dive | Edge caching vs origin, cache-control headers, cache busting strategies, CDN for dynamic content (edge computing), multi-CDN for resilience. |
| 6.3 | Search engines | Why SQL LIKE is not search, how inverted indexes work, Elasticsearch architecture, index design, keeping DB and search index in sync. |
| 6.4 | Data pipelines & analytics | OLTP vs OLAP, why you should not run analytics on your production DB, data warehouse, ETL vs ELT, columnar storage. |

---

## Chapter 7 — Scale Tiers (What Breaks at Each Level)

> Theory meets reality. This chapter walks through exactly what fails as you grow
> from 1K to 10M DAU and what to do about it at each stage.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 7.1 | 1K–10K DAU — the startup phase | Single server risks, the minimum viable production setup, why you should add a load balancer even at this stage. |
| 7.2 | 10K–100K DAU — the first real scale | The DB becomes the bottleneck, adding read replicas and caching, making the app stateless, the monitoring you must have. |
| 7.3 | 100K–1M DAU — distributed systems begin | Write scaling, async queues for heavy work, API gateway, search, blob storage migration, service decomposition starts. |
| 7.4 | 1M–10M DAU — everything is distributed | DB sharding, multi-region, cache at every layer, rate limiting at scale, the operational maturity required. |
| 7.5 | 10M+ DAU — hyper scale | How companies like Twitter, Netflix, and Uber solved these problems. Custom infrastructure, edge computing, global consistency challenges. |

---

## Chapter 8 — Core Tradeoffs (Every Decision Has a Cost)

> In system design, there is no "best" — only "best for this situation."
> This chapter is about making conscious tradeoff decisions, not following rules.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 8.1 | Consistency vs Availability (CAP) | The real meaning of CAP, strong vs eventual vs causal consistency, which systems choose which and why. |
| 8.2 | Latency vs Throughput | Why optimizing one hurts the other, batching as a throughput technique, real-time streaming as a latency technique. |
| 8.3 | SQL vs NoSQL | Not a religious debate — a pattern-matching exercise. How to choose based on access patterns, consistency needs, and team familiarity. |
| 8.4 | Sync vs Async | When you must be synchronous and when async is always better. The UX tradeoffs of async operations. |
| 8.5 | Horizontal vs Vertical scaling | Cost, complexity, limits, and when each is the right tool. Why vertical scaling is underrated at medium scale. |
| 8.6 | Monolith vs Microservices | The real cost of microservices (operational, not code). When to split and when to stay monolithic. The strangler fig pattern. |
| 8.7 | Strong consistency vs Performance | The cost of distributed transactions, two-phase commit, saga pattern, how to design around the need for distributed consistency. |

---

## Chapter 9 — Putting It Together (Real System Walkthroughs)

> Apply everything from chapters 0–8 to design real systems from scratch.
> Each walkthrough covers requirements → estimation → architecture → bottlenecks → scale.

| Lesson | Title | What you will learn |
|--------|-------|-------------------|
| 9.1 | Design a URL shortener | Simple but covers hashing, DB choice, caching, and redirection at scale. Good first walkthrough. |
| 9.2 | Design a social media feed | Fan-out on write vs fan-out on read, the celebrity problem, caching feeds, infinite scroll pagination. |
| 9.3 | Design a chat system (WhatsApp-style) | WebSockets, message delivery guarantees, presence (online/offline), group messaging, message storage. |
| 9.4 | Design a notification system | Push vs pull, multi-channel delivery (email, SMS, push), rate limiting, deduplication, delivery tracking. |
| 9.5 | Design a ride-sharing backend (Uber-style) | Real-time location, geospatial indexing, matching algorithm, surge pricing, trip state machine. |
| 9.6 | Design a video streaming platform (YouTube-style) | Upload pipeline, video transcoding, adaptive bitrate, CDN strategy, recommendation system at scale. |

---

## Reading Order Recommendations

**If you are preparing for a system design interview (1–2 weeks):**
```
Chapter 0 (all) → Chapter 3 (3.1–3.5) → Chapter 4 (all) → Chapter 7 (all) → Chapter 8 (all) → Chapter 9 (9.1, 9.2, 9.3)
```

**If you are a developer who wants to understand production systems:**
```
Chapter 0 → Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5 → Chapter 7
```

**If you are debugging a slow system right now:**
```
Chapter 0.5 → Chapter 3.2 → Chapter 4.1 → Chapter 3.3 → Chapter 7 (find your tier)
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ⚡ | Performance-critical concept |
| ⚠️ | Common mistake / anti-pattern |
| 🔀 | Tradeoff decision point |
| 🏭 | Real-world example from a known company |
| 🧪 | Concept you can test locally |

---

*Total: 9 chapters · 52 lessons*
*Start with Chapter 0, Lesson 0.1 →*