# Lesson 8.5 — Horizontal vs Vertical Scaling

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 8.4 — Sync vs Async](./lesson-8.3-sql-vs-nosql.md) | Next: [Lesson 8.6 — Monolith vs Microservices](./lesson-8.6-monolith-vs-microservices.md)

---

## What this lesson covers

- The mechanics of both scaling strategies
- Cost comparison — when vertical is actually cheaper
- Why vertical scaling is underrated at medium scale
- The prerequisite for horizontal scaling (stateless design)
- Which components scale each way and why

---

## 1. Definitions

**Vertical scaling (scale up):** Replace your server with a bigger one. More CPU cores, more RAM, faster SSD. Same machine, more powerful.

**Horizontal scaling (scale out):** Add more servers. Keep the same size, but run more of them. Distribute load across the fleet.

---

## 2. The Cost Reality — Vertical is Often Cheaper

Teams default to horizontal scaling because "infinite horizontal scaling" sounds appealing. But at medium scale, vertical scaling is frequently cheaper and simpler.

```
Scenario: Your app server is at 80% CPU under load
Need: 2× more capacity

Option A — Vertical scale:
  Current: 4 vCPU, 16GB RAM → $100/month
  Upgraded: 8 vCPU, 32GB RAM → $180/month
  Cost increase: $80/month
  Complexity increase: 0 (same number of servers)
  Implementation time: 5 minutes (change instance type)

Option B — Horizontal scale:
  Current: 1 × 4 vCPU → $100/month
  Scaled: 2 × 4 vCPU → $200/month
  Cost increase: $100/month
  Complexity increase: must add load balancer ($20/month), must make app stateless
  Implementation time: 1–2 days (load balancer config, stateless refactor)
```

For 2× capacity at medium scale, vertical scaling is cheaper AND simpler. Horizontal scaling has fixed overhead (load balancer, stateless design, deployment automation) that only pays off at higher scale.

---

## 3. When Vertical Scaling Has a Ceiling

Vertical scaling has a hard limit — the largest available instance type. On AWS, the largest EC2 instance has 448 vCPUs and 24TB RAM (`u-24tb1.metal`). Most workloads never need this, but databases can.

```
PostgreSQL primary scaling ceiling:
  Largest feasible instance: 128 vCPU, 2TB RAM → ~$15,000/month
  Write throughput at max vertical: ~50,000 writes/second

  If you need > 50,000 writes/second → vertical ceiling reached → must shard (horizontal)

Web servers:
  Vertical ceiling rarely reached — horizontal is preferred for redundancy anyway
  (two small servers > one big server for fault tolerance)
```

The ceiling varies by component. Web servers hit practical redundancy limits before hitting performance limits. Databases may hit performance limits before redundancy limits.

---

## 4. Component-by-Component Scaling Strategy

| Component | Primary scaling strategy | Why |
|-----------|------------------------|-----|
| Web / app servers | Horizontal | Stateless by design — trivial to add more |
| PostgreSQL (reads) | Horizontal (read replicas) | Adding replicas distributes read load |
| PostgreSQL (writes) | Vertical first, then horizontal (sharding) | Sharding adds enormous complexity |
| Redis | Vertical first (more RAM), then horizontal (Redis Cluster) | Single-node Redis is simpler; cluster for very large datasets |
| Kafka | Horizontal (more brokers, more partitions) | Designed for horizontal from the ground up |
| Message queue workers | Horizontal (more consumers) | Stateless workers trivially add more |
| Elasticsearch | Horizontal (more nodes) | Designed for horizontal distribution |
| Object storage (S3) | Neither — it scales automatically | Managed service |

---

## 5. The Prerequisite for Horizontal Scaling

You cannot horizontally scale a stateful component. Before adding a second app server, the app must be stateless (Lesson 0.4).

```
Checklist before horizontal scaling app servers:
  ✅ Sessions stored in Redis (not in server memory)
  ✅ Files stored in S3 (not on server disk)
  ✅ No cron jobs in the app process (move to dedicated scheduler)
  ✅ No in-memory rate limit counters (move to Redis)
  ✅ Configuration via environment variables (not server-local files)

Checklist before horizontal scaling DB (sharding):
  ✅ Application routes queries by shard key
  ✅ No cross-shard JOINs in critical paths
  ✅ ID generation works across shards (UUID v7 or Snowflake ID)
  ✅ Schema migrations run on all shards
```

Attempting horizontal scaling without these prerequisites creates subtle, hard-to-debug bugs.

---

## 6. The Right Decision Framework

```
Is the component stateless?
  YES → Horizontal is easy. Scale out.
  NO  → Stateful: use vertical first. Horizontal requires significant work.

Is the component's ceiling approachable?
  Ceiling far away → vertical scale, simpler, cheaper at this scale.
  Ceiling near → horizontal scale is necessary despite complexity.

Do you need redundancy (SPOF elimination)?
  YES → Horizontal even if one instance could handle the load.
        (2 × half-capacity = same throughput but redundant)
  NO  → Vertical is fine.
```

### Practical guidance by scale tier

| Tier | App Servers | Database (reads) | Database (writes) |
|------|------------|-----------------|-----------------|
| 1K–10K DAU | 2 horizontal (for redundancy) | None needed | Vertical (small) |
| 10K–100K DAU | 3–5 horizontal | 1 replica | Vertical (medium) |
| 100K–1M DAU | 5–15 horizontal | 2–3 replicas | Vertical (large) |
| 1M–10M DAU | 15–50 horizontal | 3–5 replicas | Vertical (largest) or functional sharding |
| 10M+ DAU | 50–500 horizontal | Many replicas | Horizontal sharding |

---

## Summary

- Vertical scaling: bigger machine. Simple, no code changes, limited by the largest available instance.
- Horizontal scaling: more machines. Requires stateless design, load balancing, deployment automation.
- At medium scale, vertical is often cheaper and simpler — the overhead of horizontal (load balancer, stateless refactor) must be justified.
- Vertical has a ceiling — web servers rarely reach it, databases may.
- Scale stateless components (app servers, workers) horizontally from the start (redundancy). Scale stateful components (DB) vertically first, horizontally only when necessary.
- The prerequisite for horizontal app server scaling: sessions in Redis, files in S3, no local state anywhere.

---

> Next: [Lesson 8.6 — Monolith vs Microservices](./lesson-8.6-monolith-vs-microservices.md)

---

# Lesson 8.6 — Monolith vs Microservices

> **Chapter 8 — Core Tradeoffs**
> Previous: [Lesson 8.5 — Horizontal vs Vertical Scaling](./lesson-8.5-horizontal-vs-vertical.md) | Next: [Lesson 8.7 — Strong Consistency vs Performance](./lesson-8.7-distributed-transactions.md)

---

## What this lesson covers

- What a monolith actually is (not what you think it is)
- The real cost of microservices — operational, not code
- When microservices are the right choice
- The strangler fig pattern for decomposition
- The modular monolith — the underused middle path

---

## 1. Definitions

**Monolith:** All application code runs in one deployable unit. One codebase, one deploy, one process (or a few processes for redundancy). The code may be well-organized internally (modules, packages) — that is a modular monolith.

**Microservices:** Application is split into many independent services. Each service has its own codebase, deployment pipeline, and database. Services communicate over the network (REST, gRPC, message queue).

The key distinction is the **deployment unit and the network boundary**, not internal code organization.

---

## 2. The Real Cost of Microservices

Teams often adopt microservices expecting simpler code. They get more complex operations instead.

| Problem in monolith | "Solution" via microservices | Hidden cost introduced |
|--------------------|----------------------------|----------------------|
| Code is large and hard to navigate | Split into small services | Distributed system: network calls, latency, partial failures |
| One team cannot deploy without coordinating | Independent deployments | API versioning, backwards compatibility requirements |
| One part of the app uses too much CPU | Independent scaling | Service discovery, load balancing, inter-service auth |
| Hard to test one component in isolation | Each service independently testable | Integration testing across services is much harder |
| Monolithic database is overloaded | Each service owns its DB | No cross-service JOINs, distributed transactions needed |

**The honest summary:** microservices solve organizational problems (team independence, independent deployment) at the cost of operational complexity (distributed system challenges). They do not solve code complexity — they move it.

---

## 3. What a Well-Structured Monolith Looks Like

A monolith is not "one giant file." A well-organized monolith has clear internal structure:

```
myapp/
├── modules/
│   ├── users/
│   │   ├── repository.py     ← DB access layer
│   │   ├── service.py        ← business logic
│   │   └── api.py            ← HTTP handlers
│   ├── orders/
│   │   ├── repository.py
│   │   ├── service.py
│   │   └── api.py
│   └── payments/
│       ├── repository.py
│       ├── service.py
│       └── api.py
├── shared/
│   ├── database.py
│   └── auth.py
└── main.py
```

This monolith has clean boundaries between modules. Each module owns its logic. If a module needs to become a service later, the boundary is already clear — extraction is straightforward.

---

## 4. When Microservices Are the Right Choice

Microservices make sense when these conditions are met:

**Condition 1 — You have multiple autonomous teams.**
The primary benefit of microservices is that Team A can deploy without coordinating with Team B. If you have one team, this benefit does not exist.

**Condition 2 — Services have genuinely different scaling requirements.**
If your image processing service needs 20 GPU instances and your API needs 10 small instances, running them together wastes resources or limits either service.

**Condition 3 — You need fault isolation between components.**
A crashing recommendation engine should not take down checkout. Service boundaries enforce this isolation.

**Condition 4 — The team has the operational maturity to run distributed systems.**
Microservices require: service discovery, distributed tracing, inter-service auth, API versioning, distributed transaction patterns. These require dedicated platform engineering.

**If fewer than 3 of these are true: do not use microservices.**

---

## 5. The Modular Monolith — The Underused Middle Path

A modular monolith has clear internal module boundaries (enforced by code conventions or tooling), deployed as a single unit. It captures most of the code organization benefits of microservices without the operational cost.

```
Benefits of modular monolith:
  ✅ Clear ownership: each module is owned by one team
  ✅ Independent development: teams work in their module without touching others
  ✅ Simple deployment: one deployment pipeline
  ✅ No network calls: module-to-module calls are in-process (fast, reliable)
  ✅ Simple testing: no need to mock other services
  ✅ Easy extraction: when a module genuinely needs to be a service, the boundary is clean
```

Many companies (Stack Overflow, Shopify, Basecamp, GitHub) run successful products at high scale on well-architected monoliths or modular monoliths.

---

## 6. When to Extract a Service — The Correct Signal

Do not extract a service because "that module is getting complex." Extract when:

```
✅ The module has genuinely different infrastructure needs
   (image processing needs GPU; auth needs strict security isolation)

✅ The module needs to be deployed by a different team at a different cadence
   (notification service deploys hourly; core API deploys weekly)

✅ The module is a natural product boundary
   (Stripe is a payment "microservice" your app calls)

✅ The module's failure should be isolated from core product failure
   (recommendation engine failure should not break checkout)

✅ The module is a clear bottleneck that needs independent scaling
   (search service needs 20 Elasticsearch nodes; rest of app needs 5 servers)
```

---

## 7. The Strangler Fig Pattern

When you have an existing monolith and need to extract a service, do it incrementally:

```
Step 1: Identify the extraction target (one service at a time)
        Never try to break a monolith into 20 services simultaneously.

Step 2: Build the new service alongside the monolith
        New service implements the same functionality.

Step 3: Route a small percentage of traffic to the new service
        (canary deployment: 1% → 5% → 25% → 100%)

Step 4: Monitor: same behavior, correct results, acceptable latency?
        If yes → continue migration.
        If no  → route back to monolith, fix, retry.

Step 5: Remove the code from the monolith once 100% migrated.
        The monolith is "strangled" — extracted service by service.
```

This approach keeps risk low. The monolith is always the fallback.

---

## Summary

- Monolith: one deployable unit. Simple ops, harder team independence at large org size.
- Microservices: independent deployable services. Team independence, but distributed system complexity.
- The real cost of microservices is operational (distributed tracing, API versioning, inter-service auth) not code.
- Adopt microservices when: multiple autonomous teams, different scaling requirements per service, operational maturity to run distributed systems.
- Modular monolith: clear internal boundaries, single deployment. Gets most code benefits without operational cost. Underused and underrated.
- Extract services one at a time using the strangler fig pattern — never rewrite the whole monolith at once.

---

> Next: [Lesson 8.7 — Strong Consistency vs Performance](./lesson-8.7-distributed-transactions.md)