# Lesson 0.1 — How to Think About Scale

> **Chapter 0 — Foundations**
> Previous: [Index](../INDEX.md) | Next: [Lesson 0.2 — Anatomy of a System](./lesson-0.2-anatomy-of-a-system.md)

---

## What this lesson covers

- What DAU, RPS, throughput, and latency actually mean
- How to convert DAU into requests per second (the number that actually matters)
- Back-of-envelope estimation — the skill interviewers test and production engineers use daily
- Why "our system is slow" is not a problem statement — and how to make it one

---

## 1. The Numbers That Actually Matter

When someone says "our system needs to handle 100K users", that sentence is almost useless for engineering. You need to convert it into concrete numbers before you can design anything.

The four numbers you always need:

| Metric | What it means | Why it matters |
|--------|--------------|----------------|
| **DAU** (Daily Active Users) | Unique users who use the product on a given day | Starting point for all estimation |
| **RPS** (Requests Per Second) | How many HTTP requests hit your servers per second | Determines server count and load balancer config |
| **Throughput** | How much data moves through the system per second (MB/s, GB/s) | Determines network bandwidth and storage I/O |
| **Latency** | How long a single request takes end-to-end (ms) | Determines user experience and SLA requirements |

These are related but different. A system can have high throughput and high latency (a batch job that processes 10GB/hour but each item takes 30 seconds). Another system can have low latency and low throughput (a simple API that responds in 5ms but only serves 10 users).

---

## 2. Converting DAU → RPS (The Most Important Skill)

This is the core estimation technique. Here is the formula and the reasoning behind it.

```
RPS = (DAU × requests_per_user_per_day) / seconds_in_day
```

Seconds in a day = 86,400. You will use this number constantly. Memorize it.

But users are not evenly distributed across 24 hours. Traffic peaks in the evening and valleys at night. A common rule of thumb:

```
Peak RPS = Average RPS × 2 to 3
```

This peak multiplier is what you actually need to design for.

### Worked Example — A social media app with 1M DAU

Assume each user makes about 100 requests per day (browsing feed, posting, loading images, notifications).

```
Average RPS = (1,000,000 × 100) / 86,400
           = 100,000,000 / 86,400
           ≈ 1,157 RPS

Peak RPS    = 1,157 × 3
           ≈ 3,500 RPS
```

So "1M DAU" actually means your system needs to handle roughly 3,500 requests per second at peak. This is the number you take into your architecture decisions.

### Reference table — DAU to Peak RPS

| DAU | Requests/user/day | Average RPS | Peak RPS (3×) |
|-----|------------------|-------------|----------------|
| 10K | 50 | ~6 | ~18 |
| 100K | 50 | ~58 | ~174 |
| 1M | 100 | ~1,157 | ~3,500 |
| 10M | 100 | ~11,574 | ~35,000 |
| 100M | 100 | ~115,740 | ~350,000 |

This table shows why "100M DAU" is a genuinely hard engineering problem and why "10K DAU" is mostly a reliability problem, not a scale problem.

---

## 3. Storage Estimation

After RPS, the next thing to estimate is how much data you will store and how fast it will grow.

```
Storage per day = writes_per_second × data_size_per_write × 86,400
```

### Worked Example — A messaging app with 1M DAU

Assume each user sends 20 messages per day. Each message is 200 bytes of text on average.

```
Writes per second = (1,000,000 × 20) / 86,400 ≈ 231 writes/sec

Storage per day   = 231 × 200 bytes × 86,400
                  = 231 × 200 × 86,400
                  ≈ 3.99 GB/day
                  ≈ 4 GB/day

Storage per year  = 4 GB × 365 ≈ 1.46 TB/year
```

Now add media (images, videos) and that number grows by 100× or more. This is why WhatsApp and similar apps need petabyte-scale object storage.

---

## 4. Latency — What "Fast" Actually Means

Latency is the time from when a request is sent to when the response arrives. It has components:

```
Total latency = network latency + server processing time + DB query time + ...
```

You need reference numbers to reason about latency. These are rough but widely used:

| Operation | Typical latency |
|-----------|----------------|
| L1 cache read | ~0.5 ns |
| L2 cache read | ~7 ns |
| RAM read | ~100 ns |
| SSD read | ~100 µs (0.1 ms) |
| HDD read | ~10 ms |
| Network within same data center | ~0.5 ms |
| Network cross-region (US east → US west) | ~40 ms |
| Network cross-continent (US → Europe) | ~80 ms |
| Redis read (network included) | ~1 ms |
| PostgreSQL query (simple, indexed) | ~2–5 ms |
| PostgreSQL query (complex, unindexed) | ~100ms to seconds |

The practical takeaway: a RAM read is 200,000× faster than a network round-trip within a data center. This is why caching (keeping data in RAM) is such a powerful optimization.

---

## 5. Throughput — Data Movement at Scale

Throughput is how much work the system does per unit of time. Think of it as the width of a pipe, while latency is how fast water flows through it.

Common throughput reference points:

| Component | Typical throughput |
|-----------|--------------------|
| A single CPU core (simple computation) | ~500M operations/sec |
| A gigabit network link | ~125 MB/s |
| SSD sequential read | ~500 MB/s to 3 GB/s |
| HDD sequential read | ~100–200 MB/s |
| Redis | ~100K–1M operations/sec |
| PostgreSQL (writes) | ~1K–10K writes/sec (depends heavily on config) |
| Kafka | ~100K–1M messages/sec per broker |

---

## 6. Back-of-Envelope Estimation — The Method

In interviews and in real engineering discussions, you will be asked to estimate things quickly. Here is the method:

**Step 1 — Clarify what you are estimating.** Write it down explicitly. "I am estimating peak RPS for the read API."

**Step 2 — Identify your starting number.** Usually DAU or total users.

**Step 3 — Make your assumptions explicit and reasonable.** "I will assume each user makes 50 requests per day."

**Step 4 — Calculate step by step.** Do not skip steps. Round aggressively — precision is not the goal, order of magnitude is.

**Step 5 — Sanity check.** Does the number feel right? If you estimated that Gmail handles 1 request per second, something is wrong.

### The numbers to memorize

```
Seconds in a day   = 86,400    ≈ 100K (round up for easy math)
Seconds in a month = 2,592,000 ≈ 2.5M
Seconds in a year  = 31,536,000 ≈ 30M

KB = 10^3 bytes
MB = 10^6 bytes
GB = 10^9 bytes
TB = 10^12 bytes
```

---

## 7. What "Scale" Actually Means

Scale is not just about user count. A system "at scale" has these characteristics:

- **Failure is normal.** At 1,000 servers, one will fail today. Your system must handle it without the user noticing.
- **The math changes.** A bug that affects 0.01% of requests is invisible at 100 users. At 10M users, it affects 1,000 people every day.
- **Coordination costs money.** Two servers must agree on shared state (sessions, cache, locks). That agreement takes time and can fail.
- **Every component is a potential bottleneck.** Scale exposes the weakest link in your chain.

---

## Summary

- Convert DAU → RPS using `(DAU × requests/user/day) / 86,400`, then multiply by 2–3 for peak
- Storage grows as `writes/sec × size/write × 86,400` per day
- Latency components add up — network round trips dominate at scale
- Throughput is the width of your pipe; latency is the speed through it
- In estimation, order of magnitude matters more than precision

---

## ⚠️ Common Mistakes

- Designing for average load instead of peak load — your system will fall over during peak
- Forgetting that 86,400 is "only" ~100K, so 1M DAU at 100 requests each = 1B requests per day = ~11,500 RPS average
- Ignoring write throughput when estimating DB capacity — writes are much more expensive than reads

---

## 🔀 Key Decision This Lesson Informs

Once you know your peak RPS, you can answer:
- How many app servers do I need? (RPS / requests a single server can handle)
- Does my database need read replicas? (if DB reads/sec exceed single server capacity)
- Do I need a CDN? (if static asset bandwidth is significant)

---

> Next: [Lesson 0.2 — The Anatomy of a System](./lesson-0.2-anatomy-of-a-system.md)