# Lesson 7.5 — 10M+ DAU: Hyper Scale

> **Chapter 7 — Scale Tiers**
> Previous: [Lesson 7.4 — 1M–10M DAU](./lesson-7.4-1m-to-10m.md) | Next: [Chapter 8 — Core Tradeoffs](../chapter-8/lesson-8.1-consistency-vs-availability.md)

---

## What this lesson covers

- What engineering looks like at 100M+ DAU (Twitter, Netflix, Uber scale)
- Database horizontal sharding in practice
- Custom infrastructure decisions companies make at this scale
- The global consistency problem — CAP in practice at extreme scale
- How real companies solved specific 10M+ scale problems
- The human and organizational challenges that dominate at hyper scale

---

## 1. The Numbers at Hyper Scale

```
100M DAU × 100 requests/user/day = 10B requests/day
10B / 86,400 ≈ 115,740 RPS average
Peak RPS ≈ 350,000 RPS

Twitter at peak: ~400,000 tweets/day read 50B times (fan-out problem)
Netflix: 250M subscribers, 1B+ hours of video streamed/day
Uber: 5M trips/day, real-time location updates every 4 seconds per driver
WhatsApp: 100B messages/day, 2B users
```

At this scale, the problems are fundamentally different from what lower tiers face. You are not tuning queries or choosing between Redis and Memcached — you are building custom infrastructure.

---

## 2. Database Horizontal Sharding in Practice

At 100M DAU, no single database server can handle writes. Horizontal sharding is unavoidable.

### How Instagram sharded PostgreSQL

Instagram (before the Facebook acquisition) ran on PostgreSQL and sharded by `user_id`:

```
Sharding scheme:
  2,000 logical shards (not physical — logical shards can be moved)
  Each logical shard maps to a physical shard (many-to-one)

  user_id = 42
  logical_shard = hash(42) % 2000 = 847
  physical_shard = shard_map[847]  ← looked up in a central mapping table

Benefits of logical sharding:
  Adding physical capacity: move logical shards between physical nodes
  No rehashing needed — shard_map is updated, not the data
  2000 logical shards is large enough to distribute evenly across any reasonable number of physical nodes
```

### How Notion sharded PostgreSQL (2021)

Notion ran a single PostgreSQL database until they had millions of users. Their sharding approach:

```
Challenge: could not take downtime, had 20TB of data, live users

Approach:
  1. Added a "workspace_id" column to every table (their shard key)
  2. Built a routing layer that mapped workspace_id → shard
  3. Migrated one workspace at a time to the new sharded setup
  4. The routing layer made the migration invisible to users

Key insight: they did not shard by user_id but by workspace (tenant) —
each workspace's data lives entirely on one shard, so no cross-shard queries
are needed for any workspace-scoped operation.
```

### The cross-shard problem in practice

```
Before sharding:
  SELECT posts.*, users.name
  FROM posts
  JOIN users ON posts.user_id = users.id
  WHERE posts.created_at > NOW() - INTERVAL '1 day'
  → trivial join, one query

After user_id-based sharding:
  posts may be on Shard A (user 42's posts)
  users are on Shard B (user 42's profile)
  → Join is impossible at the database level

Solution: denormalize user name into the posts table
  posts table: {post_id, user_id, user_name, body, created_at}
  → No join needed, but user_name must be updated in all posts when user changes name
```

This is why sharding forces denormalization — and why denormalization decisions made before sharding determine how painful the migration is.

---

## 3. The Twitter Fan-Out Problem

Twitter's core challenge: when Katy Perry (80M followers) tweets, 80M users' feeds need to show that tweet.

### Approach 1 — Fan-out on write (push model)

When a user tweets, immediately write to all followers' feed caches.

```
Katy Perry tweets:
  For each of 80M followers:
    redis.lpush(f"feed:{follower_id}", tweet_id)

Problem: 80M Redis writes in < 1 second
         Tweet delivery delay for late followers
         Storage: every follower has a copy in their Redis feed
```

This works for most users (with small follower counts), but breaks for celebrities.

### Approach 2 — Fan-out on read (pull model)

When a user loads their feed, pull tweets from everyone they follow.

```
User A opens their feed (follows 500 people):
  SELECT tweets FROM following WHERE user_id IN (500 ids)
  ORDER BY created_at DESC LIMIT 50
  → 500 DB queries (one per followed user)
  → Or one big IN query on a massive tweets table
  → Slow for users following many accounts
```

### Twitter's actual solution — hybrid approach

```
Normal users (< 1M followers): fan-out on write
  → Tweet written to followers' feed caches immediately
  → Feed is pre-built, instant to load

Celebrity users (> 1M followers): fan-out on read
  → Tweet NOT pushed to follower feeds
  → When a user loads their feed, the system injects celebrity tweets in real-time
  → "Is any followed celebrity user? If yes, fetch their latest tweets separately"

Result:
  - Normal users: instant feed load from cache
  - Celebrity followers: feed stitched from cache (normal users) + real-time (celebrities)
  - Celebrity tweet reaches 80M feeds in minutes, not milliseconds — but that is acceptable
```

This hybrid approach is now the standard for social feed systems at scale.

---

## 4. Netflix — Video Delivery at Scale

Netflix streams 250M subscribers × ~2 hours/day = 500M hours of video per day. The core challenge is delivering large video files to users worldwide with minimal buffering.

### Open Connect — Netflix's Custom CDN

Netflix built their own CDN (Open Connect Appliances — OCA) because commercial CDNs were too expensive and insufficiently customizable.

```
Netflix architecture:
  Content Catalog (AWS) → decides what to serve from where
  Open Connect Appliances (ISP data centers globally) → serve actual video bytes

  When you watch Netflix:
  1. Netflix API (AWS): determines which OCA server near you has the content
  2. Your player connects directly to the OCA server
  3. Video bytes flow from OCA → your device (never through AWS)

  AWS handles: user auth, content decisions, billing, recommendations
  OCA handles: all video byte delivery (>90% of Netflix traffic)
```

Netflix pre-positions popular content on OCAs during off-peak hours. The most popular 10% of content accounts for 90% of views — pre-positioning that 10% handles most load.

### Adaptive Bitrate Streaming (ABR)

```
Same video stored in multiple quality tiers:
  4K: 20 Mbps bitrate
  1080p: 8 Mbps
  720p: 4 Mbps
  480p: 1.5 Mbps
  360p: 0.5 Mbps

Your player starts at 360p (fast initial load)
Player monitors network speed:
  Good bandwidth → switch to higher quality
  Bad bandwidth → switch to lower quality
  Segment-by-segment (every 4 seconds) — seamless quality switching

User never buffers; quality fluctuates instead.
```

### Chaos Engineering at Netflix Scale

Netflix famously runs "Chaos Monkey" — a tool that randomly terminates production servers. This forces engineers to build systems that survive individual component failures.

At 100M subscribers, you cannot wait for failures to expose weaknesses. You proactively introduce failures to find them first.

---

## 5. Uber — Real-Time Location at Scale

Uber's core challenge: millions of drivers sending GPS location every 4 seconds, and matching algorithms that need to find the nearest driver in real time.

```
Scale:
  5M trips/day
  500K active drivers at peak
  Each driver: 1 location update every 4 seconds
  Total: 500K / 4 = 125,000 location writes/second
```

### Geospatial indexing

Uber uses **geohash** to index locations:

```
Geohash: divides the world into a grid
  Each cell has a string ID (e.g. "tdr1ue")
  Adjacent cells have similar prefixes
  Precision level determines cell size

Driver location: lat=19.0760, lng=72.8777 → geohash "te7ud9"

Finding nearby drivers:
  1. Compute geohash of rider location
  2. Find all drivers whose geohash matches the prefix (same cell)
  3. Also check adjacent cells (8 neighbors)
  → Sub-millisecond lookup vs scanning all 500K driver locations
```

### Supply and demand at scale

Uber maintains an in-memory map of driver locations using a ring (consistent hash ring) distributed across many servers. Each region of the map is owned by a specific server — location updates and queries for that region go to that server.

```
City divided into cells:
  Server A owns: downtown, financial district
  Server B owns: airport, suburban north
  ...

Driver in downtown sends location update:
  → Routes to Server A
  → Server A updates in-memory map
  → Rider in downtown requests a driver:
  → Routes to Server A
  → Server A returns nearest available drivers
```

This avoids distributed coordination for the common case (rider and nearby driver are in the same cell).

---

## 6. The Global Consistency Problem

At hyper scale with multi-region deployments, global consistency is the hardest unsolved problem.

```
Scenario: User changes password in India region
  → Written to India replica
  → Must reach US primary (async replication: 180ms lag)
  → During those 180ms: user can log in with OLD password in US
  → Security violation

Options:
  A) Route all auth operations to US primary (adds 180ms latency for Indian users)
  B) Use synchronous cross-region replication (adds 180ms to every password change)
  C) Invalidate auth sessions globally via a separate fast path (distributed cache invalidation)
```

Most companies choose option C — a fast invalidation path (Redis Pub/Sub, or a global KV store like DynamoDB Global Tables) for security-critical state, while accepting eventual consistency for non-critical data.

```python
def change_password(user_id: int, new_hash: str):
    # Write to local DB (will replicate to other regions eventually)
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", new_hash, user_id)

    # Immediately invalidate ALL sessions globally across all regions
    # (uses a fast cross-region invalidation mechanism)
    for region in ALL_REGIONS:
        regional_redis[region].delete(f"session:user:{user_id}:*")

    # Set a flag in global KV store: "this user must re-authenticate"
    global_kv.set(f"force_reauth:{user_id}", "1", ex=300)
```

---

## 7. The Human Problems at Hyper Scale

Beyond the technical challenges, hyper-scale companies face organizational problems that are harder to solve than any system design question.

**The knowledge problem:** No single engineer understands the entire system. Critical institutional knowledge lives in people's heads. An engineer leaves and takes irreplaceable knowledge with them.

**Solution:** Extensive internal documentation, architecture review boards, incident post-mortems written publicly, runbooks for every operational procedure.

**The coordination problem:** 500 engineers, 50 teams, all changing systems simultaneously. A change to the notification service breaks the payment service because of an undocumented shared dependency.

**Solution:** Strict API contracts, service ownership registries (who owns what), architectural decision records (ADRs), change freeze periods before major events.

**The on-call burnout problem:** 350,000 RPS means when something goes wrong, thousands of users are affected per second. On-call engineers get paged constantly.

**Solution:** SLO-based alerting (alert on burn rate, not individual errors), runbooks that empower any engineer to fix common issues, blameless post-mortems, rotation schedules that limit on-call frequency.

---

## Summary

- At 10M+ DAU, the engineering problems are database horizontal sharding, global consistency, custom infrastructure, and organizational coordination
- Instagram's logical sharding (2000 logical shards mapped to physical shards) enables migration without rehashing
- Twitter's hybrid fan-out: push for normal users, pull for celebrities — the standard pattern for social feeds at scale
- Netflix's Open Connect: custom CDN placed in ISP data centers, pre-positioned popular content, adaptive bitrate streaming
- Uber's geohash: divides map into cells for sub-millisecond nearest-driver lookup at 125K updates/second
- Global consistency: use fast invalidation paths for security-critical data, accept eventual consistency for non-critical data
- The human problems at hyper scale — knowledge silos, coordination, on-call burnout — are often harder than the technical ones

---

## ✅ Chapter 7 Complete

Chapter 7 walked through every scale tier with concrete architecture decisions:

- **7.1** 1K–10K: Reliability over scale. Two servers, managed DB, stateless design, basic monitoring.
- **7.2** 10K–100K: Fix indexes first, add PgBouncer, add Redis caching, add read replica.
- **7.3** 100K–1M: Move heavy work async, add message queue, introduce API gateway, consider first service extraction.
- **7.4** 1M–10M: Write scaling (functional sharding), multi-region, cache at every layer, distributed rate limiting.
- **7.5** 10M+: Horizontal sharding, custom CDN, geospatial indexing, global consistency, organizational scale.

---

> Next: [Chapter 8 — Core Tradeoffs](../chapter-8/lesson-8.1-consistency-vs-availability.md)