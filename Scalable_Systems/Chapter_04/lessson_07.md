# Lesson 4.7 — Distributed Caching Problems

> **Chapter 4 — The Caching Layer**
> Previous: [Lesson 4.6 — Redis Deep Dive](./lesson-4.6-redis-deep-dive.md) | Next: [Chapter 5 — Async Processing and Message Queues](../chapter-5/lesson-5.1-why-async.md)

---

## What this lesson covers

- Cache consistency across multiple app servers
- The cold start problem after deploys and restarts
- Regional cache coherence in multi-region setups
- Cache poisoning — security considerations
- Monitoring your cache in production

---

## 1. Cache Consistency Across Multiple App Servers

With one app server, cache management is simple. With 10 app servers all reading and writing the same Redis instance, subtle consistency issues emerge.

### The stale local cache problem

If app servers maintain any local (in-process) cache in addition to Redis, they can diverge:

```
App Server 1 updates user:42 in DB
App Server 1 deletes user:42 from Redis ✅
App Server 1 local cache still has old user:42 ❌
App Server 2 local cache still has old user:42 ❌

Next 5 seconds: all requests routed to Server 1 or 2 get stale profile
After 5s local TTL: local cache expires, next read hits Redis → fresh
```

**Solution: broadcast local cache invalidations via Redis Pub/Sub**

```python
# When a key is updated:
def invalidate(key: str):
    redis.delete(key)                          # invalidate shared Redis cache
    redis.publish("cache:invalidate", key)     # tell all servers to clear local cache

# Each app server subscribes on startup:
def start_cache_invalidation_listener():
    pubsub = redis.pubsub()
    pubsub.subscribe("cache:invalidate")
    for message in pubsub.listen():
        if message['type'] == 'message':
            key = message['data']
            local_cache.delete(key)            # clear from local cache
```

Now all app servers clear their local cache within milliseconds of a write.

---

## 2. The Cold Start Problem

After a fresh deployment, Redis restart, or Redis Cluster rebalance, the cache is empty. Every request is a cache miss. All traffic hits the database simultaneously.

This is a stampede at the system level — not just one key expiring, but the entire cache gone.

### The impact

```
Normal operation (warm cache):
  10,000 req/sec → 95% cache hit → 500 DB queries/sec

After deploy (cold cache):
  10,000 req/sec → 0% cache hit → 10,000 DB queries/sec
  DB was sized for 500 queries/sec — it is now 20× overloaded
  DB starts timing out → all requests fail → full outage
```

### Solution 1 — Cache pre-warming

Before taking traffic, run a script that populates the cache with likely-to-be-requested data:

```python
def warm_cache():
    # Pre-populate top 1000 most-active users
    active_users = db.query("""
        SELECT user_id FROM user_activity
        WHERE last_active > NOW() - INTERVAL '24 hours'
        ORDER BY activity_count DESC
        LIMIT 1000
    """)
    for user_id in active_users:
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)
        redis.setex(f"user:{user_id}", 300, json.dumps(user))

    # Pre-populate homepage content
    featured = db.query("SELECT * FROM featured_products LIMIT 20")
    redis.setex("homepage:featured", 300, json.dumps(featured))

    print(f"Cache warmed with {len(active_users)} users")

# Run before opening traffic to new deployment
warm_cache()
```

**Limitation:** You can only pre-warm data you know will be requested. Long-tail requests will still miss.

### Solution 2 — Traffic ramping

After a deploy, gradually increase the traffic sent to the new servers:

```
T=0: 5% of traffic → new servers (cache 5% warm)
T=5min: 25% of traffic → new servers (cache warmer)
T=10min: 50% of traffic → new servers
T=15min: 100% of traffic → new servers (cache reasonably warm)
```

The old servers (still taking 95% of traffic) protect the DB while the new servers warm their cache.

### Solution 3 — Rate limit DB during cold start

Apply aggressive rate limiting to DB queries during cold start, accepting that some requests will be slower:

```python
cold_start_db_semaphore = asyncio.Semaphore(50)  # max 50 concurrent DB queries

async def get_user_cold_start_safe(user_id):
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)

    async with cold_start_db_semaphore:  # limit concurrent DB hits
        # Check cache again inside semaphore (another request may have populated it)
        cached = redis.get(f"user:{user_id}")
        if cached:
            return json.loads(cached)
        return await fetch_from_db(user_id)
```

---

## 3. Multi-Region Cache Coherence

In a multi-region setup (servers in US, Europe, India), each region typically has its own Redis instance for low latency. This creates a cache coherence challenge.

```
Architecture:
  Region US:     App Servers + Redis US + DB Primary (US)
  Region EU:     App Servers + Redis EU
  Region India:  App Servers + Redis India

DB: Primary in US, read replicas in EU and India
```

### The problem

```
User in India updates their profile
Write → DB Primary (US) → replication to India replica (lag: ~50ms)
Invalidation → Redis India is cleared

But what about Redis EU and Redis US?
A US user might load Alice's profile from Redis US → stale data
Until the US cache TTL expires, US users see the old name
```

For most social data, this cross-region staleness (a few minutes) is acceptable. For security-sensitive data (permissions, auth tokens), it is not.

### Solution 1 — Cross-region cache invalidation event

```
Write in India:
  1. Update DB Primary (US)
  2. Delete Redis India
  3. Publish "invalidate:user:42" to global Kafka topic

EU service (subscriber):
  4. Receive invalidation event
  5. Delete user:42 from Redis EU

US service (subscriber):
  6. Receive invalidation event
  7. Delete user:42 from Redis US

All regions now serve fresh data on next request
```

**Latency of propagation:** Kafka cross-region delivery is typically 100–500ms. During this window, other regions may serve stale data.

### Solution 2 — Accept regional staleness with TTL

For non-critical data, set a short TTL (1–5 minutes) and accept that other regions may be 1–5 minutes stale. Simple and correct for most use cases.

### Solution 3 — Always-primary reads for critical data

For security-sensitive data (permissions, payment methods, auth tokens), always read from the US primary DB directly — skip regional caches:

```python
def get_user_permissions(user_id: int, is_critical: bool = False):
    if is_critical:
        # Always read from primary — never cache security data across regions
        return primary_db.query("SELECT permissions FROM users WHERE id = ?", user_id)

    # Non-critical: use regional cache
    cached = regional_redis.get(f"permissions:{user_id}")
    if cached:
        return json.loads(cached)
    permissions = read_replica.query("SELECT permissions FROM users WHERE id = ?", user_id)
    regional_redis.setex(f"permissions:{user_id}", 60, json.dumps(permissions))
    return permissions
```

---

## 4. Cache Poisoning — Security Consideration

Cache poisoning occurs when an attacker manipulates the cache to serve malicious or incorrect content to other users.

### How it happens

**Shared cache keys for different users:**
```python
# BUG: all users share the same cache key for "my profile"
def get_my_profile(request):
    user_id = request.user_id
    # If the attacker makes the key not include user_id:
    cached = redis.get("my_profile")  # WRONG: same key for all users
    if not cached:
        profile = db.query("SELECT * FROM users WHERE id = ?", user_id)
        redis.setex("my_profile", 300, json.dumps(profile))
    return json.loads(redis.get("my_profile"))

# If Attacker (user:99) loads the page first → cache key = attacker's profile
# When Alice (user:42) loads the page → gets attacker's profile from cache!
```

**Fix:** Always include the user_id (and any other dimension that affects the content) in the cache key:

```python
cache_key = f"profile:{user_id}"  # user-specific key
```

**CDN-level poisoning:** Be careful caching responses at the CDN for content that should be user-specific. CDNs cache by URL — if your URL does not vary by user and the response contains user data, all users see the first user's data.

```
URL: GET /api/me
Response: {"name": "Alice", "email": "alice@example.com"}

If CDN caches this response: all users see Alice's data ← catastrophic
```

Fix: use `Cache-Control: private` for user-specific responses (tells CDN not to cache) or `Vary: Authorization` (tells CDN to cache separately per Authorization header value).

---

## 5. Monitoring Your Cache in Production

Without monitoring, cache problems are invisible until they cause outages.

### Essential metrics to track

```bash
# 1. Hit ratio (most important)
redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"
# hit_ratio = keyspace_hits / (keyspace_hits + keyspace_misses)
# Alert if < 80%

# 2. Memory usage
redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human"
# Alert if > 85% of maxmemory

# 3. Eviction rate
redis-cli INFO stats | grep evicted_keys
# Alert if rising (means cache is too small)

# 4. Connection count
redis-cli INFO clients | grep connected_clients
# Alert if approaching maxclients (default 10000)

# 5. Latency
redis-cli --latency  # measures round-trip latency to Redis
redis-cli --latency-history  # latency over time
# Alert if p99 > 5ms (something is wrong)

# 6. Slow commands
redis-cli SLOWLOG GET 10  # last 10 slow commands (default threshold: 10ms)
```

### Dashboard: the four caching KPIs

| KPI | Target | Action if missed |
|-----|--------|-----------------|
| Hit ratio | > 90% | Increase TTL, increase maxmemory, audit cache key design |
| Memory utilization | 70–85% | Below 70% = wasted money; above 85% = risk of eviction |
| p99 latency | < 2ms | Check for slow commands, network issues, memory swap |
| Eviction rate | Near 0 | Increase maxmemory or change eviction policy |

---

## ✅ Chapter 4 Complete

Chapter 4 has covered the full caching layer:

- **4.1** Mental model — cache hit/miss/ratio, the five caching layers, Redis vs Memcached, key design, TTL
- **4.2** Writing strategies — cache-aside, write-through, write-back, read-through, when to use each
- **4.3** Cache invalidation — TTL, event-driven, cache tags, versioned keys, race conditions
- **4.4** Cache stampede — thundering herd, mutex lock, probabilistic early expiration, background refresh, hot keys
- **4.5** Eviction policies — LRU, LFU, volatile vs allkeys, memory sizing
- **4.6** Redis deep dive — all data structures with use cases, Sentinel vs Cluster, Pub/Sub, persistence
- **4.7** Distributed caching problems — multi-server consistency, cold start, multi-region coherence, cache poisoning, monitoring

---

> Next: [Chapter 5 — Async Processing and Message Queues](../chapter-5/lesson-5.1-why-async.md)