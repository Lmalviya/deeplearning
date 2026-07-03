# Lesson 4.4 — Cache Stampede and Thundering Herd

> **Chapter 4 — The Caching Layer**
> Previous: [Lesson 4.3 — Cache Invalidation](./lesson-4.3-cache-invalidation.md) | Next: [Lesson 4.5 — Eviction Policies](./lesson-4.5-eviction-policies.md)

---

## What this lesson covers

- What cache stampede is and why it brings down production systems
- The four solutions: mutex lock, probabilistic early expiration, background refresh, request coalescing
- Hot key problem — a single cache key receiving millions of requests
- How to design around stampedes from the start

---

## 1. The Cache Stampede — What Happens

A cache stampede (also called thundering herd) occurs when a popular cache key expires and many concurrent requests all miss the cache simultaneously, all rush to the database, and all try to repopulate the same key.

```
Normal operation:
  Key "homepage:featured" is cached (TTL: 60 seconds)
  1,000 requests/second → all hit cache → fast, DB untouched

At T=0 seconds: TTL expires, key is deleted

T=0 to T=2 seconds (DB query time):
  1,000 requests/second × 2 seconds = 2,000 concurrent requests
  All 2,000 miss cache simultaneously
  All 2,000 query the DB at the same time → DB gets 2,000× normal load
  DB may time out, crash, or cascade failure to the entire system

T=2 seconds: First query completes, repopulates cache
  Remaining requests start hitting cache again
  But 2,000 requests have already hammered the DB
```

This is not theoretical. Many production incidents are caused by a stampede on a popular key that expires during high traffic.

---

## 2. Solution 1 — Mutex Lock (Simple, Common)

Only one request is allowed to regenerate the cache. All other requests wait.

```python
import redis
import time

def get_homepage_featured(redis_client, db):
    cache_key = "homepage:featured"
    lock_key = "lock:homepage:featured"

    # Step 1: try cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Step 2: cache miss — try to acquire lock
    # NX = only set if not exists, EX = expire in 10 seconds
    lock_acquired = redis_client.set(lock_key, "1", nx=True, ex=10)

    if lock_acquired:
        try:
            # This thread won the lock — regenerate cache
            data = db.query("SELECT * FROM featured_products LIMIT 10")
            redis_client.setex(cache_key, 60, json.dumps(data))
            return data
        finally:
            redis_client.delete(lock_key)
    else:
        # Another thread is regenerating — wait and retry
        time.sleep(0.1)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        # If still not cached after wait, fall through to DB
        return db.query("SELECT * FROM featured_products LIMIT 10")
```

**What this achieves:** Only one DB query runs during the stampede window. All other requests wait 100ms then get the cached result.

**Tradeoff:**
- All waiting requests experience a ~100ms delay
- If the lock holder crashes, the lock expires after 10 seconds and another thread takes over
- Complex to implement correctly (lock expiry, crash recovery, retry logic)

---

## 3. Solution 2 — Probabilistic Early Expiration (XFetch)

Instead of waiting for the key to expire, some requests proactively refresh the cache before it expires — before a stampede can start.

The algorithm (from the XFetch paper): as a key approaches its expiry, requests have an increasing probability of deciding to refresh it early.

```python
import math
import random
import time

def get_with_early_refresh(redis_client, db, key: str, ttl: int, beta: float = 1.0):
    """
    beta: higher = more aggressive early refresh (typically 1.0)
    """
    cached_raw = redis_client.get(key)

    if cached_raw:
        cached = json.loads(cached_raw)
        remaining_ttl = redis_client.ttl(key)  # seconds left

        # Probabilistic check: should we refresh early?
        # Probability increases as TTL decreases
        # recompute_time = how long the DB query takes (estimated)
        recompute_time = cached.get('_recompute_time', 0.1)
        gap = remaining_ttl - recompute_time * beta * math.log(random.random())

        if gap > 0:
            return cached['data']  # not yet time to refresh — return cached

        # Decided to refresh early — fall through to DB query below

    # Cache miss or early refresh decision
    start = time.time()
    data = db.query("SELECT * FROM featured_products LIMIT 10")
    recompute_time = time.time() - start

    # Store data along with recompute time for the formula
    payload = json.dumps({'data': data, '_recompute_time': recompute_time})
    redis_client.setex(key, ttl, payload)
    return data
```

**What this achieves:** Individual requests randomly start refreshing the cache before it expires. The earlier, the higher the chance of refreshing. By the time the key actually expires, it has likely already been refreshed — no stampede.

**Tradeoff:** Slightly wasteful (some redundant DB queries), but gracefully avoids the stampede without locking.

---

## 4. Solution 3 — Background Refresh

Separate the cache TTL from the "should I refresh?" decision. The cache key never expires — instead, a background job refreshes it before it would go stale.

```python
# Cache entry stores both data and a "refresh_at" timestamp
def get_featured(redis_client, db):
    cached_raw = redis_client.get("homepage:featured")

    if cached_raw:
        cached = json.loads(cached_raw)
        refresh_at = cached['refresh_at']

        if time.time() < refresh_at:
            return cached['data']  # still fresh

        # Stale but still serve it — trigger async refresh
        trigger_background_refresh()  # sends message to worker queue
        return cached['data']  # return stale data immediately (no wait!)

    # Truly no cache — must wait for DB
    return refresh_and_cache(redis_client, db)

def refresh_and_cache(redis_client, db):
    data = db.query("SELECT * FROM featured_products LIMIT 10")
    payload = {
        'data': data,
        'refresh_at': time.time() + 55  # refresh in 55 seconds (before 60s TTL)
    }
    redis_client.setex("homepage:featured", 120, json.dumps(payload))  # 2min max TTL as safety
    return data
```

**What this achieves:** Users always get an immediate response (either fresh or slightly stale). The DB is queried by a background worker, not by a user-facing request. No stampede is possible because user requests never wait for DB regeneration.

**Tradeoff:** Users may see slightly stale data during the refresh window. Background worker adds infrastructure complexity. Does not work for user-specific cached data (each user's data would need its own refresh job).

**Best for:** Public, shared cached data — homepage content, trending lists, site-wide configuration.

---

## 5. Solution 4 — Request Coalescing

Collapse multiple simultaneous requests for the same key into a single DB query. All waiting requests share the single result.

```python
import asyncio

# In-flight requests for cache misses
in_flight: dict = {}

async def get_user(user_id: int):
    cache_key = f"user:{user_id}"

    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Check if another request is already fetching this
    if cache_key in in_flight:
        # Wait for the in-flight request to complete
        return await in_flight[cache_key]

    # First request for this key — create a future others can await
    future = asyncio.Future()
    in_flight[cache_key] = future

    try:
        data = await db.query_async("SELECT * FROM users WHERE id = ?", user_id)
        redis.setex(cache_key, 300, json.dumps(data))
        future.set_result(data)
        return data
    except Exception as e:
        future.set_exception(e)
        raise
    finally:
        del in_flight[cache_key]
```

**What this achieves:** 1,000 concurrent requests for the same missing key result in 1 DB query. All 1,000 await the same result.

**Tradeoff:** Only works within a single process (in-flight dict is local). For multiple app servers, need a distributed version using Redis locks. Adds complexity.

---

## 6. The Hot Key Problem

A different but related problem: a single cache key that receives millions of requests per second.

```
Example: A post by a celebrity goes viral
  "post:12345" → 500,000 requests/second

Redis is single-threaded for commands.
At ~1M operations/second, a single Redis node can become the bottleneck.
A single hot key can saturate Redis's CPU even though the key is always cached.
```

### Solutions for hot keys

**Solution A — Local in-process cache**

Cache the hot key in the application's own memory, not just in Redis. Each app server holds its own copy.

```python
from cachetools import TTLCache

# Local in-process cache (per app server)
local_cache = TTLCache(maxsize=1000, ttl=5)  # 5 second local TTL

def get_post(post_id: int):
    # L1: check local in-process cache (zero network latency)
    if post_id in local_cache:
        return local_cache[post_id]

    # L2: check Redis
    cached = redis.get(f"post:{post_id}")
    if cached:
        data = json.loads(cached)
        local_cache[post_id] = data  # populate local cache
        return data

    # L3: DB
    data = db.query("SELECT * FROM posts WHERE id = ?", post_id)
    redis.setex(f"post:{post_id}", 300, json.dumps(data))
    local_cache[post_id] = data
    return data
```

With 10 app servers and a 5-second local TTL, Redis receives at most `(10 servers × 60 seconds) / 5 second TTL = 120` requests per minute for a hot key — regardless of user traffic.

**Solution B — Key replication / sharding within Redis**

Create multiple copies of the hot key across Redis and randomly distribute reads:

```python
import random

HOT_KEY_REPLICAS = 10

def get_hot_post(post_id: int):
    replica = random.randint(0, HOT_KEY_REPLICAS - 1)
    cache_key = f"post:{post_id}:replica:{replica}"

    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    data = db.query("SELECT * FROM posts WHERE id = ?", post_id)

    # Populate all replicas
    pipeline = redis.pipeline()
    for i in range(HOT_KEY_REPLICAS):
        pipeline.setex(f"post:{post_id}:replica:{i}", 300, json.dumps(data))
    pipeline.execute()
    return data
```

Now 500K requests/second are spread across 10 replica keys = 50K requests/key. Manageable for Redis.

---

## 7. Designing Around Stampedes

The best solution is to avoid stampedes by design:

**Use staggered TTLs:** Add random jitter to TTLs so not all keys expire at the same time (covered in Lesson 4.1).

**Avoid shared expensive queries:** If your homepage query takes 2 seconds and is shared by all users, a stampede on it is catastrophic. Break it into smaller, less expensive queries with shorter TTLs.

**Pre-warm the cache:** Before deployment, run a script to populate the cache with likely-to-be-requested data. Prevents cold-start stampedes after a deploy.

**Use background refresh for critical shared keys:** The homepage, trending lists, and site configuration should be refreshed by background jobs, never by user requests.

---

## Summary

- Cache stampede: a popular key expires, thousands of requests simultaneously miss cache, all hit the DB, system crashes
- **Mutex lock:** One request regenerates, others wait. Simple, but adds latency for waiting requests.
- **Probabilistic early expiration (XFetch):** Requests randomly refresh before expiry. No locking needed. Graceful.
- **Background refresh:** Background job refreshes cache before expiry. Users always get instant response (may be slightly stale). Best for public shared data.
- **Request coalescing:** Multiple concurrent misses share one DB query. Works per-process.
- **Hot key problem:** A single key receiving millions of requests saturates Redis. Fix with local in-process cache or key replication.
- Design to avoid stampedes: staggered TTLs, pre-warming, background refresh for critical keys.

---

## ⚠️ Common Mistakes

- No stampede protection on high-traffic shared cache keys — the system works fine until a key expires at peak traffic
- Mutex lock with no timeout — if the lock holder crashes without releasing the lock, all waiting requests block forever
- Background refresh that re-uses the user request thread — defeats the purpose; refresh must be truly async
- Setting the same TTL for all keys of the same type — synchronized expiry = synchronized stampede
- Ignoring hot keys in Redis — 500K req/sec on one key saturates Redis regardless of caching strategy

---

> Next: [Lesson 4.5 — Eviction Policies](./lesson-4.5-eviction-policies.md)