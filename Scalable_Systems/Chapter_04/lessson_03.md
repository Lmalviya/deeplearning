# Lesson 4.3 — Cache Invalidation — The Hard Problem

> **Chapter 4 — The Caching Layer**
> Previous: [Lesson 4.2 — Cache Writing Strategies](./lesson-4.2-cache-writing-strategies.md) | Next: [Lesson 4.4 — Cache Stampede and Thundering Herd](./lesson-4.4-cache-stampede.md)

---

## What this lesson covers

- Why cache invalidation is genuinely hard
- The three invalidation approaches: TTL, event-driven, cache tags
- The race conditions that cause stale data bugs
- Strategies for complex invalidation scenarios (related data, cascading invalidation)
- The "two hard problems" in computer science and what they mean in practice

---

## 1. Why Cache Invalidation is Hard

Phil Karlton's famous quote: *"There are only two hard things in computer science: cache invalidation and naming things."*

Cache invalidation is hard because it is a **distributed consistency problem**. You have two copies of data (DB and cache), and you need them to agree. The challenge:

- The DB write and cache invalidation happen in separate operations — a crash between them leaves them inconsistent
- Multiple app servers may be reading and writing simultaneously — race conditions are possible
- Complex data models have cascading dependencies — invalidating one item may require invalidating dozens of related items
- You cannot invalidate what you do not know is cached

---

## 2. Approach 1 — TTL-Based Expiry (Simplest)

Let items expire naturally. After the TTL, the next request re-fetches from DB.

```python
redis.setex("user:42", ttl=300, value=json.dumps(user))
# After 5 minutes, the key expires automatically
# Next request misses, fetches from DB, re-caches
```

**This is the simplest invalidation strategy and is correct for many use cases.**

### When TTL is sufficient

- **Acceptable staleness window:** If "user profile might be 5 minutes stale" is fine for your use case, TTL alone is enough
- **Low write frequency:** If data changes once a day, a 5-minute TTL is essentially always fresh
- **High read volume:** The 5-minute cache dramatically reduces DB load even with occasional staleness

### When TTL is not sufficient

- User-visible updates that must be reflected immediately ("I just changed my profile picture — why is the old one still showing?")
- Inventory counts — showing "5 in stock" when actually 0 causes overselling
- Security-sensitive data — a revoked session token should be invalid immediately, not in 5 minutes

---

## 3. Approach 2 — Event-Driven Invalidation (Delete on Write)

The moment data changes in the DB, delete (or update) the corresponding cache key.

```python
def update_user_profile(user_id: int, data: dict):
    # 1. Update DB (source of truth)
    db.execute("UPDATE users SET name=%s, email=%s WHERE id=%s",
               data['name'], data['email'], user_id)

    # 2. Invalidate cache immediately
    redis.delete(f"user:{user_id}")

    # Next read will be a cache miss → re-fetch from DB → re-cache
```

**This is the standard approach** for data that must be fresh after a write.

### The race condition in event-driven invalidation

Even with delete-on-write, there is a subtle race condition:

```
Timeline:
  T=0: Thread A reads user:42 from DB (cache miss)
  T=1: Thread B writes new data to DB for user:42
  T=2: Thread B deletes cache key "user:42"
  T=3: Thread A writes OLD data to cache ("user:42" = old name)
  T=4: All subsequent reads return STALE data until TTL expires

Thread A's write at T=3 overwrites the invalidation done at T=2.
```

This is the **read-during-write race condition**. It is rare but real.

### Solution — Short TTL as a safety net

Even with event-driven invalidation, always set a TTL. This bounds the maximum staleness window in case the race condition occurs or an invalidation is missed:

```python
# Event-driven invalidation as primary mechanism
redis.delete(f"user:{user_id}")

# Short TTL on re-cache as safety net
redis.setex(f"user:{user_id}", ttl=60, value=json.dumps(user))  # 60 seconds max staleness
```

The delete handles the common case. The TTL handles edge cases.

---

## 4. The Problem of Cascading Invalidation

Real data has relationships. When one piece of data changes, multiple cached items may be stale.

```
User Alice (user:42) updates her name

What needs to be invalidated?
  - user:42                         ← her profile
  - feed:follower:99                ← follower's feed shows her name
  - feed:follower:100               ← another follower
  - search:results:"Alice"          ← search results showing her old name
  - comments:post:55:user:42        ← comments she made showing her old name
  - activity:user:42                ← her activity feed
  ... potentially thousands of keys
```

Invalidating everything related to a user change can be enormous.

### Solution 1 — Selective invalidation (invalidate what you know)

Track which cache keys are affected by a given write and invalidate only those.

```python
def update_user_profile(user_id: int, data: dict):
    db.execute("UPDATE users SET name=%s WHERE id=%s", data['name'], user_id)

    # Invalidate known related keys
    pipeline = redis.pipeline()
    pipeline.delete(f"user:{user_id}")
    pipeline.delete(f"user:{user_id}:public_profile")
    # Do NOT try to invalidate follower feeds — too many, handle with TTL
    pipeline.execute()
```

Invalidate the keys you know and accept TTL-based staleness for the rest.

### Solution 2 — Cache tags (group invalidation)

Associate cache keys with tags. When a tag is invalidated, all keys with that tag expire.

This is not built into Redis natively but can be implemented:

```python
def cache_set_with_tags(key: str, value, ttl: int, tags: list):
    pipeline = redis.pipeline()
    pipeline.setex(key, ttl, value)
    for tag in tags:
        pipeline.sadd(f"tag:{tag}", key)       # track keys per tag
        pipeline.expire(f"tag:{tag}", ttl + 60) # tag set expires after items
    pipeline.execute()

def invalidate_tag(tag: str):
    keys = redis.smembers(f"tag:{tag}")
    if keys:
        pipeline = redis.pipeline()
        for key in keys:
            pipeline.delete(key)
        pipeline.delete(f"tag:{tag}")
        pipeline.execute()

# Usage
cache_set_with_tags(
    key=f"user:{user_id}:profile",
    value=json.dumps(user),
    ttl=300,
    tags=[f"user:{user_id}", "profiles"]  # tag with user ID
)

# On user update: invalidate everything tagged with user:42
invalidate_tag(f"user:{user_id}")
# This deletes all cached items for this user at once
```

CDNs (Cloudflare, Fastly) have native cache tag support — you can tag CDN responses and purge all content for a tag with one API call.

### Solution 3 — Versioned cache keys

Instead of invalidating a key, change the key itself. The old key becomes orphaned and expires via TTL.

```python
# Store the current version in Redis
def get_user_version(user_id: int) -> int:
    version = redis.get(f"user:{user_id}:version")
    return int(version) if version else 1

def get_user(user_id: int) -> dict:
    version = get_user_version(user_id)
    cache_key = f"user:{user_id}:v{version}"

    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)

    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    redis.setex(cache_key, 300, json.dumps(user))
    return user

def update_user(user_id: int, data: dict):
    db.execute("UPDATE users ...")
    # Bump version — old cache key becomes orphaned, new key forces miss
    redis.incr(f"user:{user_id}:version")
    # Old key "user:42:v3" is now orphaned — will expire via TTL
    # Next read uses "user:42:v4" which is a miss → fetches fresh data
```

**Advantage:** No explicit deletion needed. Changing the version atomically points all readers to a new key.
**Disadvantage:** Old keys accumulate until their TTL expires (memory overhead).

---

## 5. The Double-Delete Pattern

To close the race condition window from Section 3, some teams use a double-delete:

```python
def update_user(user_id: int, data: dict):
    # First delete — before the write (handles any concurrent reads that
    # would otherwise overwrite the cache with old data after our write)
    redis.delete(f"user:{user_id}")

    # DB write
    db.execute("UPDATE users SET name=%s WHERE id=%s", data['name'], user_id)

    # Short sleep or async delay — let concurrent reads that started before
    # the first delete finish their cache write
    time.sleep(0.05)  # 50ms

    # Second delete — cleans up any stale writes that snuck in
    redis.delete(f"user:{user_id}")
```

This is a pragmatic solution to the race condition. The 50ms delay means any concurrent read that was in-flight during the write will have completed its cache write — which we then immediately delete.

In practice, for most applications the race condition is rare enough that simple delete-on-write is sufficient. Use double-delete only when you have evidence of stale cache bugs in high-concurrency writes.

---

## 6. Invalidation in Microservices

When service A writes data and service B caches it, invalidation becomes harder — B does not know when A writes.

```
Order Service  →  writes order to DB
Product Service → caches product details (inventory count)
Order Service creates an order that depletes inventory
Product Service cache is stale — still shows old inventory count
```

**Solution: event-driven invalidation via message queue**

```
Order Service:
  1. Write to DB
  2. Publish "inventory_updated" event to Kafka

Product Service (consumer):
  1. Receive "inventory_updated" event
  2. Delete cached product:99 from its Redis cache
  3. Next read re-fetches fresh data
```

This decouples services — Product Service does not need to know about Order Service's implementation, just the events it publishes.

---

## Summary

- Cache invalidation is hard because it is a distributed consistency problem between DB and cache
- TTL-based expiry: simplest, correct for many use cases, but bounded staleness is unavoidable
- Event-driven (delete on write): immediate consistency after writes, but race conditions are possible
- Always set a TTL even with event-driven invalidation — bounds staleness from race conditions
- Cascading invalidation: use cache tags for group invalidation or versioned keys to avoid explicit deletion
- In microservices: publish invalidation events via message queue — services subscribe and clear their own caches
- The double-delete pattern closes the race condition window for high-concurrency write scenarios

---

## ⚠️ Common Mistakes

- Invalidating cache before the DB write commits — if the DB write fails, the cache has been cleared for no reason and the next read re-populates with old data
- No TTL on event-driven invalidation — if an invalidation is missed (server crash, bug), the stale entry lives forever
- Trying to invalidate everything related to a write — the complexity becomes unmanageable; use TTL for distant relationships
- Invalidating from a different service without coordination — service A writes and sends a cache invalidation event, but service B has not yet finished writing related data; the cache re-populates with partially consistent data

---

> Next: [Lesson 4.4 — Cache Stampede and Thundering Herd](./lesson-4.4-cache-stampede.md)